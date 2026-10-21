import math
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path as FilePath

import numpy
import pytest
from numpy.typing import NDArray
from scipy.sparse import coo_array
from scipy.sparse.csgraph import dijkstra

import semita
from semita import _search

# Each step to a neighbor, across, up or diagonally, with its length; the steps back are the same undirected edges.
_NEIGHBOR_STEPS = ((1, 0, 1.0), (0, 1, 1.0), (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)))


def _shortest_distances(
    passable: NDArray[numpy.bool_], start: tuple[int, int], costs: NDArray[numpy.float64] | None = None
) -> NDArray[numpy.float64]:
    """The distances semita should find, or with `costs` the costs, by scipy's Dijkstra over every legal step, a step
    costing its length times the mean of its two tiles' costs: the oracle."""
    if costs is not None:
        passable = passable & numpy.isfinite(costs)
    width, height = passable.shape
    tiles = numpy.arange(width * height).reshape(width, height)
    sources, targets, lengths = [], [], []
    for di, dj, length in _NEIGHBOR_STEPS:
        from_is, to_is = slice(0, width - di), slice(di, width)
        if dj >= 0:
            from_js, to_js = slice(0, height - dj), slice(dj, height)
        else:
            from_js, to_js = slice(-dj, height), slice(0, height + dj)
        open_step = passable[from_is, from_js] & passable[to_is, to_js]
        if di and dj:
            open_step &= passable[to_is, from_js] & passable[from_is, to_js]
        sources.append(tiles[from_is, from_js][open_step])
        targets.append(tiles[to_is, to_js][open_step])
        if costs is None:
            lengths.append(numpy.full(int(open_step.sum()), length))
        else:
            mean_costs = (costs[from_is, from_js] + costs[to_is, to_js])[open_step] / 2
            lengths.append(length * mean_costs)
    steps = coo_array(
        (numpy.concatenate(lengths), (numpy.concatenate(sources), numpy.concatenate(targets))),
        shape=(width * height, width * height),
    )
    distances = dijkstra(steps.tocsr(), directed=False, indices=start[0] * height + start[1])
    return numpy.asarray(distances).reshape(width, height)


def _random_grid(width: int, height: int, blocked: float, seed: int) -> NDArray[numpy.bool_]:
    return numpy.random.default_rng(seed).random((width, height)) >= blocked


def _walled_grid(seed: int) -> NDArray[numpy.bool_]:
    """Rooms: walls every few tiles, each with a few gaps, the kind of grid jump point search turns most in."""
    rng = numpy.random.default_rng(seed)
    passable = numpy.ones((61, 47), dtype=bool)
    for wall in range(6, 61, 9):
        passable[wall, :] = False
        passable[wall, rng.choice(47, size=3, replace=False)] = True
    for wall in range(5, 47, 8):
        passable[:, wall] = False
        passable[rng.choice(61, size=4, replace=False), wall] = True
    return passable


_TEST_GRIDS = {
    "open": numpy.ones((40, 30), dtype=bool),
    "a row": _random_grid(1, 50, 0.1, 1),
    "a column": _random_grid(50, 1, 0.1, 2),
    "sparse": _random_grid(64, 48, 0.15, 3),
    "dense": _random_grid(64, 48, 0.35, 4),
    "mazelike": _random_grid(80, 60, 0.42, 5),
    "rooms": _walled_grid(6),
}


def _tiles_of(passable: NDArray[numpy.bool_], count: int, seed: int) -> list[tuple[int, int]]:
    """`count` passable tiles at random."""
    open_tiles = [(int(i), int(j)) for i, j in numpy.argwhere(passable)]
    return random.Random(seed).choices(open_tiles, k=count)


@pytest.mark.parametrize("name", list(_TEST_GRIDS))
def test_every_distance_is_the_shortest_paths(name: str) -> None:
    passable = _TEST_GRIDS[name]
    for start in _tiles_of(passable, 5, seed=11):
        found = semita.flood_from(passable, start).costs
        expected = _shortest_distances(passable, start)
        assert found.shape == passable.shape
        assert found.dtype == numpy.float64
        assert (numpy.isinf(found) == numpy.isinf(expected)).all()
        reachable = numpy.isfinite(expected)
        assert numpy.abs(found[reachable] - expected[reachable]).max() < 1e-9


@pytest.mark.parametrize("name", list(_TEST_GRIDS))
def test_a_path_is_a_shortest_one_of_legal_steps(name: str) -> None:
    passable = _TEST_GRIDS[name]
    starts, ends = _tiles_of(passable, 40, seed=12), _tiles_of(passable, 40, seed=13)
    for start, end in zip(starts, ends, strict=True):
        expected = _shortest_distances(passable, start)[end]
        path = semita.find_path(passable, start, end)
        if math.isinf(expected):
            assert path is None
            continue
        assert path is not None
        assert abs(path.distance - expected) < 1e-9
        tiles = [(int(i), int(j)) for i, j in path.tiles]
        assert (tiles[0], tiles[-1]) == (start, end)
        walked = 0.0
        for (i, j), (next_i, next_j) in zip(tiles, tiles[1:], strict=False):
            di, dj = next_i - i, next_j - j
            assert max(abs(di), abs(dj)) == 1
            assert passable[next_i, next_j]
            if di and dj:
                assert passable[i + di, j] and passable[i, j + dj], "the path cuts a corner"
            walked += math.sqrt(2) if di and dj else 1.0
        assert abs(walked - path.distance) < 1e-9


def test_the_start_is_a_path_to_itself_even_on_a_blocked_tile() -> None:
    passable = numpy.zeros((3, 3), dtype=bool)
    path = semita.find_path(passable, (1, 1), (1, 1))
    assert path is not None
    assert path.distance == 0.0
    assert path.tiles.tolist() == [[1, 1]]
    distances = semita.flood_from(passable, (1, 1)).costs
    assert distances[1, 1] == 0.0
    assert numpy.isinf(numpy.delete(distances.ravel(), 4)).all()


def test_no_path_leaves_a_blocked_start_or_enters_a_blocked_end() -> None:
    passable = numpy.ones((4, 4), dtype=bool)
    passable[0, 0] = passable[3, 3] = False
    assert semita.find_path(passable, (0, 0), (2, 2)) is None
    assert semita.find_path(passable, (2, 2), (3, 3)) is None
    assert numpy.isinf(semita.flood_from(passable, (2, 2)).costs[3, 3])


def test_no_path_cuts_a_corner() -> None:
    passable = numpy.array([[True, False], [False, True]])
    assert semita.find_path(passable, (0, 0), (1, 1)) is None
    assert numpy.isinf(semita.flood_from(passable, (0, 0)).costs[1, 1])


def test_a_grid_of_numbers_costs_its_values() -> None:
    grid = numpy.array([[7, 7, 7], [0, 0, 7], [7, 7, 7]], dtype=numpy.uint8)
    path = semita.find_path(grid, (0, 0), (2, 0))
    assert path is not None
    # Round the blocked tiles, never diagonally past (1, 1), each tile costing 7.
    assert path.tiles.tolist() == [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2], [2, 1], [2, 0]]
    assert (path.distance, path.cost) == (6.0, 42.0)


def test_a_grid_that_is_not_c_contiguous_is_read_by_its_indices() -> None:
    passable = _TEST_GRIDS["dense"]
    start, end = _tiles_of(passable, 2, seed=14)
    for grid in (passable, _random_costs(passable, seed=14)):
        transposed = numpy.ascontiguousarray(grid.T).T
        assert not transposed.flags.c_contiguous
        flood, same_flood = semita.flood_from(transposed, start), semita.flood_from(grid, start)
        assert numpy.array_equal(flood.costs, same_flood.costs)
        assert numpy.array_equal(flood.distances, same_flood.distances)
        path, same = semita.find_path(transposed, start, end), semita.find_path(grid, start, end)
        assert (path is None) == (same is None)
        if path is not None and same is not None:
            assert numpy.array_equal(path.tiles, same.tiles)


def test_a_paths_tiles_are_read_only() -> None:
    path = semita.find_path(numpy.ones((3, 3), dtype=bool), (0, 0), (2, 2))
    assert path is not None
    with pytest.raises(ValueError, match="read-only"):
        path.tiles[0, 0] = 1


@pytest.mark.parametrize("tile", [(5, 0), (0, 4), (-1, 0), (0, -1)])
def test_a_tile_outside_the_grid_raises(tile: tuple[int, int]) -> None:
    passable = numpy.ones((5, 4), dtype=bool)
    with pytest.raises(IndexError, match=re.escape(f"the start {tile} is outside the grid's 5 by 4 tiles")):
        semita.flood_from(passable, tile)
    with pytest.raises(IndexError, match=re.escape(f"the end {tile} is outside")):
        semita.find_path(passable, (0, 0), tile)


@pytest.mark.parametrize("dtype", [numpy.int8, numpy.uint16, numpy.int32, numpy.int64, numpy.float32, numpy.float64])
def test_a_grid_whose_tiles_cost_the_same_finds_what_a_grid_of_bools_finds(dtype: type[numpy.number]) -> None:
    passable = _TEST_GRIDS["dense"]
    grid = (passable * 3).astype(dtype)
    start, end = _tiles_of(passable, 2, seed=15)
    path, same = semita.find_path(grid, start, end), semita.find_path(passable, start, end)
    assert (path is None) == (same is None)
    if path is not None and same is not None:
        assert numpy.array_equal(path.tiles, same.tiles)
        assert (path.distance, path.cost) == (same.distance, 3 * same.cost)
    found, expected = semita.flood_from(grid, start).costs, semita.flood_from(passable, start).costs
    reachable = numpy.isfinite(expected)
    assert (numpy.isinf(found) == ~reachable).all()
    assert numpy.abs(found[reachable] - 3 * expected[reachable]).max() < 1e-9


def test_a_grid_given_as_nested_lists_is_read_as_numbers() -> None:
    path = semita.find_path([[1, 1, 1], [0, 0, 1], [1, 1, 1]], (0, 0), (2, 0))
    assert path is not None
    assert (path.distance, path.cost) == (6.0, 6.0)


def test_zero_and_inf_block_a_tile() -> None:
    for blocking in (0.0, math.inf):
        grid = numpy.array([[1.0, blocking, 1.0]])
        assert semita.find_path(grid, (0, 0), (0, 2)) is None
        assert semita.flood_from(grid, (0, 0)).costs.tolist() == [[0.0, math.inf, math.inf]]
        # As for any blocked start: it is a path to itself, and nothing leaves it.
        assert semita.flood_from(grid, (0, 1)).costs.tolist() == [[math.inf, 0.0, math.inf]]
        path = semita.find_path(grid, (0, 1), (0, 1))
        assert path is not None
        assert (path.cost, path.distance) == (0.0, 0.0)


@pytest.mark.parametrize("cost", [-1.0, -math.inf, math.nan])
def test_a_negative_or_nan_cost_raises(cost: float) -> None:
    grid = numpy.ones((3, 4))
    grid[1, 2] = cost
    message = re.escape(f"the cost at (1, 2) is {cost!r}; a tile's cost is positive, or 0 or inf to block it")
    with pytest.raises(ValueError, match=message):
        semita.find_path(grid, (0, 0), (0, 1))
    with pytest.raises(ValueError, match=message):
        semita.flood_from(grid, (0, 0))


def test_a_grid_is_two_dimensional_and_of_bools_or_real_numbers() -> None:
    with pytest.raises(ValueError, match="two-dimensional, got 1"):
        semita.find_path(numpy.ones(5, dtype=bool), (0, 0), (1, 1))
    for grid in (
        numpy.array([["a", "b"]]),
        numpy.array([[object()]]),
        numpy.zeros((2, 2), dtype="datetime64[s]"),
        numpy.ones((2, 2), dtype=complex),
    ):
        with pytest.raises(TypeError, match=re.escape(f"bools or real numbers, got {grid.dtype}")):
            semita.flood_from(grid, (0, 0))


def test_the_c_module_checks_the_buffers_it_is_handed() -> None:
    passable = numpy.ones((3, 3), dtype=bool)
    with pytest.raises(TypeError, match="9 items of format 'd'"):
        _search.flood_from(passable, 0, 0, numpy.empty(8), numpy.empty(9))
    with pytest.raises(TypeError, match="items of format 'd'"):
        _search.flood_from(passable, 0, 0, numpy.empty(9), numpy.empty(9, dtype=numpy.float32))
    for grid in (numpy.ones((3, 3), dtype=numpy.float32), numpy.ones((3, 3), dtype=numpy.uint8)):
        with pytest.raises(TypeError, match="2-D C-contiguous buffer of bool or double"):
            _search.find_path(grid, 0, 0, 2, 2)
    with pytest.raises(IndexError, match="the start"):
        _search.flood_from(passable, 3, 0, numpy.empty(9), numpy.empty(9))


def test_searches_on_threads_at_once_find_what_they_find_one_at_a_time() -> None:
    grids = [_TEST_GRIDS["dense"], _TEST_GRIDS["rooms"], _TEST_GRIDS["mazelike"]]
    jobs = []
    for seed in range(600):
        passable = grids[seed % len(grids)]
        start, end = _tiles_of(passable, 2, seed=seed)
        jobs.append((passable, start, end))

    def search(job: tuple[NDArray[numpy.bool_], tuple[int, int], tuple[int, int]]) -> tuple[object, ...]:
        """All a search finds, on bools and on costs, whole: a flood's sum would read inf wherever a tile is blocked."""
        passable, start, end = job
        found: list[object] = []
        for grid in (passable, _random_costs(passable, seed=17)):
            path = semita.find_path(grid, start, end)
            found.append(None if path is None else (path.cost, path.distance, path.tiles.tolist()))
            flood = semita.flood_from(grid, start)
            found.append((flood.costs.tolist(), flood.distances.tolist()))
        return tuple(found)

    one_at_a_time = [search(job) for job in jobs]
    with ThreadPoolExecutor(8) as pool:
        at_once = list(pool.map(search, jobs))
    assert at_once == one_at_a_time


def _random_costs(passable: NDArray[numpy.bool_], seed: int, *, blocked: float = 0.05) -> NDArray[numpy.float64]:
    """A grid of costs from 0.2 to 5 where `passable` is true, `inf` on a share `blocked` of those tiles, and 0
    elsewhere."""
    rng = numpy.random.default_rng(seed)
    costs = rng.uniform(0.2, 5.0, size=passable.shape)
    costs[rng.random(passable.shape) < blocked] = math.inf
    return numpy.where(passable, costs, 0.0)


def _expected_costs(grid: NDArray[numpy.float64], start: tuple[int, int]) -> NDArray[numpy.float64]:
    """What the oracle finds on a grid of costs, whose 0 and inf block."""
    return _shortest_distances(grid > 0, start, grid)


@pytest.mark.parametrize("name", list(_TEST_GRIDS))
def test_every_cost_is_the_cheapest_paths(name: str) -> None:
    grid = _random_costs(_TEST_GRIDS[name], seed=18)
    for start in _tiles_of(grid > 0, 5, seed=19):
        found = semita.flood_from(grid, start).costs
        expected = _expected_costs(grid, start)
        assert (numpy.isinf(found) == numpy.isinf(expected)).all()
        reachable = numpy.isfinite(expected)
        assert numpy.abs(found[reachable] - expected[reachable]).max() < 1e-9


@pytest.mark.parametrize("name", list(_TEST_GRIDS))
def test_a_path_on_costs_is_a_cheapest_one_of_legal_steps(name: str) -> None:
    grid = _random_costs(_TEST_GRIDS[name], seed=20)
    open_ground = (grid > 0) & numpy.isfinite(grid)
    starts, ends = _tiles_of(grid > 0, 30, seed=21), _tiles_of(grid > 0, 30, seed=22)
    for start, end in zip(starts, ends, strict=True):
        expected = _expected_costs(grid, start)[end]
        path = semita.find_path(grid, start, end)
        if math.isinf(expected) and start != end:
            assert path is None
            continue
        assert path is not None
        assert abs(path.cost - expected) < 1e-9
        tiles = [(int(i), int(j)) for i, j in path.tiles]
        assert (tiles[0], tiles[-1]) == (start, end)
        walked = paid = 0.0
        for (i, j), (next_i, next_j) in zip(tiles, tiles[1:], strict=False):
            di, dj = next_i - i, next_j - j
            assert max(abs(di), abs(dj)) == 1
            assert open_ground[next_i, next_j]
            if di and dj:
                assert open_ground[i + di, j] and open_ground[i, j + dj], "the path cuts a corner"
            length = math.sqrt(2) if di and dj else 1.0
            walked += length
            paid += length * (grid[i, j] + grid[next_i, next_j]) / 2
        assert abs(walked - path.distance) < 1e-9
        assert abs(paid - path.cost) < 1e-9


def test_a_flood_on_bools_has_the_costs_for_distances() -> None:
    passable = _TEST_GRIDS["rooms"]
    for start in _tiles_of(passable, 5, seed=28):
        flood = semita.flood_from(passable, start)
        assert numpy.array_equal(flood.costs, flood.distances)


def test_a_floods_distances_are_its_cheapest_paths_lengths() -> None:
    passable = _TEST_GRIDS["sparse"]
    grid = _random_costs(passable, seed=29)
    for start in _tiles_of(grid > 0, 4, seed=31):
        flood = semita.flood_from(grid, start)
        reachable = numpy.isfinite(flood.costs)
        assert (numpy.isfinite(flood.distances) == reachable).all()
        # Costs from 0.2 to 5 make every cheapest path the only one, so A* finds the one the flood measured.
        for end in _tiles_of(grid > 0, 15, seed=32):
            path = semita.find_path(grid, start, end)
            if path is None:
                assert not reachable[end]
                continue
            assert flood.costs[end] == pytest.approx(path.cost, abs=1e-9)
            assert flood.distances[end] == pytest.approx(path.distance, abs=1e-9)
        # No path is shorter than a shortest one.
        plain = semita.flood_from(passable & numpy.isfinite(grid), start)
        assert (flood.distances[reachable] >= plain.distances[reachable] - 1e-9).all()


def test_a_floods_arrays_are_its_own() -> None:
    flood = semita.flood_from(numpy.ones((3, 3), dtype=bool), (0, 0))
    flood.costs[1, 1] = 99.0
    assert flood.distances[1, 1] == pytest.approx(math.sqrt(2))


def test_a_path_costs_the_same_both_ways() -> None:
    grid = _random_costs(_TEST_GRIDS["sparse"], seed=25)
    for one, other in zip(_tiles_of(grid > 0, 10, seed=26), _tiles_of(grid > 0, 10, seed=27), strict=True):
        there = semita.flood_from(grid, one).costs[other]
        back = semita.flood_from(grid, other).costs[one]
        assert there == pytest.approx(back, abs=1e-9) or math.isinf(there) and math.isinf(back)


def test_a_cheap_lane_draws_the_path_off_the_straight_line() -> None:
    grid = numpy.ones((5, 21))
    grid[4, :] = 0.25
    path = semita.find_path(grid, (0, 0), (0, 20))
    assert path is not None
    assert (path.tiles[:, 0] == 4).sum() >= 10
    assert path.cost < 20 < path.distance


def test_the_readme_example_runs_as_it_says() -> None:
    readme = (FilePath(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    (example,) = re.findall(r"```python\n(.*?)```", readme, re.DOTALL)
    exec(compile(example, "README.md", "exec"), {})
