"""Shortest paths and distances on a grid.

A grid is a 2-D array of bools or numbers, passable where nonzero, NaN included, and indexed `[i, j]`. Bools and
uint8 are read as they are; a grid of any other number is converted on each call. A step goes to one of the eight
neighboring tiles, 1 long straight and sqrt(2) diagonally, and only between passable tiles; a diagonal step needs
both tiles beside it passable too, so no path cuts a corner. The start is a path to itself, 0 long, even on a blocked
tile, but no step leaves or enters a blocked one.

`costs`, a grid of numbers of the same shape, gives each tile the cost of moving one unit of distance across it: 1 for
normal ground, less where movement is faster, more where it is slower or best avoided. A step then costs its length
times the mean of the costs of the two tiles it joins, so a path costs the same both ways. A passable tile's cost is
positive, and `inf` blocks the tile as `passable` does; the costs of tiles `passable` blocks are never read. Without
costs, a step costs its length. Costs that are C-contiguous float64 are read as they are; any others are converted on
each call.

Each search allocates its own memory and releases the GIL, so searches run in parallel on threads. A grid and its
costs must not be written to while a search reads them.
"""

from dataclasses import dataclass
from operator import index

import numpy
from numpy.typing import ArrayLike, NDArray

from semita import _search
from semita._version import __version__

__all__ = ["Path", "__version__", "distances_from", "find_path"]


@dataclass(frozen=True, slots=True, eq=False)
class Path:
    """A cheapest path between two tiles, which without costs is a shortest one."""

    distance: float
    """Its length, the sum of its steps."""
    cost: float
    """What it costs, the sum of its steps' costs: its `distance` without costs."""
    tiles: NDArray[numpy.intc]
    """Its tiles in order, the start and the end included, as rows `(i, j)`; read-only."""


def find_path(
    passable: ArrayLike, start: tuple[int, int], end: tuple[int, int], *, costs: ArrayLike | None = None
) -> Path | None:
    """A cheapest path from `start` to `end` over the `passable` tiles, or `None` if there is none: a shortest one by
    jump point search without `costs`, and by A* with them. Of several equally cheap paths it is one, but which is not
    part of the contract.

    Raises `IndexError` for a start or an end outside the grid; `ValueError` for an array that is not 2-D, costs of
    another shape than the grid's, or a passable tile whose cost is zero, negative or NaN; and `TypeError` for a grid
    that holds neither bools nor numbers, or costs that are not real numbers.
    """
    grid = _as_grid(passable)
    start_i, start_j = _checked_tile(grid, start, "start")
    end_i, end_j = _checked_tile(grid, end, "end")
    cost, distance, tiles = _search.find_path(grid, start_i, start_j, end_i, end_j, _as_costs(costs, grid))
    if tiles is None:
        return None
    # An array over bytes is read-only.
    return Path(distance, cost, numpy.frombuffer(tiles, dtype=numpy.intc).reshape(-1, 2))


def distances_from(
    passable: ArrayLike, start: tuple[int, int], *, costs: ArrayLike | None = None
) -> NDArray[numpy.float64]:
    """The cost of a cheapest path from `start` to each tile over the `passable` tiles, by Dijkstra's algorithm, in an
    array of the grid's shape: without `costs`, the length of a shortest one; `inf` where no path reaches.

    Raises as `find_path` does.
    """
    grid = _as_grid(passable)
    start_i, start_j = _checked_tile(grid, start, "start")
    distances = numpy.empty(grid.shape, dtype=numpy.float64)
    _search.distances_from(grid, start_i, start_j, distances, _as_costs(costs, grid))
    return distances


def _as_grid(passable: ArrayLike) -> NDArray[numpy.bool_] | NDArray[numpy.uint8]:
    """`passable` as a C-contiguous array of bools or bytes: one of bools or bytes as it is, copied only if it is not
    C-contiguous, and one of any other number as bools, true where nonzero."""
    grid = numpy.asarray(passable)
    if grid.ndim != 2:
        raise ValueError(f"a grid is two-dimensional, got {grid.ndim} dimensions")
    if grid.dtype == numpy.bool_ or grid.dtype == numpy.uint8:
        return numpy.ascontiguousarray(grid)
    if not numpy.issubdtype(grid.dtype, numpy.number):
        raise TypeError(f"a grid holds bools or numbers, got {grid.dtype}")
    # Compared straight into a C-ordered array: one copy, whatever the grid's own order.
    return numpy.not_equal(grid, 0, order="C")


def _as_costs(
    costs: ArrayLike | None, grid: NDArray[numpy.bool_] | NDArray[numpy.uint8]
) -> NDArray[numpy.float64] | None:
    """`costs` as a C-contiguous array of doubles of the grid's shape, copied only if it is not one already."""
    if costs is None:
        return None
    array = numpy.asarray(costs)
    if array.shape != grid.shape:
        raise ValueError(f"the costs have the shape {array.shape}, the grid {grid.shape}")
    if not (numpy.issubdtype(array.dtype, numpy.integer) or numpy.issubdtype(array.dtype, numpy.floating)):
        raise TypeError(f"costs are real numbers, got {array.dtype}")
    return numpy.ascontiguousarray(array, dtype=numpy.float64)


def _checked_tile(
    grid: NDArray[numpy.bool_] | NDArray[numpy.uint8], tile: tuple[int, int], which: str
) -> tuple[int, int]:
    """`tile` as two ints, raising `IndexError` if the grid has no such tile; a negative index counts as outside."""
    i, j = (index(coordinate) for coordinate in tile)
    width, height = grid.shape
    if not (0 <= i < width and 0 <= j < height):
        raise IndexError(f"the {which} ({i}, {j}) is outside the grid's {width} by {height} tiles")
    return i, j
