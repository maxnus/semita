"""Shortest and cheapest paths and distances on a grid.

A grid is a 2-D array, indexed `[i, j]`, whose values are the tiles' costs: the cost of moving one unit of distance
across each, 1 for normal ground, less where movement is faster, more where it is slower or best avoided. 0 and `inf`
block a tile. A grid of bools is the simplest: `True` costs 1 and `False` blocks. A cost is otherwise positive; NaN
and negatives raise `ValueError`.

A step goes to one of the eight neighboring tiles, 1 long straight and sqrt(2) diagonally, and only between passable
tiles; a diagonal step needs both tiles beside it passable too, so no path cuts a corner. A step costs its length
times the mean of the costs of the two tiles it joins, so a path costs the same both ways. The start is a path to
itself, costing nothing, even on a blocked tile, but no step leaves or enters a blocked one.

A grid whose passable tiles all cost the same, a grid of bools included, is searched by jump point search; any other
by A*, which visits more tiles, the more so the cheaper the cheapest tile is, since its estimate of what is left is
the straight distance times that cost. A grid of bools, or a C-contiguous one of float64, is read as it is; any
other is converted on each call.

Each search allocates its own memory and releases the GIL, so searches run in parallel on threads. A grid must not be
written to while a search reads it.
"""

from dataclasses import dataclass
from operator import index

import numpy
from numpy.typing import ArrayLike, NDArray

from semita import _search
from semita._version import __version__

__all__ = ["Flood", "Path", "__version__", "find_path", "flood_from"]


@dataclass(frozen=True, slots=True, eq=False)
class Path:
    """A cheapest path between two tiles, which on a grid of bools is a shortest one."""

    distance: float
    """Its length, the sum of its steps."""
    cost: float
    """What it costs, the sum of its steps' costs: its `distance` on a grid of bools."""
    tiles: NDArray[numpy.intc]
    """Its tiles in order, the start and the end included, as rows `(i, j)`; read-only."""


@dataclass(frozen=True, slots=True, eq=False)
class Flood:
    """The cheapest path from one tile to every tile of a grid, which on a grid of bools is a shortest one."""

    costs: NDArray[numpy.float64]
    """Each tile's cheapest path's cost, in an array of the grid's shape: `inf` where no path reaches."""
    distances: NDArray[numpy.float64]
    """The length of each tile's cheapest path, of one of them if several cost the same: `inf` where no path
    reaches, and `costs` on a grid of bools."""


def find_path(grid: ArrayLike, start: tuple[int, int], end: tuple[int, int]) -> Path | None:
    """A cheapest path from `start` to `end` over the `grid`, or `None` if there is none. Of several equally cheap paths
    it is one, but which is not part of the contract.

    Raises `IndexError` for a start or an end outside the grid; `ValueError` for an array that is not 2-D, or a tile
    whose cost is negative or NaN; and `TypeError` for one that holds neither bools nor real numbers.
    """
    tiles_grid = _as_grid(grid)
    start_i, start_j = _checked_tile(tiles_grid, start, "start")
    end_i, end_j = _checked_tile(tiles_grid, end, "end")
    cost, distance, tiles = _search.find_path(tiles_grid, start_i, start_j, end_i, end_j)
    if tiles is None:
        return None
    # An array over bytes is read-only.
    return Path(distance, cost, numpy.frombuffer(tiles, dtype=numpy.intc).reshape(-1, 2))


def flood_from(grid: ArrayLike, start: tuple[int, int]) -> Flood:
    """The cheapest path from `start` to each tile over the `grid`, by Dijkstra's algorithm: its cost and its length,
    found in the one pass.

    Raises as `find_path` does.
    """
    tiles_grid = _as_grid(grid)
    start_i, start_j = _checked_tile(tiles_grid, start, "start")
    costs = numpy.empty(tiles_grid.shape, dtype=numpy.float64)
    distances = numpy.empty(tiles_grid.shape, dtype=numpy.float64)
    _search.flood_from(tiles_grid, start_i, start_j, costs, distances)
    return Flood(costs, distances)


def _as_grid(grid: ArrayLike) -> NDArray[numpy.bool_] | NDArray[numpy.float64]:
    """`grid` as a C-contiguous array of bools, or of doubles, its costs: one of bools or of doubles as it is, copied
    only if it is not C-contiguous, and one of any other real number converted to doubles."""
    array = numpy.asarray(grid)
    if array.ndim != 2:
        raise ValueError(f"a grid is two-dimensional, got {array.ndim} dimensions")
    if array.dtype == numpy.bool_:
        return numpy.ascontiguousarray(array)
    if not (numpy.issubdtype(array.dtype, numpy.integer) or numpy.issubdtype(array.dtype, numpy.floating)):
        raise TypeError(f"a grid holds bools or real numbers, got {array.dtype}")
    return numpy.ascontiguousarray(array, dtype=numpy.float64)


def _checked_tile(
    grid: NDArray[numpy.bool_] | NDArray[numpy.float64], tile: tuple[int, int], which: str
) -> tuple[int, int]:
    """`tile` as two ints, raising `IndexError` if the grid has no such tile; a negative index counts as outside."""
    i, j = (index(coordinate) for coordinate in tile)
    width, height = grid.shape
    if not (0 <= i < width and 0 <= j < height):
        raise IndexError(f"the {which} ({i}, {j}) is outside the grid's {width} by {height} tiles")
    return i, j
