"""Shortest paths and distances on a grid.

A grid is a 2-D array of bools, or of bytes passable where nonzero, indexed `[i, j]`. A step goes to one of the eight
neighboring tiles, 1 long straight and sqrt(2) diagonally, and only between passable tiles; a diagonal step needs
both tiles beside it passable too, so no path cuts a corner. The start is a path to itself, 0 long, even on a blocked
tile, but no step leaves or enters a blocked one.

Each search allocates its own memory and releases the GIL, so searches run in parallel on threads. A grid must not be
written to while a search reads it.
"""

from dataclasses import dataclass
from operator import index

import numpy
from numpy.typing import ArrayLike, NDArray

from semita import _search

__all__ = ["Path", "distances_from", "find_path"]
# pyproject.toml reads the version from here.
__version__ = "0.1.0"


@dataclass(frozen=True, slots=True, eq=False)
class Path:
    """A shortest path between two tiles."""

    distance: float
    """Its length, the sum of its steps."""
    tiles: NDArray[numpy.intc]
    """Its tiles in order, the start and the end included, as rows `(i, j)`; read-only."""


def find_path(passable: ArrayLike, start: tuple[int, int], end: tuple[int, int]) -> Path | None:
    """A shortest path from `start` to `end` over the `passable` tiles, by jump point search, or `None` if there is
    none. Of several equally short paths it is one, but which is not part of the contract.

    Raises `IndexError` for a start or an end outside the grid, `ValueError` for an array that is not 2-D, and
    `TypeError` for one that is neither bool nor uint8.
    """
    grid = _as_grid(passable)
    start_i, start_j = _checked_tile(grid, start, "start")
    end_i, end_j = _checked_tile(grid, end, "end")
    distance, tiles = _search.find_path(grid, start_i, start_j, end_i, end_j)
    if tiles is None:
        return None
    # An array over bytes is read-only.
    return Path(distance, numpy.frombuffer(tiles, dtype=numpy.intc).reshape(-1, 2))


def distances_from(passable: ArrayLike, start: tuple[int, int]) -> NDArray[numpy.float64]:
    """The length of a shortest path from `start` to each tile over the `passable` tiles, by Dijkstra's algorithm,
    in an array of the grid's shape: `inf` where no path reaches.

    Raises as `find_path` does.
    """
    grid = _as_grid(passable)
    start_i, start_j = _checked_tile(grid, start, "start")
    distances = numpy.empty(grid.shape, dtype=numpy.float64)
    _search.distances_from(grid, start_i, start_j, distances)
    return distances


def _as_grid(passable: ArrayLike) -> NDArray[numpy.bool_] | NDArray[numpy.uint8]:
    """`passable` as a C-contiguous array of bools or bytes, copied only if it is not one already."""
    grid = numpy.asarray(passable)
    if grid.ndim != 2:
        raise ValueError(f"a grid is two-dimensional, got {grid.ndim} dimensions")
    if grid.dtype != numpy.bool_ and grid.dtype != numpy.uint8:
        raise TypeError(f"a grid holds bool or uint8, got {grid.dtype}")
    return numpy.ascontiguousarray(grid)


def _checked_tile(
    grid: NDArray[numpy.bool_] | NDArray[numpy.uint8], tile: tuple[int, int], which: str
) -> tuple[int, int]:
    """`tile` as two ints, raising `IndexError` if the grid has no such tile; a negative index counts as outside."""
    i, j = (index(coordinate) for coordinate in tile)
    width, height = grid.shape
    if not (0 <= i < width and 0 <= j < height):
        raise IndexError(f"the {which} ({i}, {j}) is outside the grid's {width} by {height} tiles")
    return i, j
