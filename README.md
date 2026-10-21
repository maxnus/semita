# semita

Shortest and cheapest paths and distances on a grid, in C, for Python: jump point search for one path, A* for one
where tiles cost differently, and Dijkstra for the cheapest paths from a point to every tile. *Semita* is Latin for a
footpath.

Written for [NachOS](https://github.com/maxnus/NachOS), a StarCraft II bot library, though nothing in it is about
StarCraft.

```python
import numpy

import semita

passable = numpy.ones((5, 5), dtype=bool)
passable[2, :4] = False  # a wall, open at its far end

path = semita.cheapest_path(passable, (0, 0), (4, 0))
assert round(path.distance, 2) == 10.83  # round the wall's end
assert path.tiles[:3].tolist() == [[0, 0], [1, 1], [1, 2]]

assert semita.cheapest_path(passable, (0, 0), (2, 1)) is None  # the end is blocked

paths = semita.cheapest_paths_from(passable, (0, 0))  # to every tile, in one pass
assert paths.distances[4, 0] == path.distance
assert paths.distances[2, 1] == float("inf")
assert paths.path_to((4, 0)).cost == path.cost

# Costs: each value is the cost of moving a unit of distance across its tile, 1 on normal ground and more where it
# is best avoided; 0 blocks, as False does.
costs = numpy.ones((7, 7))
costs[2:5, 2:5] = 10.0  # a patch to keep out of
around = semita.cheapest_path(costs, (3, 0), (3, 6))
assert round(around.cost, 2) == round(around.distance, 2) == 7.66  # round the patch, on ground costing 1
assert semita.cheapest_path(numpy.ones((7, 7), dtype=bool), (3, 0), (3, 6)).distance == 6.0  # straight, on bools
```

## The rules

- A grid is a 2-D numpy array of bools or real numbers, or anything numpy reads as one, indexed `[i, j]`. Its values
  are the tiles' costs: the cost of moving one unit of distance across each, 1 for normal ground, less where movement
  is faster, more where it is slower or best avoided.
- 0 and `inf` block a tile. A grid of bools is the simplest: `True` costs 1 and `False` blocks. Otherwise a cost is
  positive; NaN and negatives raise `ValueError`.
- A step goes to one of the eight neighboring tiles, 1 long straight and √2 diagonally, and only between passable
  tiles. A diagonal step needs both tiles beside it passable too, so no path cuts a corner. A step costs its length
  times the mean of the costs of the two tiles it joins, so a path costs the same both ways.
- The start is a path to itself, costing nothing, even on a blocked tile, but no step leaves or enters a blocked one.
- A start or an end outside the grid raises `IndexError`; a negative index counts as outside.
- `cheapest_path` returns a `Path`, its `distance`, its `cost` and its `tiles` from the start to the end, or `None` if
  there is no path. Of several equally cheap paths it returns one, but which is not part of the contract.
- `cheapest_paths_from` returns the cheapest path from the start to every tile, found in one pass, as
  `CheapestPaths`: their `costs` and their `distances`, two read-only arrays of the grid's shape, `inf` where no path
  reaches, and `path_to(tile)`, a tile's path as a `Path`, or `None` where none reaches. Where several paths to a tile
  cost the same, it holds one of them. On a grid of bools the distances are the costs.
- A grid whose passable tiles all cost the same, a grid of bools included, is searched by jump point search; any other
  by A*, which visits more tiles, the more so the cheaper the cheapest tile is, since its estimate of what is left is
  the straight distance times that cost.
- A grid of bools, or a C-contiguous one of float64, is read as it is; any other is converted on each call, so one
  searched often is best kept as one of those.

Each search allocates its own memory and releases the GIL, so searches run in parallel on threads. A grid must not be
written to while a search reads it.

## Building

semita is a C extension on Python's stable ABI, so one wheel serves every Python from 3.12 on a platform. Installing
from source compiles it with the platform's own compiler: MSVC on Windows, gcc or clang elsewhere.

```bash
uv sync --extra dev
uv run pytest
```

The tests check every search against scipy's Dijkstra over the same steps.

## Releasing

Publish a GitHub release tagged `v` and the new version, such as `v0.2.0`. The version is the tag, which
setuptools-scm reads from git; no file holds it.

The `release` workflow then builds the sdist, builds from it a wheel for Linux x86_64 and aarch64, Windows AMD64 and
macOS arm64 and x86_64, tests and audits each for the stable ABI alone, checks that the tag is a version,
and uploads them to PyPI. Every pull request builds and checks the same files without uploading them.

PyPI trusts the workflow through trusted publishing, from the repository's `pypi` environment, so no token is stored
anywhere.
