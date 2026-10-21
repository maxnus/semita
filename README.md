# semita

Shortest and cheapest paths and distances on a grid, in C, for Python: jump point search for one path, A* for one
where tiles have costs, and Dijkstra for every tile's distance or cost from a point. *Semita* is Latin for a footpath.

Written for [NachOS](https://github.com/maxnus/NachOS), a StarCraft II bot library, though nothing in it is about
StarCraft.

```python
import numpy

import semita

passable = numpy.ones((5, 5), dtype=bool)
passable[2, :4] = False  # a wall, open at its far end

path = semita.find_path(passable, (0, 0), (4, 0))
assert round(path.distance, 2) == 10.83  # round the wall's end
assert path.tiles[:3].tolist() == [[0, 0], [1, 1], [1, 2]]

assert semita.find_path(passable, (0, 0), (2, 1)) is None  # the end is blocked

distances = semita.distances_from(passable, (0, 0))
assert distances[4, 0] == path.distance
assert distances[2, 1] == float("inf")

# Costs: moving a unit of distance across a tile costs 1 on normal ground, more where it is best avoided.
open_ground = numpy.ones((7, 7), dtype=bool)
costs = numpy.ones((7, 7))
costs[2:5, 2:5] = 10.0  # a patch to keep out of
around = semita.find_path(open_ground, (3, 0), (3, 6), costs=costs)
assert round(around.cost, 2) == round(around.distance, 2) == 7.66  # round the patch, on ground costing 1
assert semita.find_path(open_ground, (3, 0), (3, 6)).distance == 6.0  # straight through it, without costs
```

## The rules

- A grid is a 2-D numpy array of bools or numbers, or anything numpy reads as one, passable where nonzero, NaN
  included, and indexed `[i, j]`. Bools and uint8 are read as they are; a grid of any other number is converted on
  each call, so one searched often is best kept as bools.
- A step goes to one of the eight neighboring tiles, 1 long straight and √2 diagonally, and only between passable
  tiles. A diagonal step needs both tiles beside it passable too, so no path cuts a corner.
- The start is a path to itself, 0 long, even on a blocked tile, but no step leaves or enters a blocked one.
- A start or an end outside the grid raises `IndexError`; a negative index counts as outside.
- `find_path` returns a `Path`, its `distance`, its `cost` and its `tiles` from the start to the end, or `None` if
  there is no path. Of several equally cheap paths it returns one, but which is not part of the contract.
- `distances_from` returns an array of the grid's shape, `inf` where no path reaches.
- `costs=`, a grid of real numbers of the grid's shape, gives each tile the cost of moving one unit of distance across
  it: 1 for normal ground, less where movement is faster, more where it is slower or best avoided. A step then costs
  its length times the mean of the costs of the two tiles it joins, so a path costs the same both ways.
  `find_path` returns the cheapest path, its `cost` beside its `distance`, and `distances_from` each tile's cost.
- A passable tile's cost is positive, and `inf` blocks the tile; zero, negative or NaN raises `ValueError`. The
  costs of tiles `passable` blocks are never read.
- Without costs a path is found by jump point search; with them, by A*, which visits more tiles, the more so the
  cheaper the cheapest tile is, since its estimate of what is left is the straight distance times that cost.

Each search allocates its own memory and releases the GIL, so searches run in parallel on threads. A grid must not
be written to while a search reads it.

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
