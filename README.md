# semita

Shortest paths and distances on a grid, in C, for Python: jump point search for one path, Dijkstra for every
tile's distance from a point. *Semita* is Latin for a footpath.

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
```

## The rules

- A grid is a 2-D numpy array of bools, or of bytes passable where nonzero, indexed `[i, j]`.
- A step goes to one of the eight neighboring tiles, 1 long straight and √2 diagonally, and only between passable
  tiles. A diagonal step needs both tiles beside it passable too, so no path cuts a corner.
- The start is a path to itself, 0 long, even on a blocked tile, but no step leaves or enters a blocked one.
- A start or an end outside the grid raises `IndexError`; a negative index counts as outside.
- `find_path` returns a `Path`, its `distance` and its `tiles` from the start to the end, or `None` if there is no
  path. Of several equally short paths it returns one, but which is not part of the contract.
- `distances_from` returns an array of the grid's shape, `inf` where no path reaches.

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
