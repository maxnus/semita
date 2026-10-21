/*
 * The searches semita runs, on a grid of tiles.
 *
 * Tile (i, j) of a grid `width` by `height` is at index i * height + j, passable where nonzero. A step goes to one of
 * the eight neighboring tiles, 1 long straight and sqrt(2) diagonally, and only between passable tiles. A diagonal
 * step needs both tiles beside it passable too, so no path cuts a corner.
 *
 * Every search allocates its own memory and touches no global state, so searches can run at once on any threads.
 */

#ifndef SEMITA_SEARCH_H
#define SEMITA_SEARCH_H

#include <stdint.h>

typedef struct {
    const uint8_t *passable;
    int width;
    int height;
} Grid;

typedef enum {
    SEARCH_DONE = 0,
    SEARCH_OUT_OF_MEMORY = 1,
    /* The search contradicted itself, which is a bug in semita. */
    SEARCH_BROKEN = 2,
} SearchStatus;

/*
 * A shortest path from tile `start` to tile `end`, by jump point search: its tiles as (i, j) pairs, allocated into
 * `tiles` for the caller to free, their number into `tile_count` and its length into `distance`. The start alone is a
 * path to itself, blocked or not. Where no path exists, `tiles` is NULL, `tile_count` 0 and `distance` infinite.
 */
SearchStatus find_path_by_jump_points(
    const Grid *grid, int start, int end, int **tiles, int *tile_count, double *distance
);

/*
 * The length of a shortest path from tile `start` to every tile, by Dijkstra's algorithm, into `distances`, which
 * holds one for every tile of the grid: infinite where no path reaches, and 0 at the start, blocked or not.
 */
SearchStatus fill_distances(const Grid *grid, int start, double *distances);

/* ---- Shared by the searches ---- */

/* The eight steps, the four straight ones first. */
static const int STEP_I[8] = {1, -1, 0, 0, 1, 1, -1, -1};
static const int STEP_J[8] = {0, 0, 1, -1, 1, -1, 1, -1};

#define SQRT_2 1.4142135623730951

static inline int is_passable(const Grid *grid, int i, int j) {
    return i >= 0 && i < grid->width && j >= 0 && j < grid->height && grid->passable[i * grid->height + j];
}

/* Whether a step of (di, dj) from the passable tile (i, j) lands on a passable tile without cutting a corner. */
static inline int can_step(const Grid *grid, int i, int j, int di, int dj) {
    if (!is_passable(grid, i + di, j + dj)) return 0;
    return !(di && dj) || (is_passable(grid, i + di, j) && is_passable(grid, i, j + dj));
}

#endif /* SEMITA_SEARCH_H */
