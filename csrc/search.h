/*
 * The searches semita runs, on a grid of tiles.
 *
 * Tile (i, j) of a grid `width` by `height` is at index i * height + j. A grid is either of bools, a tile passable
 * where true, or of costs, the cost of moving one unit of distance across each tile: positive, and 0 or infinite
 * where the tile is blocked. A step goes to one of the eight neighboring tiles, 1 long straight and sqrt(2)
 * diagonally, and only between passable tiles. A diagonal step needs both tiles beside it passable too, so no path
 * cuts a corner. A step costs its length times the mean of the costs of the two tiles it joins, and on a grid of bools
 * its length.
 *
 * Every search allocates its own memory and touches no global state, so searches can run at once on any threads.
 */

#ifndef SEMITA_SEARCH_H
#define SEMITA_SEARCH_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/* One of `passable` and `costs` is NULL. */
typedef struct {
    const uint8_t *passable;
    const double *costs;
    int width;
    int height;
} Grid;

typedef enum {
    SEARCH_DONE = 0,
    SEARCH_OUT_OF_MEMORY = 1,
    /* The search contradicted itself, which is a bug in semita. */
    SEARCH_BROKEN = 2,
    /* A tile's cost is negative or NaN. */
    SEARCH_BAD_COST = 3,
} SearchStatus;

/*
 * Check every tile's cost, and find the cheapest and the dearest of the passable tiles into `cheapest` and `dearest`,
 * infinite and 0 if there is none. A cost that is negative or NaN gives SEARCH_BAD_COST, with its tile in `bad_tile`.
 */
SearchStatus check_costs(const Grid *grid, double *cheapest, double *dearest, int *bad_tile);

/*
 * A shortest path from tile `start` to tile `end` on a grid of bools, or of costs all the same, by jump point search,
 * which reads only which tiles are passable: its tiles as (i, j)
 * pairs, allocated into `tiles` for the caller to free, their number into `tile_count` and its length into
 * `distance`. The start alone is a path to itself, blocked or not. Where no path exists, `tiles` is NULL, `tile_count`
 * 0 and `distance` infinite.
 */
SearchStatus find_path_by_jump_points(
    const Grid *grid, int start, int end, int **tiles, int *tile_count, double *distance
);

/*
 * The cheapest path from tile `start` to tile `end` on a grid of costs, by A*, whose estimate of what is left is
 * the octile distance times `cheapest`, the cheapest cost on the grid: its tiles as for find_path_by_jump_points, its
 * cost into `cost` and its length into `distance`.
 */
SearchStatus find_cheapest_path(
    const Grid *grid, int start, int end, double cheapest, int **tiles, int *tile_count, double *cost,
    double *distance
);

/*
 * The cheapest path from tile `start` to every tile, by Dijkstra's algorithm: its cost into `costs`, its length into
 * `distances` and the tile it arrives from into `parents`, each holding one for every tile of the grid. A cost and a
 * length are infinite where no path reaches, and 0 at the start, blocked or not; on a grid of bools the two are the
 * same. The start's parent is -1, and the parent of a tile no path reaches is never written. Of several equally cheap
 * paths to a tile, the one recorded is one of them.
 */
SearchStatus fill_flood(const Grid *grid, int start, double *costs, double *distances, int *parents);

/*
 * The path that ends at `end`, allocated into `tiles` as (i, j) pairs, walking `parent` back from it to a tile whose
 * parent is -1, each tile and its parent on one straight or diagonal run, and its length into `distance` unless it is
 * NULL. A walk longer than the grid has tiles, which only a broken chain of parents makes, gives SEARCH_BROKEN.
 */
SearchStatus write_path(const Grid *grid, int end, const int *parent, int **tiles, int *tile_count, double *distance);

/* ---- Shared by the searches ---- */

/* The eight steps, the four straight ones first. */
static const int STEP_I[8] = {1, -1, 0, 0, 1, 1, -1, -1};
static const int STEP_J[8] = {0, 0, 1, -1, 1, -1, 1, -1};

#define SQRT_2 1.4142135623730951

static inline int is_passable(const Grid *grid, int i, int j) {
    if (i < 0 || i >= grid->width || j < 0 || j >= grid->height) return 0;
    int tile = i * grid->height + j;
    if (!grid->costs) return grid->passable[tile];
    double cost = grid->costs[tile];
    return cost > 0.0 && cost < INFINITY;
}

/* Whether a step of (di, dj) from the passable tile (i, j) lands on a passable tile without cutting a corner. */
static inline int can_step(const Grid *grid, int i, int j, int di, int dj) {
    if (!is_passable(grid, i + di, j + dj)) return 0;
    return !(di && dj) || (is_passable(grid, i + di, j) && is_passable(grid, i, j + dj));
}

/* The length of a shortest path between two tiles with nothing in the way: the exact length of a straight or diagonal
   run, and what the A* searches estimate is left. */
static inline double octile_distance(int i, int j, int other_i, int other_j) {
    int di = abs(i - other_i), dj = abs(j - other_j);
    return di > dj ? di + (SQRT_2 - 1.0) * dj : dj + (SQRT_2 - 1.0) * di;
}

/* What a step of (di, dj) from tile `from` costs. */
static inline double step_cost(const Grid *grid, int from, int di, int dj) {
    double length = di && dj ? SQRT_2 : 1.0;
    if (!grid->costs) return length;
    return length * (grid->costs[from] + grid->costs[from + di * grid->height + dj]) / 2.0;
}

#endif /* SEMITA_SEARCH_H */
