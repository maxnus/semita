/*
 * A shortest path between two tiles, by jump point search: A* that skips along straight and diagonal runs and puts
 * on its heap only the tiles where a path may turn. Without corner cutting, a diagonal run has no forced neighbors,
 * and a straight run has one wherever a tile beside it opens while the tile beside the last one was blocked.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "heap.h"
#include "search.h"

#define NO_TILE (-1)
#define FROM_THE_START (-1)

/* What a search knows of a tile: nothing yet, a cost to it that may still fall, or its final cost. A tile's cost,
   parent and arrival are read only once it is reached, so the search never fills them for the whole grid. */
enum { UNREACHED = 0, REACHED = 1, SETTLED = 2 };

/* The step with these components, or -1. */
static int step_of(int di, int dj) {
    for (int step = 0; step < 8; step++) {
        if (STEP_I[step] == di && STEP_J[step] == dj) return step;
    }
    return -1;
}

/* Whether a straight run along (di, dj) has a forced neighbor at (i, j), on the side (side_i, side_j). */
static inline int forced_beside(const Grid *grid, int i, int j, int di, int dj, int side_i, int side_j) {
    return is_passable(grid, i + side_i, j + side_j) && !is_passable(grid, i - di + side_i, j - dj + side_j);
}

/* The tile a straight run from (i, j) along (di, dj) stops at: the end, or a tile with a forced neighbor. NO_TILE if
   it reaches a blocked tile or the grid's edge first. */
static int jump_straight(const Grid *grid, int i, int j, int di, int dj, int end_i, int end_j) {
    /* The two sides of the run. */
    int side_i = dj ? 1 : 0, side_j = di ? 1 : 0;
    for (;;) {
        i += di;
        j += dj;
        if (!is_passable(grid, i, j)) return NO_TILE;
        if (i == end_i && j == end_j) return i * grid->height + j;
        if (forced_beside(grid, i, j, di, dj, side_i, side_j) || forced_beside(grid, i, j, di, dj, -side_i, -side_j)) {
            return i * grid->height + j;
        }
    }
}

/* The tile a diagonal run from (i, j) along (di, dj) stops at: the end, or a tile from which a straight run along
   either of its components stops somewhere. NO_TILE if it reaches a blocked tile, a corner or the grid's edge first. */
static int jump_diagonal(const Grid *grid, int i, int j, int di, int dj, int end_i, int end_j) {
    for (;;) {
        if (!can_step(grid, i, j, di, dj)) return NO_TILE;
        i += di;
        j += dj;
        if (i == end_i && j == end_j) return i * grid->height + j;
        if (jump_straight(grid, i, j, di, 0, end_i, end_j) != NO_TILE ||
            jump_straight(grid, i, j, 0, dj, end_i, end_j) != NO_TILE) {
            return i * grid->height + j;
        }
    }
}

/* The directions a search goes on in from (i, j), reached by `arrival`, into `steps`; returns their number. */
static int directions_from(const Grid *grid, int i, int j, int arrival, int steps[8]) {
    int count = 0;
    if (arrival == FROM_THE_START) {
        for (int step = 0; step < 8; step++) steps[count++] = step;
        return count;
    }
    int di = STEP_I[arrival], dj = STEP_J[arrival];
    if (di && dj) {
        steps[count++] = step_of(di, 0);
        steps[count++] = step_of(0, dj);
        steps[count++] = arrival;
        return count;
    }
    steps[count++] = arrival;
    int side_i = dj ? 1 : 0, side_j = di ? 1 : 0;
    for (int sign = 1; sign >= -1; sign -= 2) {
        if (forced_beside(grid, i, j, di, dj, sign * side_i, sign * side_j)) {
            steps[count++] = step_of(sign * side_i, sign * side_j);
            steps[count++] = step_of(di + sign * side_i, dj + sign * side_j);
        }
    }
    return count;
}

SearchStatus find_path_by_jump_points(
    const Grid *grid, int start, int end, int **tiles, int *tile_count, double *distance
) {
    int height = grid->height;
    int end_i = end / height, end_j = end % height;
    *tiles = NULL;
    *tile_count = 0;
    *distance = INFINITY;
    if (start == end) {
        int *path = malloc(2 * sizeof *path);
        if (!path) return SEARCH_OUT_OF_MEMORY;
        path[0] = end_i;
        path[1] = end_j;
        *tiles = path;
        *tile_count = 1;
        *distance = 0.0;
        return SEARCH_DONE;
    }
    if (grid->costs) return SEARCH_BROKEN;
    if (!grid->passable[start] || !grid->passable[end]) return SEARCH_DONE;

    size_t count = (size_t)grid->width * (size_t)height;
    double *cost = malloc(count * sizeof *cost);
    int *parent = malloc(count * sizeof *parent);
    int8_t *arrival = malloc(count * sizeof *arrival);
    uint8_t *state = calloc(count, 1);
    Heap heap = {0};
    SearchStatus status = SEARCH_DONE;
    if (!cost || !parent || !arrival || !state) {
        status = SEARCH_OUT_OF_MEMORY;
        goto done;
    }
    state[start] = REACHED;
    cost[start] = 0.0;
    parent[start] = NO_TILE;
    arrival[start] = FROM_THE_START;
    if (heap_push(&heap, octile_distance(start / height, start % height, end_i, end_j), start)) {
        status = SEARCH_OUT_OF_MEMORY;
        goto done;
    }

    while (heap.size > 0) {
        int tile = heap_pop(&heap).tile;
        if (state[tile] == SETTLED) continue;
        state[tile] = SETTLED;
        if (tile == end) {
            /* The cost to the end is the path's length, the same number its cost is reported as. */
            status = write_path(grid, end, parent, tiles, tile_count, NULL);
            if (status == SEARCH_DONE) *distance = cost[end];
            break;
        }
        int i = tile / height, j = tile % height;
        int steps[8];
        int step_count = directions_from(grid, i, j, arrival[tile], steps);
        for (int k = 0; k < step_count; k++) {
            int step = steps[k], di = STEP_I[step], dj = STEP_J[step];
            int landing = di && dj ? jump_diagonal(grid, i, j, di, dj, end_i, end_j)
                                   : jump_straight(grid, i, j, di, dj, end_i, end_j);
            if (landing == NO_TILE || state[landing] == SETTLED) continue;
            int landing_i = landing / height, landing_j = landing % height;
            double landing_cost = cost[tile] + octile_distance(i, j, landing_i, landing_j);
            if (state[landing] == UNREACHED || landing_cost < cost[landing]) {
                state[landing] = REACHED;
                cost[landing] = landing_cost;
                parent[landing] = tile;
                arrival[landing] = (int8_t)step;
                double estimate = landing_cost + octile_distance(landing_i, landing_j, end_i, end_j);
                if (heap_push(&heap, estimate, landing)) {
                    status = SEARCH_OUT_OF_MEMORY;
                    goto done;
                }
            }
        }
    }
done:
    heap_free(&heap);
    free(cost);
    free(parent);
    free(arrival);
    free(state);
    return status;
}
