/*
 * The cheapest path between two tiles on a grid with costs, by A*. Its estimate of what is left, the octile distance
 * times the cheapest cost on the grid, never exceeds what a step there can cost, so the first time the search
 * settles the end it has found the cheapest path.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "heap.h"
#include "search.h"

/* What a search knows of a tile: nothing yet, a cost to it that may still fall, or its final cost. */
enum { UNREACHED = 0, REACHED = 1, SETTLED = 2 };

SearchStatus find_cheapest_path(
    const Grid *grid, int start, int end, double cheapest, int **tiles, int *tile_count, double *cost,
    double *distance
) {
    int height = grid->height;
    int end_i = end / height, end_j = end % height;
    *tiles = NULL;
    *tile_count = 0;
    *cost = *distance = INFINITY;
    if (start == end) {
        int *path = malloc(2 * sizeof *path);
        if (!path) return SEARCH_OUT_OF_MEMORY;
        path[0] = end_i;
        path[1] = end_j;
        *tiles = path;
        *tile_count = 1;
        *cost = *distance = 0.0;
        return SEARCH_DONE;
    }
    if (!is_passable(grid, start / height, start % height) || !is_passable(grid, end_i, end_j)) return SEARCH_DONE;

    size_t count = (size_t)grid->width * (size_t)height;
    double *best = malloc(count * sizeof *best);
    int *parent = malloc(count * sizeof *parent);
    uint8_t *state = calloc(count, 1);
    Heap heap = {0};
    SearchStatus status = SEARCH_DONE;
    if (!best || !parent || !state) {
        status = SEARCH_OUT_OF_MEMORY;
        goto done;
    }
    state[start] = REACHED;
    best[start] = 0.0;
    parent[start] = -1;
    if (heap_push(&heap, cheapest * octile_distance(start / height, start % height, end_i, end_j), start)) {
        status = SEARCH_OUT_OF_MEMORY;
        goto done;
    }

    while (heap.size > 0) {
        int tile = heap_pop(&heap).tile;
        if (state[tile] == SETTLED) continue;
        state[tile] = SETTLED;
        if (tile == end) {
            status = write_path(grid, end, parent, tiles, tile_count, distance);
            if (status == SEARCH_DONE) *cost = best[end];
            break;
        }
        int i = tile / height, j = tile % height;
        for (int step = 0; step < 8; step++) {
            int di = STEP_I[step], dj = STEP_J[step];
            if (!can_step(grid, i, j, di, dj)) continue;
            int neighbor = tile + di * height + dj;
            if (state[neighbor] == SETTLED) continue;
            double neighbor_cost = best[tile] + step_cost(grid, tile, di, dj);
            if (state[neighbor] == UNREACHED || neighbor_cost < best[neighbor]) {
                state[neighbor] = REACHED;
                best[neighbor] = neighbor_cost;
                parent[neighbor] = tile;
                double estimate = neighbor_cost + cheapest * octile_distance(i + di, j + dj, end_i, end_j);
                if (heap_push(&heap, estimate, neighbor)) {
                    status = SEARCH_OUT_OF_MEMORY;
                    goto done;
                }
            }
        }
    }
done:
    heap_free(&heap);
    free(best);
    free(parent);
    free(state);
    return status;
}
