/* The cheapest path from one tile to every tile, by Dijkstra's algorithm: each path's cost, its length and the tile
   it arrives from. */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "heap.h"
#include "search.h"

SearchStatus fill_flood(const Grid *grid, int start, double *costs, double *distances, int *parents) {
    int height = grid->height;
    size_t tile_count = (size_t)grid->width * (size_t)height;
    /* A tile's parent is written when it is reached, and read only where its cost is finite. */
    for (size_t tile = 0; tile < tile_count; tile++) costs[tile] = distances[tile] = INFINITY;
    costs[start] = distances[start] = 0.0;
    parents[start] = -1;
    if (!is_passable(grid, start / height, start % height)) return SEARCH_DONE;

    uint8_t *settled = calloc(tile_count, 1);
    if (!settled) return SEARCH_OUT_OF_MEMORY;
    Heap heap = {0};
    SearchStatus status = SEARCH_DONE;
    if (heap_push(&heap, 0.0, start)) {
        status = SEARCH_OUT_OF_MEMORY;
        goto done;
    }
    while (heap.size > 0) {
        int tile = heap_pop(&heap).tile;
        if (settled[tile]) continue;
        settled[tile] = 1;
        int i = tile / height, j = tile % height;
        for (int step = 0; step < 8; step++) {
            int di = STEP_I[step], dj = STEP_J[step];
            if (!can_step(grid, i, j, di, dj)) continue;
            int neighbor = tile + di * height + dj;
            if (settled[neighbor]) continue;
            double cost = costs[tile] + step_cost(grid, tile, di, dj);
            if (cost < costs[neighbor]) {
                costs[neighbor] = cost;
                distances[neighbor] = distances[tile] + (di && dj ? SQRT_2 : 1.0);
                parents[neighbor] = tile;
                if (heap_push(&heap, cost, neighbor)) {
                    status = SEARCH_OUT_OF_MEMORY;
                    goto done;
                }
            }
        }
    }
done:
    heap_free(&heap);
    free(settled);
    return status;
}
