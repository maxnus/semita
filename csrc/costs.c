/* The costs a grid gives its tiles, checked before a search reads them. */

#include <math.h>
#include <stddef.h>

#include "search.h"

SearchStatus check_costs(const Grid *grid, double *cheapest, int *bad_tile) {
    size_t tile_count = (size_t)grid->width * (size_t)grid->height;
    double lowest = INFINITY;
    for (size_t tile = 0; tile < tile_count; tile++) {
        if (!grid->passable[tile]) continue;
        double cost = grid->costs[tile];
        /* Written so that NaN fails it too. */
        if (!(cost > 0.0)) {
            *bad_tile = (int)tile;
            return SEARCH_BAD_COST;
        }
        if (cost < lowest) lowest = cost;
    }
    *cheapest = lowest;
    return SEARCH_DONE;
}
