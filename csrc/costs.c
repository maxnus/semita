/* The costs a grid gives its tiles, checked before a search reads them. */

#include <math.h>
#include <stddef.h>

#include "search.h"

SearchStatus check_costs(const Grid *grid, double *cheapest, double *dearest, int *bad_tile) {
    size_t tile_count = (size_t)grid->width * (size_t)grid->height;
    double lowest = INFINITY, highest = 0.0;
    for (size_t tile = 0; tile < tile_count; tile++) {
        double cost = grid->costs[tile];
        if (cost == 0.0 || cost == INFINITY) continue;
        /* Written so that NaN fails it too. */
        if (!(cost > 0.0)) {
            *bad_tile = (int)tile;
            return SEARCH_BAD_COST;
        }
        if (cost < lowest) lowest = cost;
        if (cost > highest) highest = cost;
    }
    *cheapest = lowest;
    *dearest = highest;
    return SEARCH_DONE;
}
