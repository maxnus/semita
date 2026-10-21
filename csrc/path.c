/* A path, written out from the parents a search left. */

#include <stdlib.h>

#include "search.h"

SearchStatus write_path(const Grid *grid, int end, const int *parent, int **tiles, int *tile_count, double *distance) {
    int height = grid->height;
    long long count = 1;
    double length = 0.0;
    long long grid_tiles = (long long)grid->width * height;
    for (int tile = end; parent[tile] != -1; tile = parent[tile]) {
        if (parent[tile] < 0 || parent[tile] >= grid_tiles) return SEARCH_BROKEN;
        int di = abs(tile / height - parent[tile] / height), dj = abs(tile % height - parent[tile] % height);
        if (di && dj && di != dj) return SEARCH_BROKEN;
        count += di > dj ? di : dj;
        length += di && dj ? SQRT_2 * di : (double)(di + dj);
        if (count > grid_tiles) return SEARCH_BROKEN;
    }
    int *path = malloc(2 * (size_t)count * sizeof *path);
    if (!path) return SEARCH_OUT_OF_MEMORY;
    int position = (int)count - 1;
    int i = end / height, j = end % height;
    path[2 * position] = i;
    path[2 * position + 1] = j;
    for (int tile = end; parent[tile] != -1; tile = parent[tile]) {
        int from_i = parent[tile] / height, from_j = parent[tile] % height;
        int back_i = (from_i > i) - (from_i < i), back_j = (from_j > j) - (from_j < j);
        while (i != from_i || j != from_j) {
            i += back_i;
            j += back_j;
            position--;
            path[2 * position] = i;
            path[2 * position + 1] = j;
        }
    }
    if (position != 0) {
        free(path);
        return SEARCH_BROKEN;
    }
    *tiles = path;
    *tile_count = (int)count;
    if (distance) *distance = length;
    return SEARCH_DONE;
}
