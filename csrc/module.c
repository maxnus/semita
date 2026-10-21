/*
 * semita._search, the searches for Python, on the stable ABI. It reads and fills the buffers the Python layer hands
 * it, checks them, and releases the GIL while it searches.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "search.h"

/* Read `object` as a grid, C-contiguous and two-dimensional: of bools, or of doubles, its costs. */
static int read_grid(PyObject *object, Py_buffer *view, Grid *grid) {
    if (PyObject_GetBuffer(object, view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0) return -1;
    const char *format = view->format ? view->format : "B";
    int of_bools = strcmp(format, "?") == 0 && view->itemsize == 1;
    int of_costs = strcmp(format, "d") == 0 && view->itemsize == (Py_ssize_t)sizeof(double);
    if (view->ndim != 2 || !(of_bools || of_costs)) {
        PyErr_SetString(PyExc_TypeError, "the grid must be a 2-D C-contiguous buffer of bool or double");
        PyBuffer_Release(view);
        return -1;
    }
    if (view->shape[0] > INT_MAX || view->shape[1] > INT_MAX ||
        (view->shape[1] > 0 && view->shape[0] > INT_MAX / view->shape[1])) {
        PyErr_SetString(PyExc_ValueError, "the grid has too many tiles");
        PyBuffer_Release(view);
        return -1;
    }
    grid->passable = of_bools ? (const uint8_t *)view->buf : NULL;
    grid->costs = of_costs ? (const double *)view->buf : NULL;
    grid->width = (int)view->shape[0];
    grid->height = (int)view->shape[1];
    return 0;
}

/* Read `object` as a writable, C-contiguous buffer of `count` doubles. */
static int read_output(PyObject *object, Py_buffer *view, Py_ssize_t count) {
    if (PyObject_GetBuffer(object, view, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0) return -1;
    if (!view->format || strcmp(view->format, "d") != 0 || view->itemsize != (Py_ssize_t)sizeof(double) ||
        view->len != (Py_ssize_t)sizeof(double) * count) {
        PyErr_Format(PyExc_TypeError, "the output must be a C-contiguous buffer of %zd items of format 'd'", count);
        PyBuffer_Release(view);
        return -1;
    }
    return 0;
}

/* The index of tile (i, j), or -1 with IndexError set if the grid has no such tile. */
static int tile_at(const Grid *grid, int i, int j, const char *which) {
    if (i < 0 || i >= grid->width || j < 0 || j >= grid->height) {
        PyErr_Format(PyExc_IndexError, "the %s (%d, %d) is outside the grid's %d by %d tiles", which, i, j,
                     grid->width, grid->height);
        return -1;
    }
    return i * grid->height + j;
}

/* Raise for a search that did not finish, while the grid can still be read; 0 if it did. */
static int check_status(SearchStatus status, const Grid *grid, int bad_tile) {
    switch (status) {
    case SEARCH_DONE:
        return 0;
    case SEARCH_OUT_OF_MEMORY:
        PyErr_NoMemory();
        return -1;
    case SEARCH_BAD_COST: {
        PyObject *cost = PyFloat_FromDouble(grid->costs[bad_tile]);
        if (cost) {
            PyErr_Format(PyExc_ValueError, "the cost at (%d, %d) is %R; a tile's cost is positive, or 0 or inf to "
                         "block it", bad_tile / grid->height, bad_tile % grid->height, cost);
            Py_DECREF(cost);
        }
        return -1;
    }
    default:
        PyErr_SetString(PyExc_RuntimeError, "the search contradicted itself, which is a bug in semita");
        return -1;
    }
}

/* The cheapest path from `start` to `end`: by jump point search on a grid of bools or of costs all the same, its
   length scaled by that cost, and by A* on any other grid of costs. */
static SearchStatus find_any_path(
    const Grid *grid, int start, int end, int **tiles, int *tile_count, double *cost, double *distance, int *bad_tile
) {
    double cheapest = 1.0, dearest = 1.0;
    if (grid->costs) {
        SearchStatus status = check_costs(grid, &cheapest, &dearest, bad_tile);
        if (status != SEARCH_DONE) return status;
        if (cheapest != dearest && cheapest < INFINITY) {
            return find_cheapest_path(grid, start, end, cheapest, tiles, tile_count, cost, distance);
        }
    }
    SearchStatus status = find_path_by_jump_points(grid, start, end, tiles, tile_count, distance);
    /* A path of no steps costs nothing, even on a grid with no passable tile to give a cost. */
    *cost = *distance == 0.0 ? 0.0 : cheapest * *distance;
    return status;
}

static PyObject *find_path(PyObject *module, PyObject *args) {
    (void)module;
    PyObject *grid_object;
    int start_i, start_j, end_i, end_j;
    if (!PyArg_ParseTuple(args, "Oiiii:find_path", &grid_object, &start_i, &start_j, &end_i, &end_j)) return NULL;
    Py_buffer grid_view;
    Grid grid;
    if (read_grid(grid_object, &grid_view, &grid) < 0) return NULL;
    int start = tile_at(&grid, start_i, start_j, "start");
    int end = start < 0 ? -1 : tile_at(&grid, end_i, end_j, "end");
    if (end < 0) {
        PyBuffer_Release(&grid_view);
        return NULL;
    }
    int *tiles = NULL, tile_count = 0, bad_tile = -1;
    double cost = INFINITY, distance = INFINITY;
    SearchStatus status;
    Py_BEGIN_ALLOW_THREADS
    status = find_any_path(&grid, start, end, &tiles, &tile_count, &cost, &distance, &bad_tile);
    Py_END_ALLOW_THREADS
    int failed = check_status(status, &grid, bad_tile);
    PyBuffer_Release(&grid_view);
    if (failed) {
        free(tiles);
        return NULL;
    }
    if (!tiles) return Py_BuildValue("(ddO)", cost, distance, Py_None);
    Py_ssize_t size = 2 * (Py_ssize_t)tile_count * (Py_ssize_t)sizeof *tiles;
    PyObject *path = PyBytes_FromStringAndSize((const char *)tiles, size);
    free(tiles);
    if (!path) return NULL;
    PyObject *result = Py_BuildValue("(ddO)", cost, distance, path);
    Py_DECREF(path);
    return result;
}

static PyObject *distances_from(PyObject *module, PyObject *args) {
    (void)module;
    PyObject *grid_object, *distances_object;
    int start_i, start_j;
    if (!PyArg_ParseTuple(args, "OiiO:distances_from", &grid_object, &start_i, &start_j, &distances_object)) {
        return NULL;
    }
    Py_buffer grid_view, distances_view;
    Grid grid;
    if (read_grid(grid_object, &grid_view, &grid) < 0) return NULL;
    int start = tile_at(&grid, start_i, start_j, "start");
    if (start < 0 || read_output(distances_object, &distances_view, (Py_ssize_t)grid.width * grid.height) < 0) {
        PyBuffer_Release(&grid_view);
        return NULL;
    }
    int bad_tile = -1;
    double cheapest, dearest;
    SearchStatus status = SEARCH_DONE;
    Py_BEGIN_ALLOW_THREADS
    if (grid.costs) status = check_costs(&grid, &cheapest, &dearest, &bad_tile);
    if (status == SEARCH_DONE) status = fill_distances(&grid, start, (double *)distances_view.buf);
    Py_END_ALLOW_THREADS
    int failed = check_status(status, &grid, bad_tile);
    PyBuffer_Release(&distances_view);
    PyBuffer_Release(&grid_view);
    if (failed) return NULL;
    Py_RETURN_NONE;
}

static PyMethodDef module_methods[] = {
    {"find_path", find_path, METH_VARARGS,
     "find_path(grid, start_i, start_j, end_i, end_j, /)\n--\n\n"
     "The cheapest path's cost, its length and its tiles as C ints, i and j in turn, or None for the tiles if there "
     "is none. The grid is a buffer of bools, or of doubles, its costs."},
    {"distances_from", distances_from, METH_VARARGS,
     "distances_from(grid, start_i, start_j, distances, /)\n--\n\n"
     "Write every tile's cost from the start into `distances`, its distance on a grid of bools."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT, "semita._search", "semita's searches, in C.", -1, module_methods, NULL, NULL, NULL, NULL,
};

PyMODINIT_FUNC PyInit__search(void) {
    return PyModule_Create(&module_definition);
}
