/*
 * semita._search, the searches for Python, on the stable ABI. It reads and fills the buffers the Python layer hands
 * it, checks them, and releases the GIL while it searches.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <limits.h>
#include <string.h>

#include "search.h"

/* Read `object` as a grid of bools or bytes, C-contiguous and two-dimensional. */
static int read_grid(PyObject *object, Py_buffer *view, Grid *grid) {
    if (PyObject_GetBuffer(object, view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0) return -1;
    const char *format = view->format ? view->format : "B";
    if (view->ndim != 2 || view->itemsize != 1 || (strcmp(format, "?") != 0 && strcmp(format, "B") != 0)) {
        PyErr_SetString(PyExc_TypeError, "the grid must be a 2-D C-contiguous buffer of bool or uint8");
        PyBuffer_Release(view);
        return -1;
    }
    if (view->shape[0] > INT_MAX || view->shape[1] > INT_MAX ||
        (view->shape[1] > 0 && view->shape[0] > INT_MAX / view->shape[1])) {
        PyErr_SetString(PyExc_ValueError, "the grid has too many tiles");
        PyBuffer_Release(view);
        return -1;
    }
    grid->passable = (const uint8_t *)view->buf;
    grid->width = (int)view->shape[0];
    grid->height = (int)view->shape[1];
    return 0;
}

/* Read `object` as a writable, C-contiguous buffer of `count` items in `format`. */
static int read_output(PyObject *object, Py_buffer *view, const char *format, Py_ssize_t itemsize, Py_ssize_t count) {
    if (PyObject_GetBuffer(object, view, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0) return -1;
    if (!view->format || strcmp(view->format, format) != 0 || view->itemsize != itemsize ||
        view->len != itemsize * count) {
        PyErr_Format(PyExc_TypeError, "the output must be a C-contiguous buffer of %zd items of format '%s'", count,
                     format);
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

/* Raise for a search that did not finish; 0 if it did. */
static int check_status(SearchStatus status) {
    switch (status) {
    case SEARCH_DONE:
        return 0;
    case SEARCH_OUT_OF_MEMORY:
        PyErr_NoMemory();
        return -1;
    default:
        PyErr_SetString(PyExc_RuntimeError, "the search contradicted itself, which is a bug in semita");
        return -1;
    }
}

static PyObject *find_path(PyObject *module, PyObject *args) {
    (void)module;
    PyObject *grid_object, *tiles_object;
    int start_i, start_j, end_i, end_j;
    if (!PyArg_ParseTuple(args, "OiiiiO:find_path", &grid_object, &start_i, &start_j, &end_i, &end_j, &tiles_object)) {
        return NULL;
    }
    Py_buffer grid_view, tiles_view;
    Grid grid;
    if (read_grid(grid_object, &grid_view, &grid) < 0) return NULL;
    int start = tile_at(&grid, start_i, start_j, "start");
    int end = start < 0 ? -1 : tile_at(&grid, end_i, end_j, "end");
    if (end < 0) {
        PyBuffer_Release(&grid_view);
        return NULL;
    }
    Py_ssize_t pair_count = 2 * (Py_ssize_t)grid.width * grid.height;
    if (read_output(tiles_object, &tiles_view, "i", (Py_ssize_t)sizeof(int), pair_count) < 0) {
        PyBuffer_Release(&grid_view);
        return NULL;
    }
    int tile_count;
    double distance;
    SearchStatus status;
    Py_BEGIN_ALLOW_THREADS
    status = find_path_by_jump_points(&grid, start, end, (int *)tiles_view.buf, &tile_count, &distance);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&tiles_view);
    PyBuffer_Release(&grid_view);
    if (check_status(status) < 0) return NULL;
    return Py_BuildValue("(di)", distance, tile_count);
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
    if (start < 0) {
        PyBuffer_Release(&grid_view);
        return NULL;
    }
    Py_ssize_t tile_count = (Py_ssize_t)grid.width * grid.height;
    if (read_output(distances_object, &distances_view, "d", (Py_ssize_t)sizeof(double), tile_count) < 0) {
        PyBuffer_Release(&grid_view);
        return NULL;
    }
    SearchStatus status;
    Py_BEGIN_ALLOW_THREADS
    status = fill_distances(&grid, start, (double *)distances_view.buf);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&distances_view);
    PyBuffer_Release(&grid_view);
    if (check_status(status) < 0) return NULL;
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"find_path", find_path, METH_VARARGS,
     "find_path(passable, start_i, start_j, end_i, end_j, tiles, /)\n--\n\n"
     "Write a shortest path's tiles into `tiles` and return its length and its number of tiles, 0 if there is none."},
    {"distances_from", distances_from, METH_VARARGS,
     "distances_from(passable, start_i, start_j, distances, /)\n--\n\n"
     "Write every tile's distance from the start into `distances`."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT, "semita._search", "semita's searches, in C.", -1, methods, NULL, NULL, NULL, NULL,
};

PyMODINIT_FUNC PyInit__search(void) {
    return PyModule_Create(&module_definition);
}
