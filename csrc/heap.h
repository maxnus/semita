/*
 * A binary min-heap of tiles keyed by cost, growing as it fills. A tile may be pushed again at a lower cost; the
 * searches skip the entries they have already settled.
 */

#ifndef SEMITA_HEAP_H
#define SEMITA_HEAP_H

#include <stddef.h>
#include <stdlib.h>

typedef struct {
    double cost;
    int tile;
} HeapEntry;

typedef struct {
    HeapEntry *entries;
    size_t size;
    size_t capacity;
} Heap;

#define HEAP_FIRST_CAPACITY 256

/* Push `tile` at `cost`: 0 if pushed, -1 if memory ran out. */
static inline int heap_push(Heap *heap, double cost, int tile) {
    if (heap->size == heap->capacity) {
        size_t capacity = heap->capacity ? 2 * heap->capacity : HEAP_FIRST_CAPACITY;
        HeapEntry *entries = realloc(heap->entries, capacity * sizeof *entries);
        if (!entries) return -1;
        heap->entries = entries;
        heap->capacity = capacity;
    }
    size_t position = heap->size++;
    while (position > 0) {
        size_t parent = (position - 1) / 2;
        if (heap->entries[parent].cost <= cost) break;
        heap->entries[position] = heap->entries[parent];
        position = parent;
    }
    heap->entries[position].cost = cost;
    heap->entries[position].tile = tile;
    return 0;
}

/* Remove and return the cheapest entry. The heap must not be empty. */
static inline HeapEntry heap_pop(Heap *heap) {
    HeapEntry cheapest = heap->entries[0];
    HeapEntry last = heap->entries[--heap->size];
    size_t position = 0;
    for (;;) {
        size_t child = 2 * position + 1;
        if (child >= heap->size) break;
        if (child + 1 < heap->size && heap->entries[child + 1].cost < heap->entries[child].cost) child++;
        if (last.cost <= heap->entries[child].cost) break;
        heap->entries[position] = heap->entries[child];
        position = child;
    }
    if (heap->size > 0) heap->entries[position] = last;
    return cheapest;
}

static inline void heap_free(Heap *heap) {
    free(heap->entries);
    heap->entries = NULL;
    heap->size = heap->capacity = 0;
}

#endif /* SEMITA_HEAP_H */
