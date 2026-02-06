# arena-arc

A fast, chunk-based allocator for building contiguous slices and sharing them via handles.
It is designed for high-frequency, short-lived allocations where allocating each slice
individually would be too expensive.

This crate provides an `Allocator` that allocates variable-length slices from a pre-allocated
chunk and returns zero-copy handles (`ArcSlice` / `ArcSingle`). Handles are cheap to clone and
share the underlying buffer.

## Features

- Allocate many slices from a single chunk
- Cheap handle cloning with zero-copy sharing
- Configurable chunk capacity and index type
- Optional fallible initialization (`try_alloc`)
- Single-item handles (`ArcSingle`) for ergonomic access

## Quick start

```rust
use arena_arc::Allocator;

let mut allocator: Allocator<u32, u32, 16> = Allocator::new();
let handle = allocator.alloc(4, |i| (i * 2) as u32);
assert_eq!(handle.get(), &[0, 2, 4, 6]);
```

## API overview

### `Allocator<T, L, N>`

- `T`: element type
- `L`: index type used for offsets (defaults to `u32`)
- `N`: default chunk capacity (const generic)

Common methods:

- `new()`: create a new allocator with a single chunk of capacity `N`
- `alloc(len, init)`: allocate a slice of length `len` and initialize each element via `init`
- `try_alloc(len, init)`: same as `alloc`, but allows returning an error
- `alloc_single(init)`: allocate one element and return an `ArcSingle`
- `alloc_value(value)`: allocate one element from an existing value

### `ArcSlice<T, L>`

Read-only handle to a slice allocated from an `Allocator`.

- `get() -> &[T]`: borrow the slice (zero-copy)
- `clone()`: cheap, shared handle clone

### `ArcSingle<T, L>`

Read-only handle to a single element.

- `get() -> &T`: borrow the element (zero-copy)
- `from_slice(handle)`: convert an `ArcSlice` of length 1 into an `ArcSingle`

## Safety and behavior notes

- **Zombie objects**: If a slice is no longer used while other handles keep the same chunk alive,
	that slice’s memory is not reclaimed individually. It is only reclaimed when the entire chunk
	is freed.
- **Extra memory overhead**: Chunks are allocated with fixed capacity `N`. When a request exceeds
	remaining space, a new chunk is allocated and the unused tail of the old chunk becomes overhead.
- **Deferred drops**: Elements are dropped only when the last handle for a chunk is released, so
	per-element drop does not happen immediately.

## When to use

- Building many short-lived slices with similar lifetimes
- Sharing read-only data across multiple owners without copying
- Avoiding per-slice heap allocations

## Benchmarks

Benchmarks compare against `Arc<[T]>` allocations.

```bash
cargo bench --bench alloc_bench
```

## Miri checks

```bash
cargo +nightly miri test
```

## License

MIT
