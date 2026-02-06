//! A fast, chunk-based allocator for building contiguous slices and sharing them via handles.
//!
//! This crate provides [`Allocator`] to allocate variable-length slices from a pre-allocated
//! chunk and return zero-copy [`Handle`]s. The handles share the underlying buffer and are cheap
//! to clone.
//!
//! # Safety notes
//! - **Zombie objects**: When a previously allocated slice is no longer used but the same chunk
//!   is still held by other handles, that slice’s memory is not reclaimed individually. It is
//!   only reclaimed when the entire chunk is freed.
//! - **Extra memory overhead**: Chunks are allocated with a fixed capacity `N`, which can leave
//!   unused space. When a request exceeds the remaining space, a new chunk is allocated and the
//!   unused tail of the old chunk becomes additional overhead.
//! - **Deferred drops**: Elements are dropped only when the last handle for a chunk is released,
//!   so resource reclamation does not happen per element immediately.
//!
//! # Example
//! ```
//! use fast_allocator::Allocator;
//!
//! let mut allocator: Allocator<u32, u32, 16> = Allocator::new();
//! let handle = allocator.alloc(4, |i| (i * 2) as u32);
//! assert_eq!(handle.get(), &[0, 2, 4, 6]);
//! ```
use std::{cell::UnsafeCell, mem::MaybeUninit, ptr::NonNull, sync::atomic::AtomicUsize};
/// Trait bound for index types used by [`Handle`] and [`Allocator`].
///
/// Must be convertible to and from `usize` with debug-friendly errors.
pub trait IndexType:
    Copy + TryFrom<usize, Error: std::fmt::Debug> + TryInto<usize, Error: std::fmt::Debug>
{
}
impl IndexType for u16 {}
impl IndexType for u32 {}
impl IndexType for u64 {}
impl IndexType for u128 {}
impl IndexType for usize {}

#[repr(C)]
struct Header {
    // The reference count for the buffer, used to determine when it can be safely deallocated.
    ref_count: AtomicUsize,
    // The index of the next item to be allocated.
    alloc_index: AtomicUsize,
    // The capacity of the buffer.
    capacity: usize,
}

// #[repr(C)] is prevent the compiler from reordering the fields of the struct,
// which is important for our use case because we need to be able to safely access the fields of the buffer from multiple threads without worrying
#[repr(C)]
struct Buffer<T> {
    // The header of the buffer, containing the reference count, allocation index, and capacity.
    header: Header,
    // The buffer data, stored as an array of T
    data: UnsafeCell<[MaybeUninit<T>]>,
}

impl<T> Buffer<T> {
    fn left_space(&self) -> usize {
        self.header.capacity
            - self
                .header
                .alloc_index
                .load(std::sync::atomic::Ordering::Acquire)
    }
}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        // get the allocated length of the buffer
        let len = self
            .header
            .alloc_index
            .load(std::sync::atomic::Ordering::Acquire);
        // SAFETY: We are the only thread that can access the buffer, so it is safe to drop the items in the buffer.
        unsafe {
            let data = &mut *self.data.get();
            data.iter_mut().take(len).for_each(|item| {
                item.assume_init_drop();
            });
        }
    }
}

/// A reference-counted pointer to an allocation chunk.
///
/// This is an internal building block used by [`Handle`] and [`Allocator`].
struct ChunkRef<T> {
    inner: NonNull<Buffer<T>>,
}

impl<T> Clone for ChunkRef<T> {
    fn clone(&self) -> Self {
        // Increment the reference count of the buffer.
        let buffer = unsafe { self.inner.as_ref() };
        buffer
            .header
            .ref_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self { inner: self.inner }
    }
}

impl<T> Drop for ChunkRef<T> {
    fn drop(&mut self) {
        // Decrement the reference count of the buffer.
        let buffer = unsafe { self.inner.as_ref() };
        if buffer
            .header
            .ref_count
            .fetch_sub(1, std::sync::atomic::Ordering::AcqRel)
            == 1
        {
            // SAFETY: We are the last owner of the buffer, so it is safe to deallocate it.
            unsafe {
                let layout = std::alloc::Layout::for_value(self.inner.as_ref());
                std::ptr::drop_in_place(self.inner.as_ptr());
                std::alloc::dealloc(self.inner.as_ptr() as *mut u8, layout);
            }
        }
    }
}

impl<T> ChunkRef<T> {
    fn new(capacity: usize) -> Self {
        // SAFETY: We create a layout for the header and the data array.
        let header_layout = std::alloc::Layout::new::<Header>();
        let array_layout = std::alloc::Layout::array::<MaybeUninit<T>>(capacity)
            .expect("Failed to create array layout for buffer data");
        let (layout, _offset) = header_layout
            .extend(array_layout)
            .expect("Failed to extend header layout with data layout");

        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        // Construct a fat pointer to the buffer with the correct length metadata.
        let ptr = std::ptr::slice_from_raw_parts_mut(ptr as *mut (), capacity) as *mut Buffer<T>;

        unsafe {
            // Initialize the header of the buffer.
            let header = &mut (*ptr).header;
            header
                .ref_count
                .store(1, std::sync::atomic::Ordering::Release);
            header
                .alloc_index
                .store(0, std::sync::atomic::Ordering::Release);
            header.capacity = capacity;
        }

        Self {
            inner: NonNull::new(ptr).expect("Failed to create NonNull pointer"),
        }
    }

    unsafe fn buffer(&self) -> &Buffer<T> {
        unsafe { self.inner.as_ref() }
    }
}

/// A read-only handle to a slice allocated from an [`Allocator`].
///
/// Cloning a handle is cheap and shares the underlying buffer.
pub struct SliceArc<T, L: IndexType = u32> {
    // A pointer to the buffer that this handle is allocated from.
    chunk: ChunkRef<T>,
    // The index of the item in the buffer that this handle points to.
    index: L,
    // The slice length of the item that this handle points to.
    len: L,
}

unsafe impl<T, L: IndexType> Send for SliceArc<T, L> where T: Send + Sync {}
unsafe impl<T, L: IndexType> Sync for SliceArc<T, L> where T: Send + Sync {}

impl<T, L: IndexType> Clone for SliceArc<T, L> {
    fn clone(&self) -> Self {
        Self {
            chunk: self.chunk.clone(),
            index: self.index,
            len: self.len,
        }
    }
}

impl<T, L: IndexType> SliceArc<T, L> {
    pub fn get(&self) -> &[T] {
        unsafe {
            let buffer = self.chunk.buffer();
            let data = &*buffer.data.get();
            let start = self
                .index
                .try_into()
                .expect("Index exceeds index type capacity");
            let len = self
                .len
                .try_into()
                .expect("Length exceeds index type capacity");
            let slice = &data[start..start + len];
            // SAFETY: The data in this range has been initialized by the allocator
            std::mem::transmute::<&[MaybeUninit<T>], &[T]>(slice)
        }
    }
}

/// Chunk-based allocator for variable-length slices.
///
/// - `T`: element type.
/// - `L`: index type used for slice offsets (defaults to `u32`).
/// - `N`: default chunk capacity.
pub struct Allocator<T, L: IndexType = u32, const N: usize = 256> {
    chunk: ChunkRef<T>,
    phantom: std::marker::PhantomData<L>,
}

impl<T, L: IndexType, const N: usize> std::fmt::Debug for Allocator<T, L, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Allocator<{}, {}, {}>",
            std::any::type_name::<T>(),
            std::any::type_name::<L>(),
            N
        )
    }
}

impl<T, L: IndexType, const N: usize> Default for Allocator<T, L, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, L: IndexType, const N: usize> Allocator<T, L, N> {
    /// Create a new allocator with a single chunk of capacity `N`.
    pub fn new() -> Self {
        Self {
            chunk: ChunkRef::new(N),
            phantom: std::marker::PhantomData,
        }
    }

    /// Allocate a slice of length `len` and initialize elements with `init`.
    ///
    /// Returns a [`Handle`] that can be cheaply cloned.
    pub fn alloc<F>(&mut self, len: L, mut init: F) -> SliceArc<T, L>
    where
        F: FnMut(usize) -> T,
    {
        let len: usize = len.try_into().expect("Length exceeds index type capacity");
        let left_space = unsafe { self.chunk.buffer().left_space() };
        if N < len {
            // 如果请求的长度超过了chunk的容量，直接分配一个新的chunk，但不改变当前chunk的状态，以便后续的分配仍然可以使用当前chunk。
            let chunk: ChunkRef<T> = ChunkRef::new(len);
            unsafe {
                let buffer = chunk.buffer(); // 我们可以保证这个chunk是唯一的，所以可以安全地获取可变引用
                // 使用FnOnce初始化chunk的数据
                let data = &mut *buffer.data.get();
                for (i, item) in data.iter_mut().take(len).enumerate() {
                    item.as_mut_ptr().write(init(i));
                }
                // 更新chunk的分配索引
                buffer
                    .header
                    .alloc_index
                    .store(len, std::sync::atomic::Ordering::Release);
            }
            return SliceArc {
                chunk,
                index: 0
                    .try_into()
                    .expect("Index 0 should be valid for any index type"),
                len: len.try_into().expect("Length exceeds index type capacity"),
            };
        }
        if left_space < len {
            // 如果当前chunk剩余空间不足以分配请求的长度
            let chunk: ChunkRef<T> = ChunkRef::new(N);
            unsafe {
                let buffer = chunk.buffer(); // 我们可以保证这个chunk是唯一的，所以可以安全地获取可变引用
                // 使用FnOnce初始化chunk的数据
                let data = &mut *buffer.data.get();
                for (i, item) in data.iter_mut().take(len).enumerate() {
                    item.as_mut_ptr().write(init(i));
                }
                // 更新chunk的分配索引
                buffer
                    .header
                    .alloc_index
                    .store(len, std::sync::atomic::Ordering::Release);
            }
            if left_space < N - len {
                // 如果当前chunk剩余空间不足以分配请求的长度，但重分配新的chunk再填充的剩余空间比比self持有的剩余空间更大，则直接丢弃当前chunk，使用新的chunk。
                self.chunk = chunk.clone();
            }
            return SliceArc {
                chunk,
                index: 0
                    .try_into()
                    .expect("Index 0 should be valid for any index type"),
                len: len.try_into().expect("Length exceeds index type capacity"),
            };
        }
        // 如果当前chunk剩余空间足以分配请求的长度
        unsafe {
            let buffer = self.chunk.buffer();

            // 1. 读取 index
            let index = buffer
                .header
                .alloc_index
                .load(std::sync::atomic::Ordering::Relaxed);

            // 2. 使用裸指针写入，避免创建 &mut [T]
            // 获取 *mut [MaybeUninit<T>] -> *mut MaybeUninit<T>
            let base_ptr = (*buffer.data.get()).as_mut_ptr();

            for i in 0..len {
                // init(i) 可能会 panic，但这在这里是安全的（Leak on panic）
                let val = init(i);
                // ptr::write 只操作特定地址，不产生大范围的引用别名限制
                base_ptr.add(index + i).write(MaybeUninit::new(val));
            }
            // 3. 提交 index
            buffer
                .header
                .alloc_index
                .store(index + len, std::sync::atomic::Ordering::Release);
            SliceArc {
                chunk: self.chunk.clone(),
                index: index.try_into().expect("Index cap"),
                len: len.try_into().expect("Length cap"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_allocator() {
        let mut allocator: Allocator<u32, u32, 16> = Allocator::new();
        let handle1 = allocator.alloc(10, |i| i as u32);
        let handle2 = allocator.alloc(20, |i| (i + 10) as u32);
        let handle3 = handle1.clone();
        assert_eq!(handle1.get(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(
            handle2.get(),
            &[
                10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29
            ]
        );
        assert_eq!(handle3.get(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[derive(Debug)]
    struct DropCounter<'a> {
        #[allow(dead_code)]
        value: usize,
        drops: &'a AtomicUsize,
    }

    impl<'a> Drop for DropCounter<'a> {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn test_drop_runs_exactly_once_per_item() {
        static DROPS: AtomicUsize = AtomicUsize::new(0);

        {
            let mut allocator: Allocator<DropCounter, u32, 8> = Allocator::new();
            let handle = allocator.alloc(8, |i| DropCounter {
                value: i,
                drops: &DROPS,
            });
            assert_eq!(handle.get().len(), 8);
            // Cloning handle should not affect drop count until all handles are dropped.
            let _handle2 = handle.clone();
            assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        }

        // All 8 items must be dropped exactly once when the chunk is freed.
        assert_eq!(DROPS.load(Ordering::SeqCst), 8);
    }

    #[test]
    fn test_multiple_chunks_and_large_allocs() {
        static DROPS: AtomicUsize = AtomicUsize::new(0);

        {
            let mut allocator: Allocator<DropCounter, u32, 4> = Allocator::new();
            let h1 = allocator.alloc(3, |i| DropCounter {
                value: i,
                drops: &DROPS,
            });
            let h2 = allocator.alloc(2, |i| DropCounter {
                value: i + 3,
                drops: &DROPS,
            });
            let h3 = allocator.alloc(10, |i| DropCounter {
                value: i + 5,
                drops: &DROPS,
            });

            assert_eq!(h1.get().len(), 3);
            assert_eq!(h2.get().len(), 2);
            assert_eq!(h3.get().len(), 10);
            let _ = (h1, h2, h3);
        }

        // 3 + 2 + 10 items dropped exactly once.
        assert_eq!(DROPS.load(Ordering::SeqCst), 15);
    }

    #[test]
    fn test_zst_allocations() {
        // Ensure zero-sized types are handled without UB.
        let mut allocator: Allocator<(), u32, 16> = Allocator::new();
        let h1 = allocator.alloc(0, |_| ());
        let h2 = allocator.alloc(8, |_| ());
        assert_eq!(h1.get().len(), 0);
        assert_eq!(h2.get().len(), 8);
    }

    #[test]
    fn test_handle_survives_chunk_rotation() {
        let mut allocator: Allocator<u32, u32, 4> = Allocator::new();
        let h1 = allocator.alloc(4, |i| i as u32);
        // Force new chunk and potentially replace current chunk.
        let _h2 = allocator.alloc(3, |i| (i + 10) as u32);
        let h3 = allocator.alloc(4, |i| (i + 20) as u32);

        assert_eq!(h1.get(), &[0, 1, 2, 3]);
        assert_eq!(h3.get(), &[20, 21, 22, 23]);
    }

    #[test]
    fn test_many_small_allocations() {
        let mut allocator: Allocator<u32, u32, 8> = Allocator::new();
        let mut handles = Vec::new();
        for i in 0..32 {
            let h = allocator.alloc(1, |_| i as u32);
            handles.push(h);
        }
        for (i, h) in handles.iter().enumerate() {
            assert_eq!(h.get(), &[(i as u32)]);
        }
    }

    #[cfg(miri)]
    mod miri_tests {
        use super::*;
        use std::cell::Cell;

        #[test]
        fn miri_stress_realloc_and_drop() {
            // Exercise multiple allocation paths under Miri to catch UB.
            let mut allocator: Allocator<u64, u32, 8> = Allocator::new();
            let h1 = allocator.alloc(8, |i| i as u64);
            let h2 = allocator.alloc(1, |i| (i + 100) as u64);
            let h3 = allocator.alloc(16, |i| (i + 200) as u64);
            assert_eq!(h1.get()[0], 0);
            assert_eq!(h2.get()[0], 100);
            assert_eq!(h3.get()[0], 200);
            let _ = (h1, h2, h3);
        }

        #[derive(Debug)]
        struct Poison<'a> {
            alive: &'a Cell<bool>,
        }

        impl<'a> Drop for Poison<'a> {
            fn drop(&mut self) {
                // If this ever becomes false before drop, it indicates a use-after-free.
                self.alive.set(false);
            }
        }

        #[test]
        fn miri_use_after_free_guard() {
            let flag = Cell::new(true);
            {
                let mut allocator: Allocator<Poison, u32, 4> = Allocator::new();
                let handle = allocator.alloc(4, |_| Poison { alive: &flag });
                assert!(flag.get());
                drop(handle);
                // Dropping just the handle should not drop the chunk.
                assert!(flag.get());
                // allocator dropped at end of scope, which should drop the chunk.
            }
            // Flag should be flipped by drops once the chunk is freed.
            assert!(!flag.get());
        }

        #[test]
        fn miri_many_small_allocations() {
            let mut allocator: Allocator<u64, u32, 4> = Allocator::new();
            let mut handles = Vec::new();
            for i in 0..64u64 {
                let h = allocator.alloc(1, |_| i);
                handles.push(h);
            }
            for (i, h) in handles.iter().enumerate() {
                assert_eq!(h.get(), &[(i as u64)]);
            }
        }

        #[test]
        fn miri_clone_stress() {
            let mut allocator: Allocator<u32, u32, 8> = Allocator::new();
            let h = allocator.alloc(8, |i| i as u32);
            let mut clones = Vec::new();
            for _ in 0..32 {
                clones.push(h.clone());
            }
            for c in clones.iter() {
                assert_eq!(c.get()[0], 0);
            }
            drop(h);
            // All clones still valid until dropped.
            for c in clones.iter() {
                assert_eq!(c.get()[7], 7);
            }
        }
    }
}
