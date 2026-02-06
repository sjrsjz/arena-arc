use arc_slice::Allocator;
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::sync::Arc;

fn bench_allocs(c: &mut Criterion) {
    let mut group = c.benchmark_group("alloc_compare");

    for &len in &[32usize, 256, 1024, 4096] {
        group.throughput(Throughput::Elements(len as u64));

        group.bench_with_input(BenchmarkId::new("fast_allocator", len), &len, |b, &len| {
            let mut allocator: Allocator<u64, u32, 65536> = Allocator::new();
            b.iter(|| {
                let handle = allocator.alloc(len.try_into().expect("len fits u32"), |i| i as u64);
                black_box(handle.get());
            });
        });

        group.bench_with_input(BenchmarkId::new("arc_slice", len), &len, |b, &len| {
            b.iter(|| {
                let mut arc = Arc::<[u64]>::new_uninit_slice(len);
                let data = Arc::get_mut(&mut arc).expect("unique during init");
                for (i, slot) in data.iter_mut().enumerate() {
                    slot.write(i as u64);
                }
                let arc: Arc<[u64]> = unsafe { arc.assume_init() };
                black_box(&*arc);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_allocs);
criterion_main!(benches);
