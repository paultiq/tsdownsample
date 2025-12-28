use downsample_rs::lttb as lttb_mod;
use downsample_rs::minmax as minmax_mod;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dev_utils::{config, utils};

// LTTB Comparison Benchmarks

fn lttb_baseline_vs_optimized_50M(c: &mut Criterion) {
    let n = 50_000_000;
    let data = utils::get_random_array::<f32>(n, f32::MIN, f32::MAX);

    let mut group = c.benchmark_group("lttb_50M_comparison");

    group.bench_function("baseline", |b| {
        b.iter(|| lttb_mod::lttb_without_x(black_box(data.as_slice()), black_box(2_000), black_box(false)))
    });

    group.bench_function("optimized", |b| {
        b.iter(|| lttb_mod::lttb_without_x(black_box(data.as_slice()), black_box(2_000), black_box(true)))
    });

    group.finish();
}

fn lttb_with_x_baseline_vs_optimized_50M(c: &mut Criterion) {
    let n = 50_000_000;
    let data = utils::get_random_array::<f32>(n, f32::MIN, f32::MAX);
    let x = (0..n).map(|i| i as i32).collect::<Vec<i32>>();

    let mut group = c.benchmark_group("lttb_with_x_50M_comparison");

    group.bench_function("baseline", |b| {
        b.iter(|| {
            lttb_mod::lttb_with_x(
                black_box(x.as_slice()),
                black_box(data.as_slice()),
                black_box(2_000),
                black_box(false),
            )
        })
    });

    group.bench_function("optimized", |b| {
        b.iter(|| {
            lttb_mod::lttb_with_x(
                black_box(x.as_slice()),
                black_box(data.as_slice()),
                black_box(2_000),
                black_box(true),
            )
        })
    });

    group.finish();
}

// MinMax Comparison Benchmarks

fn minmax_parallel_baseline_vs_optimized_50M(c: &mut Criterion) {
    let n = 50_000_000;
    let data = utils::get_random_array::<f32>(n, f32::MIN, f32::MAX);

    let mut group = c.benchmark_group("minmax_parallel_50M_comparison");

    group.bench_function("baseline", |b| {
        b.iter(|| {
            minmax_mod::min_max_without_x_parallel(black_box(data.as_slice()), black_box(2_000))
        })
    });

    group.bench_function("optimized", |b| {
        b.iter(|| {
            minmax_mod::min_max_without_x_parallel_opt(black_box(data.as_slice()), black_box(2_000))
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    lttb_baseline_vs_optimized_50M,
    lttb_with_x_baseline_vs_optimized_50M,
    minmax_parallel_baseline_vs_optimized_50M,
);
criterion_main!(benches);
