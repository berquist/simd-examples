#[macro_use]
extern crate criterion;

use simd_examples;

use criterion::Criterion;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("calculate_var <- mars::a0", |b| {
        b.iter(|| simd_examples::calculate_var(1989.0, &simd_examples::mars::A0))
    });
    c.bench_function("calculate_var_inline <- mars::a0", |b| {
        b.iter(|| simd_examples::calculate_var_inline(1989.0, &simd_examples::mars::A0))
    });
    c.bench_function("calculate_var_avx <- mars::a0", |b| {
        b.iter(|| simd_examples::calculate_var_avx(1989.0, &simd_examples::mars::A0))
    });
    c.bench_function("calculate_var_avx_inner <- mars::a0", |b| {
        b.iter(|| unsafe {
            simd_examples::calculate_var_avx_inner(1989.0, &simd_examples::mars::A0)
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
