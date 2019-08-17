#include <array>
#include <iostream>

#include <emmintrin.h>
#include <immintrin.h>

#include <benchmark/benchmark.h>

void add_pair_manual(const std::array<double, 2> &v1,
                     const std::array<double, 2> &v2,
                     std::array<double, 2> &out) {
    out[0] = v1[0] + v2[0];
    out[1] = v1[1] + v2[1];
}

void add_pair_sse2(const std::array<double, 2> &v1,
                   const std::array<double, 2> &v2,
                   std::array<double, 2> &out) {
    _mm_store_pd(out.data(),
                 _mm_add_pd(_mm_load_pd(v1.data()),
                            _mm_load_pd(v2.data())));
}

void add_quad_manual(const std::array<double, 4> &l,
                     const std::array<double, 4> &r,
                     std::array<double, 4> &out) {
    out[0] = l[0] + r[0];
    out[1] = l[1] + r[1];
    out[2] = l[2] + r[2];
    out[3] = l[3] + r[3];
}

void add_quad_avx2_aligned(const double *l,
                           const double *r,
                           double *out) {
    _mm256_store_pd(out,
                    _mm256_add_pd(_mm256_load_pd(l),
                                  _mm256_load_pd(r)));
}

void add_quad_avx2(const std::array<double, 4> &l,
                   const std::array<double, 4> &r,
                   std::array<double, 4> &out) {
    _mm256_storeu_pd(out.data(),
                     _mm256_add_pd(_mm256_loadu_pd(l.data()),
                                   _mm256_loadu_pd(r.data())));
}

int main(int argc, char *argv[]) {
    std::cout << sizeof(double) << std::endl;
    std::cout << sizeof(std::array<double, 2>) << std::endl;
    std::cout << sizeof(std::array<double, 4>) << std::endl;

    const std::array<double, 2> v1 { 1.0, 2.0 };
    const std::array<double, 2> v2 { 3.0, 4.0 };
    std::array<double, 2> res;

    std::cout << "SSE2 addition of length 2 vectors" << std::endl;
    res = {0.0, 0.0};
    std::cout << res[0] << " " << res[1] << std::endl;
    add_pair_manual(v1, v2, res);
    std::cout << res[0] << " " << res[1] << std::endl;
    res = {0.0, 0.0};
    std::cout << res[0] << " " << res[1] << std::endl;
    add_pair_sse2(v1, v2, res);
    std::cout << res[0] << " " << res[1] << std::endl;

    const std::array<double, 4> v3 { 1.0, 2.0, 3.0, 4.0 };
    const std::array<double, 4> v4 { 5.0, 6.0, 7.0, 8.0 };
    std::array<double, 4> res2;

    std::cout << "AVX2 addition of length 4 vectors" << std::endl;
    res2 = {0.0, 0.0, 0.0, 0.0};
    std::cout << res2[0] << " " << res2[1] << " " << res2[2] << " " << res2[3] << std::endl;
    add_quad_manual(v3, v4, res2);
    std::cout << res2[0] << " " << res2[1] << " " << res2[2] << " " << res2[3] << std::endl;
    res2 = {0.0, 0.0, 0.0, 0.0};
    std::cout << res2[0] << " " << res2[1] << " " << res2[2] << " " << res2[3] << std::endl;
    add_quad_avx2(v3, v4, res2);
    std::cout << res2[0] << " " << res2[1] << " " << res2[2] << " " << res2[3] << std::endl;
    // add_quad_avx2_aligned(v3.data(), v4.data(), res2.data());
    // std::cout << res2[0] << " " << res2[1] << " " << res2[2] << " " << res2[3] << std::endl;

    // int argc, char *argv[]
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}

static void BM_add_pair_manual(benchmark::State &state) {
    const std::array<double, 2> v1 { 1.0, 2.0 };
    const std::array<double, 2> v2 { 3.0, 4.0 };
    std::array<double, 2> res;
    for (auto _ : state) {
        res = {0.0, 0.0};
        add_pair_manual(v1, v2, res);
    }
}

static void BM_add_pair_sse2(benchmark::State &state) {
    const std::array<double, 2> v1 { 1.0, 2.0 };
    const std::array<double, 2> v2 { 3.0, 4.0 };
    std::array<double, 2> res;
    for (auto _ : state) {
        res = {0.0, 0.0};
        add_pair_sse2(v1, v2, res);
    }
}

static void BM_add_quad_manual(benchmark::State &state) {
    const std::array<double, 4> v3 { 1.0, 2.0, 3.0, 4.0 };
    const std::array<double, 4> v4 { 5.0, 6.0, 7.0, 8.0 };
    std::array<double, 4> res2;
    for (auto _ : state) {
        res2 = {0.0, 0.0, 0.0, 0.0};
        add_quad_manual(v3, v4, res2);
    }
}

static void BM_add_quad_avx2(benchmark::State &state) {
    const std::array<double, 4> v3 { 1.0, 2.0, 3.0, 4.0 };
    const std::array<double, 4> v4 { 5.0, 6.0, 7.0, 8.0 };
    std::array<double, 4> res2;
    for (auto _ : state) {
        res2 = {0.0, 0.0, 0.0, 0.0};
        add_quad_avx2(v3, v4, res2);
    }
}

BENCHMARK(BM_add_pair_manual);
BENCHMARK(BM_add_pair_sse2);
BENCHMARK(BM_add_quad_manual);
BENCHMARK(BM_add_quad_avx2);

// BENCHMARK_MAIN();
