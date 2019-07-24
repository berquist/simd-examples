#include <array>
#include <iostream>
#include <emmintrin.h>

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
    const auto sv1 = _mm_load_pd(v1.data());
    const auto sv2 = _mm_load_pd(v2.data());
    const auto sres = _mm_add_pd(sv1, sv2);
    out[0] = sres[0];
    out[1] = sres[1];
}

// int main() {
//     const std::array<double, 2> v1 { 1.0, 2.0 };
//     const std::array<double, 2> v2 { 3.0, 4.0 };
//     std::array<double, 2> res;

//     res = {0.0, 0.0};
//     std::cout << res[0] << " " << res[1] << std::endl;
//     add_pair_manual(v1, v2, res);
//     std::cout << res[0] << " " << res[1] << std::endl;
//     res = {0.0, 0.0};
//     std::cout << res[0] << " " << res[1] << std::endl;
//     add_pair_sse2(v1, v2, res);
//     std::cout << res[0] << " " << res[1] << std::endl;

//     return 0;
// }

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

BENCHMARK(BM_add_pair_manual);
BENCHMARK(BM_add_pair_sse2);

BENCHMARK_MAIN();
