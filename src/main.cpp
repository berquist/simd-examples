#include <iostream>

#include <benchmark/benchmark.h>

#include "functions.h"

int main(int argc, char *argv[]) {
    std::cout << sizeof(double) << std::endl;
    std::cout << sizeof(std::array<double, 2>) << std::endl;
    std::cout << sizeof(std::array<double, 4>) << std::endl;

    MyAllocator<64> a;
 
    // allocate a char
    char* p1 = a.aligned_alloc<char>();
    if (p1)
        *p1 = 'a';
    std::cout << "allocated a char at " << (void*)p1 << '\n';
 
    // allocate an int
    int* p2 = a.aligned_alloc<int>();
    if (p2)
        *p2 = 1;
    std::cout << "allocated an int at " << (void*)p2 << '\n';
 
    // allocate an int, aligned at 32-byte boundary
    int* p3 = a.aligned_alloc<int>(32);
    if (p3)
        *p3 = 2;
    std::cout << "allocated an int at " << (void*)p3 << " (32 byte alignment)\n";

    // Need four doubles (4 * 8 bytes) plus the 32 bytes for the boundary
    MyAllocator<64> b;
    double* p4 = b.aligned_alloc<double>(32);
    if (p4) {
        p4[0] = 3.14159262;
        p4[1] = 2.0;
        p4[2] = 1.0;
        p4[3] = 3.0;
    }
    std::cout << "allocated a double at " << (void*)p4 << " (32 byte alignment)" << std::endl;

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

    MyAllocator<64> alloc_v3;
    MyAllocator<64> alloc_v4;
    MyAllocator<64> alloc_res;
    double* av3 = alloc_v3.aligned_alloc<double>(32);
    double* av4 = alloc_v4.aligned_alloc<double>(32);
    double* ares = alloc_res.aligned_alloc<double>(32);
    av3[0] = 1.0; av3[1] = 2.0; av3[2] = 3.0; av3[3] = 4.0;
    av4[0] = 5.0; av4[1] = 6.0; av4[2] = 7.0; av4[3] = 8.0;
    ares[0] = 0.0; ares[1] = 0.0; ares[2] = 0.0; ares[3] = 0.0;

    const avx2_t avx2_v3 = {1.0, 2.0, 3.0, 4.0};
    const avx2_t avx2_v4 = {5.0, 6.0, 7.0, 8.0};
    avx2_t avx2_res = {0.0, 0.0, 0.0, 0.0};

    std::cout << "AVX2 addition of length 4 vectors" << std::endl;
    std::cout << res2[0] << " " << res2[1] << " " << res2[2] << " " << res2[3] << std::endl;
    add_quad_manual(v3, v4, res2);
    std::cout << res2[0] << " " << res2[1] << " " << res2[2] << " " << res2[3] << std::endl;
    add_quad_avx2(v3, v4, res2);
    std::cout << res2[0] << " " << res2[1] << " " << res2[2] << " " << res2[3] << std::endl;
    add_quad_avx2_aligned(av3, av4, ares);
    std::cout << ares[0] << " " << ares[1] << " " << ares[2] << " " << ares[3] << std::endl;
    add_quad_avx2_aligned(avx2_v3.avx2_data, avx2_v4.avx2_data, avx2_res.avx2_data);
    std::cout << ares[0] << " " << ares[1] << " " << ares[2] << " " << ares[3] << std::endl;

    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}

static void BM_add_pair_manual(benchmark::State &state) {
    const std::array<double, 2> v1 { 1.0, 2.0 };
    const std::array<double, 2> v2 { 3.0, 4.0 };
    std::array<double, 2> res;
    for (auto _ : state) {
        add_pair_manual(v1, v2, res);
    }
}

static void BM_add_pair_sse2(benchmark::State &state) {
    const std::array<double, 2> v1 { 1.0, 2.0 };
    const std::array<double, 2> v2 { 3.0, 4.0 };
    std::array<double, 2> res;
    for (auto _ : state) {
        add_pair_sse2(v1, v2, res);
    }
}

static void BM_add_quad_manual(benchmark::State &state) {
    const std::array<double, 4> v3 { 1.0, 2.0, 3.0, 4.0 };
    const std::array<double, 4> v4 { 5.0, 6.0, 7.0, 8.0 };
    std::array<double, 4> res2;
    for (auto _ : state) {
        add_quad_manual(v3, v4, res2);
    }
}

static void BM_add_quad_avx2(benchmark::State &state) {
    const std::array<double, 4> v3 { 1.0, 2.0, 3.0, 4.0 };
    const std::array<double, 4> v4 { 5.0, 6.0, 7.0, 8.0 };
    std::array<double, 4> res2;
    for (auto _ : state) {
        add_quad_avx2(v3, v4, res2);
    }
}

static void BM_add_quad_avx2_aligned(benchmark::State &state) {
    MyAllocator<64> alloc_v3;
    MyAllocator<64> alloc_v4;
    MyAllocator<64> alloc_res;
    double* av3 = alloc_v3.aligned_alloc<double>(32);
    double* av4 = alloc_v4.aligned_alloc<double>(32);
    double* ares = alloc_res.aligned_alloc<double>(32);
    av3[0] = 1.0; av3[1] = 2.0; av3[2] = 3.0; av3[3] = 4.0;
    av4[0] = 5.0; av4[1] = 6.0; av4[2] = 7.0; av4[3] = 8.0;
    for (auto _ : state) {
        add_quad_avx2_aligned(av3, av4, ares);
    }
}

static void BM_add_quad_avx2_aligned_2(benchmark::State &state) {
    const avx2_t avx2_v3 = {1.0, 2.0, 3.0, 4.0};
    const avx2_t avx2_v4 = {5.0, 6.0, 7.0, 8.0};
    avx2_t avx2_res = {0.0, 0.0, 0.0, 0.0};
    for (auto _ : state) {
        add_quad_avx2_aligned(avx2_v3.avx2_data, avx2_v4.avx2_data, avx2_res.avx2_data);
    }
}

BENCHMARK(BM_add_pair_manual);
BENCHMARK(BM_add_pair_sse2);
BENCHMARK(BM_add_quad_manual);
BENCHMARK(BM_add_quad_avx2);
BENCHMARK(BM_add_quad_avx2_aligned);
BENCHMARK(BM_add_quad_avx2_aligned_2);
