## Benchmarks

### C++

`g++` without optimization
```
---------------------------------------------------------------------
Benchmark                           Time             CPU   Iterations
---------------------------------------------------------------------
BM_add_pair_manual               28.1 ns         28.1 ns     25085819
BM_add_pair_sse2                 15.9 ns         15.9 ns     43744342
BM_add_quad_manual               54.1 ns         54.1 ns     12685768
BM_add_quad_avx2                 17.6 ns         17.5 ns     39791525
BM_add_quad_avx2_aligned         6.22 ns         6.22 ns    105888792
BM_add_quad_avx2_aligned_2       5.72 ns         5.71 ns    114156850
```
`g++` with `-O3`
```
---------------------------------------------------------------------
Benchmark                           Time             CPU   Iterations
---------------------------------------------------------------------
BM_add_pair_manual               1.42 ns         1.42 ns    465881820
BM_add_pair_sse2                 1.18 ns         1.18 ns    588688698
BM_add_quad_manual               1.68 ns         1.67 ns    420161931
BM_add_quad_avx2                 2.38 ns         2.38 ns    295221323
BM_add_quad_avx2_aligned         2.13 ns         2.13 ns    329340867
BM_add_quad_avx2_aligned_2       2.13 ns         2.13 ns    329558417
```

Why? Because the Threadripper 2950X has 4 128-bit lanes rather than Intel's 2 256-bit lanes, so it takes roughly twice as long to perform an AVX2 instruction on an AMD chip.
