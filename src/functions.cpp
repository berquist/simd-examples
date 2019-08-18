#include <array>

#include <emmintrin.h>
#include <immintrin.h>

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

