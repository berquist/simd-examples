#pragma once

#include <array>
#include <memory>

// Taken from https://en.cppreference.com/w/cpp/memory/align
template <std::size_t N>
struct MyAllocator {
    char data[N];
    void* p;
    std::size_t sz;

    MyAllocator() : p(data), sz(N) {}

    template <typename T> T *aligned_alloc(std::size_t a = alignof(T)) {
        if (std::align(a, sizeof(T), p, sz)) {
            T* result = reinterpret_cast<T*>(p);
            p = (char*)p + sizeof(T);
            sz -= sizeof(T);
            return result;
        }
        return nullptr;
    }
};

// Taken from https://en.cppreference.com/w/cpp/language/alignas
struct alignas(32) avx2_t
{
    double avx2_data[4];
};

void add_pair_manual(const std::array<double, 2> &v1,
                     const std::array<double, 2> &v2,
                     std::array<double, 2> &out);

void add_pair_sse2(const std::array<double, 2> &v1,
                   const std::array<double, 2> &v2,
                   std::array<double, 2> &out);

void add_quad_manual(const std::array<double, 4> &l,
                     const std::array<double, 4> &r,
                     std::array<double, 4> &out);

void add_quad_avx2_aligned(const double *l,
                           const double *r,
                           double *out);

void add_quad_avx2(const std::array<double, 4> &l,
                   const std::array<double, 4> &r,
                   std::array<double, 4> &out);
