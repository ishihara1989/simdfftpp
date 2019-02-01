#pragma once

#include <x86intrin.h>

#include <complex>
#include <type_traits>
#include <vector>

#include <boost/align/aligned_allocator.hpp>

template<typename T, int N>
using SimdVector = std::vector<T, boost::alignment::aligned_allocator<T, N>>;

template<typename T, int N>
struct init_vector_struct{
    auto operator()(int size){
        return  SimdVector<T, N>(size);
    }
};

template<typename T, int N>
struct init_vector_struct<std::complex<T>,N>{
    auto operator()(int size){
        return  std::vector<std::complex<T>, boost::alignment::aligned_allocator<std::complex<T>, N/2>>(size);
    }
};

template<typename T, int N>
auto init_vector(int size){
    return  init_vector_struct<T, N>()(size);
}

template<typename T, int N>
static inline auto load(const T* src){
    if constexpr(std::is_same<float, T>::value&&N==4){
        return _mm_load_ps(src);
    } else if constexpr(std::is_same<float, T>::value&&N==8){
        return _mm256_load_ps(src);
    } else if constexpr(std::is_same<double, T>::value&&N==2){
        return _mm_load_pd(src);
    } else if constexpr(std::is_same<double, T>::value&&N==4){
        return _mm256_load_pd(src);
    } else {
        return *src;
    }
}

template<typename T, int N>
static inline auto load(const std::complex<T>* src){
    if constexpr(std::is_same<float, T>::value&&N==4){
        return _mm_load_ps(reinterpret_cast<const T*>(&src[0]));
    } else if constexpr(std::is_same<float, T>::value&&N==8){
        return _mm256_load_ps(reinterpret_cast<const T*>(&src[0]));
    } else if constexpr(std::is_same<double, T>::value&&N==2){
        return _mm_load_pd(reinterpret_cast<const T*>(&src[0]));
    } else if constexpr(std::is_same<double, T>::value&&N==4){
        return _mm256_load_pd(reinterpret_cast<const T*>(&src[0]));
    } else {
        return *src;
    }
}

static inline void store(float* dst, __m128 src){
    _mm_store_ps(dst, src);
}

static inline void store(float* dst, __m256 src){
    _mm256_store_ps(dst, src);
}

static inline void store(double* dst, __m128d src){
    _mm_store_pd(dst, src);
}

static inline void store(double* dst, __m256d src){
    _mm256_store_pd(dst, src);
}

static inline void store(std::complex<float>* dst, __m128 src){
    _mm_store_ps(reinterpret_cast<float*>(dst), src);
}

static inline void store(std::complex<float>* dst, __m256 src){
    _mm256_store_ps(reinterpret_cast<float*>(dst), src);
}

static inline void store(std::complex<double>* dst, __m128d src){
    _mm_store_pd(reinterpret_cast<double*>(dst), src);
}

static inline void store(std::complex<double>* dst, __m256d src){
    _mm256_store_pd(reinterpret_cast<double*>(dst), src);
}

static inline auto add(__m128 a, __m128 b){
    return _mm_add_ps(a, b);
}

static inline auto add(__m256 a, __m256 b){
    return _mm256_add_ps(a, b);
}

static inline auto add(__m128d a, __m128d b){
    return _mm_add_pd(a, b);
}

static inline auto add(__m256d a, __m256d b){
    return _mm256_add_pd(a, b);
}

static inline auto sub(__m128 a, __m128 b){
    return _mm_sub_ps(a, b);
}

static inline auto sub(__m256 a, __m256 b){
    return _mm256_sub_ps(a, b);
}

static inline auto sub(__m128d a, __m128d b){
    return _mm_sub_pd(a, b);
}

static inline auto sub(__m256d a, __m256d b){
    return _mm256_sub_pd(a, b);
}

static inline auto mul(__m128 a, __m128 b){
    return _mm_mul_ps(a, b);
}

static inline auto mul(__m256 a, __m256 b){
    return _mm256_mul_ps(a, b);
}

static inline auto mul(__m128d a, __m128d b){
    return _mm_mul_pd(a, b);
}

static inline auto mul(__m256d a, __m256d b){
    return _mm256_mul_pd(a, b);
}

static inline auto div(__m128 a, __m128 b){
    return _mm_div_ps(a, b);
}

static inline auto div(__m256 a, __m256 b){
    return _mm256_div_ps(a, b);
}

static inline auto div(__m128d a, __m128d b){
    return _mm_div_pd(a, b);
}

static inline auto div(__m256d a, __m256d b){
    return _mm256_div_pd(a, b);
}

static inline auto cmul(__m128 a, __m128 b){
    // [a,b,c,d] [e,f,g,h]->[ae-bf, af+be, cg-dh, ch+dg]
    //[ae, af, cg, ch] addsub [bf, be, dh, dg]
    auto r1 = _mm_permute_ps(a, 0b10100000); //[a, a, c, c]
    auto r2 = b; //[e, f, g, h]
    auto i1 = _mm_permute_ps(a, 0b11110101); //[b, b, d, d]
    auto i2 = _mm_permute_ps(b, 0b10110001); //[f, e, h, g]
    return _mm_addsub_ps(mul(r1, r2), mul(i1, i2));
}

static inline auto cmul(__m256 a, __m256 b){
    auto r1 = _mm256_permute_ps(a, 0b10100000);
    auto r2 = b;
    auto i1 = _mm256_permute_ps(a, 0b11110101);
    auto i2 = _mm256_permute_ps(b, 0b10110001);
    return _mm256_addsub_ps(mul(r1, r2), mul(i1, i2));
}

static inline auto cmul(__m128d a, __m128d b){
    // [a, b] [c, d] -> [ac-bd, ad+bc]
    // [ac, ad] addsub [bd, bc]
    auto r1 = _mm_permute_pd(a, 0b00);//a a
    auto r2 = b;
    auto i1 = _mm_permute_pd(a, 0b11);
    auto i2 = _mm_permute_pd(b, 0b01);
    _mm_addsub_pd(mul(r1, r2), mul(i1, i2));
}

static inline auto cmul(__m256d a, __m256d b){
    auto r1 = _mm256_permute_pd(a, 0b00);//a a
    auto r2 = b;
    auto i1 = _mm256_permute_pd(a, 0b11);
    auto i2 = _mm256_permute_pd(b, 0b01);
    _mm256_addsub_pd(mul(r1, r2), mul(i1, i2));
}

static inline auto jmul(__m256 a){
    const auto zero = _mm256_setzero_ps();
    const auto perm = _mm256_permute_ps(a, 0b11100001);
    return _mm256_addsub_ps(zero, perm); //[-b, a, ...]
}