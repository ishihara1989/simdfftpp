#pragma once

#include <x86intrin.h>

#include <type_traits>
#include <vector>

#include <boost/align/aligned_allocator.hpp>

template<typename T, int N>
using SimdVector = std::vector<T, boost::alignment::aligned_allocator<T, N>>;

template<typename T, int N>
auto init_vector(int size){
    return  SimdVector<T, N>(size);
}

template<typename T, int N>
inline auto load(const T* src){
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

inline void store(float* dst, __m128 src){
    _mm_store_ps(dst, src);
}

inline void store(float* dst, __m256 src){
    _mm256_store_ps(dst, src);
}

inline void store(double* dst, __m128d src){
    _mm_store_pd(dst, src);
}

inline void store(double* dst, __m256d src){
    _mm256_store_pd(dst, src);
}

inline auto add(__m128 a, __m128 b){
    return _mm_add_ps(a, b);
}

inline auto add(__m256 a, __m256 b){
    return _mm256_add_ps(a, b);
}

inline auto add(__m128d a, __m128d b){
    return _mm_add_pd(a, b);
}

inline auto add(__m256d a, __m256d b){
    return _mm256_add_pd(a, b);
}

inline auto sub(__m128 a, __m128 b){
    return _mm_sub_ps(a, b);
}

inline auto sub(__m256 a, __m256 b){
    return _mm256_sub_ps(a, b);
}

inline auto sub(__m128d a, __m128d b){
    return _mm_sub_pd(a, b);
}

inline auto sub(__m256d a, __m256d b){
    return _mm256_sub_pd(a, b);
}

inline auto mul(__m128 a, __m128 b){
    return _mm_mul_ps(a, b);
}

inline auto mul(__m256 a, __m256 b){
    return _mm256_mul_ps(a, b);
}

inline auto mul(__m128d a, __m128d b){
    return _mm_mul_pd(a, b);
}

inline auto mul(__m256d a, __m256d b){
    return _mm256_mul_pd(a, b);
}

inline auto div(__m128 a, __m128 b){
    return _mm_div_ps(a, b);
}

inline auto div(__m256 a, __m256 b){
    return _mm256_div_ps(a, b);
}

inline auto div(__m128d a, __m128d b){
    return _mm_div_pd(a, b);
}

inline auto div(__m256d a, __m256d b){
    return _mm256_div_pd(a, b);
}