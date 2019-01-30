#pragma once

#include <intrin.h>

#include <vector>

#include <boost/align/aligned_allocator.hpp>

template<typename T, int N>
using SimdVector = std::vector<T, boost::alignment::aligned_allocator<T, N>>;

template<typename T, int N>
auto init_vector(int size){
    return  SimdVector<T, N>(size);
}
