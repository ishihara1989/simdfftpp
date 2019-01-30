#include <iostream>

#include <fftpp/simd.h>
#include <fftpp/fftsimd.h>
#include <fftpp/scalar.h>

using namespace std;
int main(){
    int size = 8;
    using T=double;
    constexpr int N=4;
    auto x = init_vector<T, N>(size);
    auto xi = init_vector<T, N>(size);
    auto y = init_vector<T, N>(size);
    auto yi = init_vector<T, N>(size);
    T* xp = &x[0];
    T* xip = &xi[0];
    T* yp = &y[0];
    T* yip = &yi[0];
    for(int i=0; i<size; i++){
        xp[i] = 0;
        xip[i] = 0;
        yp[i] = 0;
        yip[i] = 0;
    }
    xp[0] = 1;
    // fftsimd_core<false, T, N>(size, 1, xp, xip, yp, yip);
    // fftsimd_core<true, T, N>(size, 1, yp, yip, xp, xip);
    fftsimd<false, T, N>(size, xp, xip, yp, yip);
    fftsimd<true, T, N>(size, yp, yip, xp, xip);
    for(int i=0; i<size; i++){
        cout << xp[i] << ", " << xip[i] << ", " << yp[i] << ", " << yip[i] << endl;
    }
    cout << endl;
    // alignas(32) double v[] = {1,2,3,4};
    // auto dv = load<double, 4>(v);
    // auto perm = _mm256_permute4x64_pd(dv, 0b11011000);
    // store(v, perm);
    // for(int i=0;i<4;i++){
    //     cout << v[i] << endl;
    // }
}