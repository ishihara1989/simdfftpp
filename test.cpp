#include <iostream>

#include <fftpp/simd.h>
#include <fftpp/fftsimd.h>
#include <fftpp/scalar.h>

using namespace std;
int main(){
    int size = 256;
    using T=float;
    constexpr int N=8;

    auto cx = init_vector<std::complex<T>, N>(size);
    auto cy = init_vector<std::complex<T>, N>(size);
    cx = {};
    cx[0] = 1;
    fftsimd<false>(size, &cx[0], &cy[0]);
    cout << endl;
    for(int i=0;i<size;i++){
        cout << cy[i] << ",";
    }
    cout << endl;
    fftsimd<true>(size, &cy[0], &cx[0]);
    for(int i=0;i<size;i++){
        cout << cx[i] << ",";
    }
    cout << endl;

    for(int i=0;i<32;i++){
        cout << WaveTable<32,false, float>().table[i]<<",";
    }
    cout << WaveTable<32,false, float>().mask << endl;
}