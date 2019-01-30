#include <iostream>

#include <fftpp/simd.h>
#include <fftpp/scalar.h>

using namespace std;
int main(){
    int N = 128;
    auto x = init_vector<float, 8>(N);
    auto xi = init_vector<float, 8>(N);
    auto y = init_vector<float, 8>(N);
    auto yi = init_vector<float, 8>(N);
    // cout << x.size() << endl;
    float* xp = reinterpret_cast<float*>(&x[0]);
    float* xip = reinterpret_cast<float*>(&xi[0]);
    float* yp = reinterpret_cast<float*>(&y[0]);
    float* yip = reinterpret_cast<float*>(&yi[0]);
    for(int i=0; i<N; i++){
        xp[i] = 0;
        xip[i] = 0;
        yp[i] = 0;
        yip[i] = 0;
    }
    xp[0] = 1;
    fft<false>(N, xp, xip, yp, yip);
    fft<true>(N, yp, yip, xp, xip);
    for(int i=0; i<N; i++){
        cout << xp[i] << ", " << xip[i] << ", " << yp[i] << ", " << yip[i] << endl;
    }
    cout << endl;
}