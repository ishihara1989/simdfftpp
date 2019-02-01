#define _USE_MATH_DEFINES
#include <cmath>
#include <type_traits>
#include <iostream>


#include <fftpp/simd.h>

#pragma GCC diagnostic ignored "-Wnarrowing"

template<int n, bool is_inverse, typename T>
struct WaveTable{
    std::complex<T> table[n];
    int mask = -1^(-1<<static_cast<int>(log2(n)));
    constexpr WaveTable(): table() {
        constexpr int s = is_inverse?1:-1;      
        for(int i=0;i<n/2;i++){
            table[2*i]=WaveTable<n/2, is_inverse, T>().table[i];
        }
        for(int i=0;i<n/8;i++){
            auto theta = 2*M_PI*(2*i+1)/n;
            const T cs = cos(theta);
            const T sn = s*sin(theta);
            table[2*i+1]={cs, sn};
            table[2*i+1+n/4]={-sn, cs};
            table[2*i+1+n/2]={-cs, -sn};
            table[2*i+1+3*n/4]={sn, -cs};
        }
    }
};

template<bool is_inverse, typename T>
struct WaveTable<1, is_inverse, T>{
    std::complex<T> table[1];
    int mask = 0;
    constexpr WaveTable(): table(){
        table[0] = 1;
    }
};
template<bool is_inverse, typename T>
struct WaveTable<2, is_inverse, T>{
    std::complex<T> table[2];
    int mask = 1;
    constexpr WaveTable(): table(){
        table[0] = {1, 0};
        table[1] = {-1, 0};
    }
};
template<bool is_inverse, typename T>
struct WaveTable<4, is_inverse, T>{
    std::complex<T> table[4];
    int mask = 3;
    constexpr WaveTable(): table(){
        constexpr int s = is_inverse?1:-1;
        table[0] = {1, 0};
        table[1] = {0, s};
        table[2] = {-1, 0};
        table[3] = {0, -s};
    }
};

// fallback
template<int n, int s, bool is_exchange_output, bool is_inverse, typename T>
struct fftcore{
    void operator()(std::complex<T>* x, std::complex<T>* y){
        const int n0 = 0;
        const int n1 = n/4;
        const int n2 = n/2;
        const int n3 = n1 + n2;
        const T theta0 = 2*M_PI/n;
        for (int p = 0; p < n1; p++){
            std::complex<T> w1p, w2p, w3p;
            if constexpr(is_inverse){
                w1p = {cos(p*theta0), sin(p*theta0)};
            } else{
                w1p = {cos(p*theta0), -sin(p*theta0)};
            }
            w2p = w1p*w1p;
            w3p = w1p*w2p;
            for (int q = 0; q < s; q++){
                constexpr std::complex<T> j = {0, 1};
                const auto a = x[q + s*(p + n0)];
                const auto b = x[q + s*(p + n1)];
                const auto c = x[q + s*(p + n2)];
                const auto d = x[q + s*(p + n3)];
                const auto  apc = a + c;
                const auto  amc = a - c;
                const auto  bpd = b + d;
                const auto jbmd = j * (b-d);
                y[q + s*(4*p)] = apc +  bpd;
                if constexpr (is_inverse){
                    y[q + s*(4*p + 1)] = w1p*(amc + jbmd);
                } else{
                    y[q + s*(4*p + 1)] = w1p*(amc - jbmd);
                }
                y[q + s*(4*p + 2)] = w2p*(apc -  bpd);
                if constexpr (is_inverse){
                    y[q + s*(4*p + 3)] = w3p*(amc - jbmd);
                } else {
                    y[q + s*(4*p + 3)] = w3p*(amc + jbmd);
                }
            }
        }
        fftcore<n/4, 4*s, !is_exchange_output, is_inverse, T>()(y, x);
    }
};

template<int s, bool is_exchange_output, bool is_inverse>
struct fftcore<1, s, is_exchange_output, is_inverse, float>{
    void operator()(std::complex<float>* x, std::complex<float>* y){}
};

template<int s, bool is_exchange_output, bool is_inverse>
struct fftcore<1, s, is_exchange_output, is_inverse, double>{
    void operator()(std::complex<double>* x, std::complex<double>* y){}
};


template<int s, bool is_exchange_output, typename T>
void fftcore2(std::complex<T>* x, std::complex<T>* y){
    for (int q = 0; q < s; q++) {
        std::complex<T>* output;
        if constexpr(is_exchange_output){
            output = x;
        } else {
            output = y;
        }
        const auto a = x[q];
        const auto b = x[q + s];
        output[q] = a + b;
        output[q + s] = a - b;
    }
}

template<int s, bool is_exchange_output, bool is_inverse>
struct fftcore<2, s, is_exchange_output, is_inverse, float>{
    void operator()(std::complex<float>* x, std::complex<float>* y){
        fftcore2<s, is_exchange_output>(x,y);
    }
};

template<int s, bool is_exchange_output, bool is_inverse>
struct fftcore<2, s, is_exchange_output, is_inverse, double>{
    void operator()(std::complex<double>* x, std::complex<double>* y){
        fftcore2<s, is_exchange_output>(x,y);
    }
};

template<int s, bool is_exchange_output, bool is_inverse, typename T>
void fftcore4(std::complex<T>* x, std::complex<T>* y){
    std::complex<T>* output;
    if constexpr(is_exchange_output){
        output = x;
    } else {
        output = y;
    }
    for (int q = 0; q < s; q++){
        constexpr std::complex<T> j = {0, 1};
        const auto a = x[q];
        const auto b = x[q + s];
        const auto c = x[q + s*2];
        const auto d = x[q + s*3];
        const auto  apc = a + c;
        const auto  amc = a - c;
        const auto  bpd = b + d;
        const auto jbmd = j * (b-d);
        output[q] = apc +  bpd;
        if constexpr (is_inverse){
            output[q + s] = (amc + jbmd);
        } else{
            output[q + s] = (amc - jbmd);
        }
        output[q + s*2] = (apc -  bpd);
        if constexpr (is_inverse){
            output[q + s*3] = (amc - jbmd);
        } else {
            output[q + s*3] = (amc + jbmd);
        }
    }
}

template<int s, bool is_exchange_output, bool is_inverse>
struct fftcore<4, s, is_exchange_output, is_inverse, float>{
    void operator()(std::complex<float>* x, std::complex<float>* y){
        fftcore4<s, is_exchange_output, is_inverse>(x,y);
    }
};

template<int s, bool is_exchange_output, bool is_inverse>
struct fftcore<4, s, is_exchange_output, is_inverse, double>{
    void operator()(std::complex<double>* x, std::complex<double>* y){
        fftcore4<s, is_exchange_output, is_inverse>(x,y);
    }
};

// TODO s=1, s=4
template<int n, int s, bool is_exchange_output, bool is_inverse>
struct fftcore<n, s, is_exchange_output, is_inverse, float>{
    void operator()(std::complex<float>* x, std::complex<float>* y){
        // constexpr static auto table = WaveTable<n, is_inverse, float>();
        constexpr int n0 = 0;
        constexpr int n1 = n/4;
        constexpr int n2 = n/2;
        constexpr int n3 = n1 + n2;
        constexpr float theta0 = 2*M_PI/n;
        using T=float;
        if constexpr(s>=8){
            for (int p = 0; p < n1; p++){
                __m256 w1p, w2p, w3p;
                // auto w = table.table[p];
                float cs = cos(theta0*p);
                float sn = sin(theta0*p);
                if constexpr(is_inverse){
                    w1p = _mm256_setr_ps(cs, sn, cs, sn, cs, sn, cs, sn);
                } else{
                    w1p = _mm256_setr_ps(cs, -sn, cs, -sn, cs, -sn, cs, -sn);
                }
                w2p = mul(w1p, w1p);
                w3p = mul(w1p, w2p);
                // auto w2 = table.table[(2*p)&table.mask];
                // cs = w2.real();
                // sn = w2.imag();
                // if constexpr(is_inverse){
                //     w2p = _mm256_setr_ps(cs, sn, cs, sn, cs, sn, cs, sn);
                // } else{
                //     w2p = _mm256_setr_ps(cs, -sn, cs, -sn, cs, -sn, cs, -sn);
                // }
                // auto w3 = table.table[(3*p)&table.mask];
                // cs = w3.real();
                // sn = w3.imag();
                // if constexpr(is_inverse){
                //     w3p = _mm256_setr_ps(cs, sn, cs, sn, cs, sn, cs, sn);
                // } else{
                //     w3p = _mm256_setr_ps(cs, -sn, cs, -sn, cs, -sn, cs, -sn);
                // }
                for (int q = 0; q < s; q+=4){
                    const auto a = load<float, 8>(&x[q + s*(p + n0)]);
                    const auto b = load<float, 8>(&x[q + s*(p + n1)]);
                    const auto c = load<float, 8>(&x[q + s*(p + n2)]);
                    const auto d = load<float, 8>(&x[q + s*(p + n3)]);
                    const auto  apc = add(a, c);
                    const auto  amc = sub(a, c);
                    const auto  bpd = add(b, d);
                    const auto  bmd = sub(b, d); //[a,b,...]
                    const auto jbmd = jmul(bmd); //[-b, a, ...]
                    store(&y[q + s*(4*p)], add(apc, bpd));
                    if constexpr (is_inverse){
                        store(&y[q + s*(4*p + 1)], mul(w1p, add(amc, jbmd)));
                    } else{
                        store(&y[q + s*(4*p + 1)], mul(w1p, sub(amc, jbmd)));
                    }
                    store(&y[q + s*(4*p + 2)], sub(apc, bpd));
                    if constexpr (is_inverse){
                        store(&y[q + s*(4*p + 3)], mul(w3p, sub(amc, jbmd)));
                    } else {
                        store(&y[q + s*(4*p + 3)], mul(w3p, add(amc, jbmd)));
                    }
                }
            }
        } else {
            for (int p = 0; p < n1; p++){
                std::complex<T> w1p, w2p, w3p;
                if constexpr(is_inverse){
                    w1p = {cos(p*theta0), sin(p*theta0)};
                } else{
                    w1p = {cos(p*theta0), -sin(p*theta0)};
                }
                w2p = w1p*w1p;
                w3p = w1p*w2p;
                for (int q = 0; q < s; q++){
                    constexpr std::complex<T> j = {0, 1};
                    const auto a = x[q + s*(p + n0)];
                    const auto b = x[q + s*(p + n1)];
                    const auto c = x[q + s*(p + n2)];
                    const auto d = x[q + s*(p + n3)];
                    const auto  apc = a + c;
                    const auto  amc = a - c;
                    const auto  bpd = b + d;
                    const auto jbmd = j * (b-d);
                    y[q + s*(4*p)] = apc +  bpd;
                    if constexpr (is_inverse){
                        y[q + s*(4*p + 1)] = w1p*(amc + jbmd);
                    } else{
                        y[q + s*(4*p + 1)] = w1p*(amc - jbmd);
                    }
                    y[q + s*(4*p + 2)] = w2p*(apc -  bpd);
                    if constexpr (is_inverse){
                        y[q + s*(4*p + 3)] = w3p*(amc - jbmd);
                    } else {
                        y[q + s*(4*p + 3)] = w3p*(amc + jbmd);
                    }
                }
            }
        } 
        fftcore<n/4, 4*s, !is_exchange_output, is_inverse, float>()(y, x);
    }
};


template<bool is_inverse, typename T = double>
void fftsimd(int N, std::complex<T>* x, std::complex<T>* y)
{
    int log2N = static_cast<int>(log2(N));
    switch(log2N){
        case 0: fftcore<1, 1, false, is_inverse, T>()(x, y);break;
        case 1: fftcore<2, 1, false, is_inverse, T>()(x, y);break;
        case 2: fftcore<4, 1, false, is_inverse, T>()(x, y);break;
        case 3: fftcore<8, 1, false, is_inverse, T>()(x, y);break;
        case 4: fftcore<16, 1, false, is_inverse, T>()(x, y);break;
        case 5: fftcore< 1<<5, 1, false, is_inverse, T>()(x, y);break;
        case 6: fftcore< 1<<6, 1, false, is_inverse, T>()(x, y);break;
        case 7: fftcore< 1<<7, 1, false, is_inverse, T>()(x, y);break;
        case 8: fftcore< 1<<8, 1, false, is_inverse, T>()(x, y);break;
        // case 9: fftcore< 1<<9, 1, false, is_inverse, T>()(x, y);break;
        // case 10: fftcore< 1<<10, 1, false, is_inverse, T>()(x, y);break;
        // case 11: fftcore< 1<<11, 1, false, is_inverse, T>()(x, y);break;
        // case 12: fftcore< 1<<12, 1, false, is_inverse, T>()(x, y);break;
        // case 13: fftcore< 1<<13, 1, false, is_inverse, T>()(x, y);break;
        // case 14: fftcore< 1<<14, 1, false, is_inverse, T>()(x, y);break;
        // case 15: fftcore< 1<<15, 1, false, is_inverse, T>()(x, y);break;
        // case 16: fftcore< 1<<16, 1, false, is_inverse, T>()(x, y);break;
        // case 17: fftcore< 1<<17, 1, false, is_inverse, T>()(x, y);break;
        // case 18: fftcore< 1<<18, 1, false, is_inverse, T>()(x, y);break;
        // case 19: fftcore< 1<<19, 1, false, is_inverse, T>()(x, y);break;
        // case 20: fftcore< 1<<20, 1, false, is_inverse, T>()(x, y);break;
        default: break;
    }
    // fft_core<is_inverse, T>(x, y);
    // if((1+log2N)/2%2-1) for (int k = 0; k < N; k++){ y[k] = x[k]; yi[k] = xi[k]; }
    if constexpr (is_inverse) for (int k = 0; k < N; k++){ y[k] /= N;}
}
