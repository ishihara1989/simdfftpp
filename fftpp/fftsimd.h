#define _USE_MATH_DEFINES
#include <cmath>
#include <type_traits>


#include <fftpp/simd.h>

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

template<int s, bool is_exchange_output, bool is_inverse, typename T>
struct fftcore<1, s, is_exchange_output, is_inverse, T>{
    void operator()(std::complex<T>* x, std::complex<T>* y){}
};

template<int s, bool is_exchange_output, bool is_inverse, typename T>
struct fftcore<2, s, is_exchange_output, is_inverse, T>{
    void operator()(std::complex<T>* x, std::complex<T>* y){
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
};

// // TODO s=1, s=4
// template<int n, int s, bool is_exchange_output, bool is_inverse>
// struct fftcore<n, s, is_exchange_output, is_inverse, float>{
//     std::enable_if< (s>=8&&n>2) > operator()(std::complex<float>* x, std::complex<float>* y){
//         const int n0 = 0;
//         const int n1 = n/4;
//         const int n2 = n/2;
//         const int n3 = n1 + n2;
//         const float theta0 = 2*M_PI/n;
//         for (int p = 0; p < n1; p++){
//             __m256 w1p, w2p, w3p;
//             float cs = cos(p*theta0);
//             float sn = sin(p*theta0);
//             if constexpr(is_inverse){
//                 w1p = _mm256_setr_ps(cs, sn, cs, sn, cs, sn, cs, sn);
//             } else{
//                 w1p = _mm256_setr_ps(cs, -sn, cs, -sn, cs, -sn, cs, -sn);
//             }
//             w2p = mul(w1p,w1p);
//             w3p = mul(w1p,w2p);
//             for (int q = 0; q < s; q+=4){
//                 constexpr std::complex<float> j = {0, 1};
//                 const auto a = load<float, 8>(&x[q + s*(p + n0)]);
//                 const auto b = load<float, 8>(&x[q + s*(p + n1)]);
//                 const auto c = load<float, 8>(&x[q + s*(p + n2)]);
//                 const auto d = load<float, 8>(&x[q + s*(p + n3)]);
//                 const auto  apc = add(a, c);
//                 const auto  amc = sub(a, c);
//                 const auto  bpd = add(b, d);
//                 const auto  bmd = sub(b, d); //[a,b,...]
//                 const auto jbmd = jmul(bmd); //[-b, a, ...]
//                 // store(&y[q + s*(4*p)], add(apc, bpd));
//                 // if constexpr (is_inverse){
//                 //     store(&y[q + s*(4*p + 1)], mul(w1p, add(amc, jbmd)));
//                 // } else{
//                 //     store(&y[q + s*(4*p + 1)], mul(w1p, sub(amc, jbmd)));
//                 // }
//                 // store(&y[q + s*(4*p + 2)], sub(apc, bpd));
//                 // if constexpr (is_inverse){
//                 //     store(&y[q + s*(4*p + 3)], mul(w3p, sub(amc, jbmd)));
//                 // } else {
//                 //     store(&y[q + s*(4*p + 3)], mul(w3p, add(amc, jbmd)));
//                 // }
//             }
//         }
//         fftcore<n/4, 4*s, !is_exchange_output, is_inverse, float>()(x, y);
//     }
// };


template<bool is_inverse, typename T = double>
void fftsimd(int N, std::complex<T>* x, std::complex<T>* y)
{
    int log2N = static_cast<int>(log2(N));
    switch(log2N){
        case 1: fftcore<2, 1, false, is_inverse, T>()(x, y);break;
        case 2: fftcore<4, 1, false, is_inverse, T>()(x, y);break;
        case 3: fftcore<8, 1, false, is_inverse, T>()(x, y);break;
        case 4: fftcore<16, 1, false, is_inverse, T>()(x, y);break;
        case 5: fftcore< 1<<5, 1, false, is_inverse, T>()(x, y);break;
        case 6: fftcore< 1<<6, 1, false, is_inverse, T>()(x, y);break;
        case 7: fftcore< 1<<7, 1, false, is_inverse, T>()(x, y);break;
        case 8: fftcore< 1<<8, 1, false, is_inverse, T>()(x, y);break;
        case 9: fftcore< 1<<9, 1, false, is_inverse, T>()(x, y);break;
        case 10: fftcore< 1<<10, 1, false, is_inverse, T>()(x, y);break;
        case 11: fftcore< 1<<11, 1, false, is_inverse, T>()(x, y);break;
        case 12: fftcore< 1<<12, 1, false, is_inverse, T>()(x, y);break;
        case 13: fftcore< 1<<13, 1, false, is_inverse, T>()(x, y);break;
        case 14: fftcore< 1<<14, 1, false, is_inverse, T>()(x, y);break;
        case 15: fftcore< 1<<15, 1, false, is_inverse, T>()(x, y);break;
        case 16: fftcore< 1<<16, 1, false, is_inverse, T>()(x, y);break;
        case 17: fftcore< 1<<17, 1, false, is_inverse, T>()(x, y);break;
        case 18: fftcore< 1<<18, 1, false, is_inverse, T>()(x, y);break;
        case 19: fftcore< 1<<19, 1, false, is_inverse, T>()(x, y);break;
        case 20: fftcore< 1<<20, 1, false, is_inverse, T>()(x, y);break;
        default: break;
    }
    // fft_core<is_inverse, T>(x, y);
    // if((1+log2N)/2%2-1) for (int k = 0; k < N; k++){ y[k] = x[k]; yi[k] = xi[k]; }
    if constexpr (is_inverse) for (int k = 0; k < N; k++){ y[k] /= N;}
}
