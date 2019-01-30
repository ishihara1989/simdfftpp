#define _USE_MATH_DEFINES
#include <cmath>
#include <type_traits>


#include <fftpp/simd.h>

template<bool is_inverse, typename T, int N>
void fftsimd_core(int n, int s, T* x, T* xi, T* y, T* yi){
    const int n0 = 0;
    const int n1 = n/4;
    const int n2 = n/2;
    const int n3 = n1 + n2;
    const T theta0 = 2*M_PI/n;
    if(n==2){
        if constexpr(std::is_same<double, T>::value) {
            if(s==1){
                const auto a = load<double, 2>(x);
                const auto ai = load<double, 2>(xi);
                const auto hadd = _mm_hadd_pd(a, a); // x[0]+x[1], x[0]+x[1]
                const auto haddi = _mm_hadd_pd(ai, ai); // xi[0]+xi[1], xi[0]+xi[1]
                const auto hsub = _mm_hsub_pd(a, a); // x[0]-x[1], x[0]-x[1]
                const auto hsubi = _mm_hsub_pd(ai, ai); // xi[0]-xi[1], xi[0]-xi[1]
                const auto pack = _mm_unpackhi_pd(hadd, hsub); // x[0]+x[1], x[0]-x[1]
                const auto packi = _mm_unpackhi_pd(haddi, hsubi); // xi[0]+xi[1], xi[0]-xi[1]
                store(y, pack);
                store(yi, packi);
                return;
            }
            if constexpr (N==4){
                if(s==2) {
                    const auto a = load<double, 4>(x);
                    const auto ai = load<double, 4>(xi);
                    const auto perm1 = _mm256_permute4x64_pd(a, 0b01000100); // x[0], x[1], x[0], x[1]
                    const auto perm1i = _mm256_permute4x64_pd(ai, 0b01000100); 
                    const auto perm2 = _mm256_permute4x64_pd(a, 0b10111011); // x[2], x[3], x[2], x[3] 
                    const auto perm2i = _mm256_permute4x64_pd(ai, 0b10111011);
                    const auto hadd = add(perm1, perm2); // x[0]+x[2],x[1]+x[3], x[0]+x[2],x[1]+x[3]
                    const auto haddi = add(perm1i, perm2i);
                    const auto hsub = sub(perm1, perm2); // x[0]-x[2], x[1]-x[3], x[0]-x[2], x[1]-x[3]
                    const auto hsubi = sub(perm1i, perm2i);
                    const auto pack = _mm256_blend_pd(hadd, hsub, 0b1100);// x[0]+x[2], x[1]+x[3], x[0]-x[2], x[1]-x[3]
                    const auto packi = _mm256_blend_pd(haddi, hsubi, 0b1100);
                    store(y, pack);
                    store(yi, packi);
                    return;
                }
            }
        } 

        if constexpr(std::is_same<float, T>::value){
            // TODO
            // fallback
            for (int q = 0; q < s; q++) {
                const T a = x[q];
                const T ai = xi[q];
                const T b = x[q + s];
                const T bi = xi[q + s];
                y[q] = a + b;
                yi[q] = ai + bi;
                y[q + s] = a - b;
                yi[q + s] = ai - bi;
            }
        }

        // any s>=N
        for(int q = 0; q < s; q+=N) {
            const auto a = load<T, N>(&x[q]);
            const auto ai = load<T, N>(&xi[q]);
            const auto b = load<T, N>(&x[q+s]);
            const auto bi = load<T, N>(&xi[q+s]);
            store(&y[q], add(a, b));
            store(&yi[q], add(ai, bi));
            store(&y[q+s], sub(a, b));
            store(&yi[q+s], sub(ai, bi));
        }
        return;
    } else if(n>=4) {
        // n >= 4
        if constexpr(std::is_same<double, T>::value){
            if constexpr(N==2){
                if(s==1){
                    // TODO
                    if(n==4){
                        // const auto ab = 
                    }
                    //TODO else
                    else for (int p = 0; p < n1; p+=2){
                        const auto w1p = _mm_setr_pd(cos(p*theta0), cos((p+1)*theta0));
                        __m128d w1pi;
                        if constexpr(is_inverse){
                            w1pi = _mm_setr_pd(sin(p*theta0), sin((p+1)*theta0));
                        } else {
                            w1pi = _mm_setr_pd(-sin(p*theta0), -sin((p+1)*theta0));
                        }
                        const auto w2p = sub(mul(w1p, w1p), mul(w1pi, w1pi));
                        auto _tmp = mul(w1p, w1pi);
                        const auto w2pi = add(_tmp, _tmp);
                        const auto w3p = sub(mul(w1p, w2p), mul(w1pi, w2pi));
                        const auto w3pi = add(mul(w1p, w2pi), mul(w1pi, w2p));
                        // const auto a = 
                    }
                }
            }
        }

        //fallback
        for (int p = 0; p < n1; p++){
            const T w1p = cos(p*theta0);
            const T w1pi = is_inverse?sin(p*theta0):-sin(p*theta0);
            const T w2p = w1p*w1p - w1pi*w1pi;
            const T w2pi = 2*w1p*w1pi;
            const T w3p = w1p*w2p - w1pi*w2pi;
            const T w3pi = w1p*w2pi + w1pi*w2p;
            for (int q = 0; q < s; q++){
                const T a = x[q + s*(p + n0)];
                const T b = x[q + s*(p + n1)];
                const T c = x[q + s*(p + n2)];
                const T d = x[q + s*(p + n3)];
                const T ai = xi[q + s*(p + n0)];
                const T bi = xi[q + s*(p + n1)];
                const T ci = xi[q + s*(p + n2)];
                const T di = xi[q + s*(p + n3)];
                const T  apc = a + c;
                const T  amc = a - c;
                const T  bpd = b + d;
                const T jbmd = di-bi;
                const T  apci = ai + ci;
                const T  amci = ai - ci;
                const T  bpdi = bi + di;
                const T jbmdi = b - d;
                y[q + s*(4*p)] = apc +  bpd;
                yi[q + s*(4*p)] = apci +  bpdi;
                if constexpr (is_inverse){
                    y[q + s*(4*p + 1)] = w1p*(amc + jbmd) - w1pi*(amci + jbmdi);
                    yi[q + s*(4*p + 1)] = w1p*(amci + jbmdi) + w1pi*(amc + jbmd);
                } else{
                    y[q + s*(4*p + 1)] = w1p*(amc - jbmd) - w1pi*(amci - jbmdi);
                    yi[q + s*(4*p + 1)] = w1p*(amci - jbmdi) + w1pi*(amc - jbmd);
                }
                y[q + s*(4*p + 2)] = w2p*(apc -  bpd) - w2pi*(apci -  bpdi);
                yi[q + s*(4*p + 2)] = w2p*(apci -  bpdi) + w2pi*(apc -  bpd);
                if constexpr (is_inverse){
                    y[q + s*(4*p + 3)] = w3p*(amc - jbmd) - w3pi*(amci - jbmdi);
                    yi[q + s*(4*p + 3)] = w3p*(amci - jbmdi) + w3pi*(amc - jbmd);
                } else {
                    y[q + s*(4*p + 3)] = w3p*(amc + jbmd) - w3pi*(amci + jbmdi);
                    yi[q + s*(4*p + 3)] = w3p*(amci + jbmdi) + w3pi*(amc + jbmd);
                }
            }
        }
        fftsimd_core<is_inverse, T, N>(n/4, 4*s, y, yi, x, xi);
    }
}

template<bool is_inverse, typename T, int N>
void fftsimd(int size, T* x, T* xi, T* y, T* yi)
{
    int log2N = static_cast<int>(log2(size));
    fftsimd_core<is_inverse, T, N>(size, 1, x, xi, y, yi);
    if((1+log2N)/2%2-1) for (int k = 0; k < size; k++){ y[k] = x[k]; yi[k] = xi[k]; }
    if constexpr (is_inverse) for (int k = 0; k < size; k++){ y[k] /= size; yi[k] /= size; }
}