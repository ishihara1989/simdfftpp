// reference scalar inplementation for Stockman 4-radix FFT

#define _USE_MATH_DEFINES
#include <cmath>
#include <complex>

template<bool is_inverse, typename T = double>
void fft_core(int n, int s, std::complex<T>* x, std::complex<T>* y)
{
    const int n0 = 0;
    const int n1 = n/4;
    const int n2 = n/2;
    const int n3 = n1 + n2;
    const T theta0 = 2*M_PI/n;

    if (n == 2) {
        for (int q = 0; q < s; q++) {
            const auto a = x[q];
            const auto b = x[q + s];
            y[q] = a + b;
            y[q + s] = a - b;
        }
    }
    else if (n > 2) {
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
        fft_core<is_inverse, T>(n/4, 4*s, y, x);
    }
}

template<bool is_inverse, typename T = double>
void fft(int N, std::complex<T>* x, std::complex<T>* y)
{
    int log2N = static_cast<int>(log2(N));
    fft_core<is_inverse, T>(N, 1, x, y);
    if((1+log2N)/2%2-1) for (int k = 0; k < N; k++){ y[k] = x[k]; }
    if constexpr (is_inverse) for (int k = 0; k < N; k++){ y[k] /= N; }
}

template<bool is_inverse, typename T = double>
void fft_core(int n, int s, T* x, T* xi, T* y, T* yi)
{
    const int n0 = 0;
    const int n1 = n/4;
    const int n2 = n/2;
    const int n3 = n1 + n2;
    const T theta0 = 2*M_PI/n;

    if (n == 2) {
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
    else if (n > 2) {
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
        fft_core<is_inverse, T>(n/4, 4*s, y, yi, x, xi);
    }
}

template<bool is_inverse, typename T = double>
void fft(int N, T* x, T* xi, T* y, T* yi)
{
    int log2N = static_cast<int>(log2(N));
    fft_core<is_inverse, T>(N, 1, x, xi, y, yi);
    if((1+log2N)/2%2-1) for (int k = 0; k < N; k++){ y[k] = x[k]; yi[k] = xi[k]; }
    if constexpr (is_inverse) for (int k = 0; k < N; k++){ y[k] /= N; yi[k] /= N; }
}