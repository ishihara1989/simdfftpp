SRC=test.cpp
HEADER=fftpp/simd.h fftpp/scalar.h
test: $(SRC) $(HEADER)
	g++ -march=native -std=c++17 -I. $(SRC) -o test.exe
	./test.exe