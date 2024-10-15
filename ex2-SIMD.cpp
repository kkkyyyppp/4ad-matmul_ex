#include <stdio.h>
#include <immintrin.h>	// visual studio C/C++ -> code generation -> Enable enhanced instruction set : /arch:AVX2

void main(){

    float A = 2;
    float B[8] = {1,2,3,4,5,6,7,8};
    float C[8] = {8,6,4,2,0,-2,-4,-6};

    __m256 a0 = _mm256_set1_ps(A);
    __m256 b0 = _mm256_load_ps(B);
    __m256 c0 = _mm256_load_ps(C);

    c0 = _mm256_fmadd_ps(a0, b0, c0);

    _mm256_store_ps(C, c0);

    printf("C = {\t");    for(int n=0;n<8;++n)    printf("%.1f,\t",C[n]);    puts("}\n");
}
