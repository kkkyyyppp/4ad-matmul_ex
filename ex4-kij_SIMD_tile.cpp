#include <stdio.h>
#include <chrono>
#include <immintrin.h>	// visual studio C/C++ -> code generation -> Enable enhanced instruction set : /arch:AVX2

#define PRT_FN		printf("%-30s", __func__)

int N = 1024, K=N, M=N*1;
float* A = new float[M * K];
float* B = new float[K * N];
float* C = new float[M * N];

void mat_mul_ijk() {	PRT_FN;
	for (int i = 0; i < M; ++i)
		for (int j = 0; j < N; ++j)
			for (int k = 0; k < K; ++k)
				C[i * N + j] += A[i * K + k] * B[k * N + j];
}

void mat_mul_kij() {	PRT_FN;
	for (int k = 0; k < K; ++k)
		for (int i = 0; i < M; ++i) {
			float a = A[i * K + k];
			for (int j = 0; j < N; ++j)
				C[i * N + j] += a * B[k * N + j];
		}
}

int TM = 32, TN = 1024, TK = 1024;

void mat_mul_tile_kij() {	PRT_FN;
	for (int i0 = 0; i0 < M; i0 += TM)
		for (int j0 = 0; j0 < N; j0 += TN)
			for (int k0 = 0; k0 < K; k0 += TK)
				for (int k = k0; k < k0 + TK; ++k)
					for (int i = i0; i < i0 + TM; ++i){
						float a = A[i * K + k];
						for (int j = j0; j < j0 + TN; ++j)
							C[i * N + j] += a * B[k * N + j];
					}
}

void mat_mul_kij_avx() {	PRT_FN;
	for (int k = 0; k < K; ++k)
		for (int i = 0; i < M; ++i) {
			__m256 a0 = _mm256_set1_ps(A[(i + 0) * K + (k + 0)]);
			for (int j = 0; j < N; j+=8) {
				__m256 c0 = _mm256_load_ps(&C[(i + 0) * N + j]);
				__m256 b0 = _mm256_load_ps(&B[(k + 0) * N + j]);
				c0 = _mm256_fmadd_ps(a0, b0, c0);
				_mm256_store_ps(&C[i * N + j], c0);
			}
		}
}

void mat_mul_kij_avx2() {	PRT_FN;
	for (int k = 0; k < K; k+=2)
		for (int i = 0; i < M; ++i) {
			__m256 a0 = _mm256_set1_ps(A[(i + 0) * K + (k + 0)]);
			__m256 a1 = _mm256_set1_ps(A[(i + 0) * K + (k + 1)]);
			for (int j = 0; j < N; j += 8) {
				__m256 c0 = _mm256_load_ps(&C[(i + 0) * N + j]);
				__m256 b0 = _mm256_load_ps(&B[(k + 0) * N + j]);
				__m256 b1 = _mm256_load_ps(&B[(k + 1) * N + j]);
				c0 = _mm256_fmadd_ps(a0, b0, c0);
				c0 = _mm256_fmadd_ps(a1, b1, c0);
				_mm256_store_ps(&C[i * N + j], c0);
			}
		}
}

void mat_mul_kij_avx8() {	PRT_FN;
	for (int k = 0; k < K; k+=8)
		for (int i = 0; i < M; ++i) {
			__m256 a0 = _mm256_set1_ps(A[(i + 0) * K + (k + 0)]);
			__m256 a1 = _mm256_set1_ps(A[(i + 0) * K + (k + 1)]);
			__m256 a2 = _mm256_set1_ps(A[(i + 0) * K + (k + 2)]);
			__m256 a3 = _mm256_set1_ps(A[(i + 0) * K + (k + 3)]);
			__m256 a4 = _mm256_set1_ps(A[(i + 0) * K + (k + 4)]);
			__m256 a5 = _mm256_set1_ps(A[(i + 0) * K + (k + 5)]);
			__m256 a6 = _mm256_set1_ps(A[(i + 0) * K + (k + 6)]);
			__m256 a7 = _mm256_set1_ps(A[(i + 0) * K + (k + 7)]);
			for (int j = 0; j < N; j += 16) {
				__m256 c0 = _mm256_load_ps(&C[(i + 0) * N + j]);

				__m256 b0 = _mm256_load_ps(&B[(k + 0) * N + j]);
				__m256 b1 = _mm256_load_ps(&B[(k + 1) * N + j]);
				__m256 b2 = _mm256_load_ps(&B[(k + 2) * N + j]);
				__m256 b3 = _mm256_load_ps(&B[(k + 3) * N + j]);
				__m256 b4 = _mm256_load_ps(&B[(k + 4) * N + j]);
				__m256 b5 = _mm256_load_ps(&B[(k + 5) * N + j]);
				__m256 b6 = _mm256_load_ps(&B[(k + 6) * N + j]);
				__m256 b7 = _mm256_load_ps(&B[(k + 7) * N + j]);

				c0 = _mm256_fmadd_ps(a0, b0, c0);
				c0 = _mm256_fmadd_ps(a1, b1, c0);
				c0 = _mm256_fmadd_ps(a2, b2, c0);
				c0 = _mm256_fmadd_ps(a3, b3, c0);
				c0 = _mm256_fmadd_ps(a4, b4, c0);
				c0 = _mm256_fmadd_ps(a5, b5, c0);
				c0 = _mm256_fmadd_ps(a6, b6, c0);
				c0 = _mm256_fmadd_ps(a7, b7, c0);

				__m256 c1= _mm256_load_ps(&C[(i + 0) * N + j + 8]);

				__m256 d0 = _mm256_load_ps(&B[(k + 0) * N + j + 8]);
				__m256 d1 = _mm256_load_ps(&B[(k + 1) * N + j + 8]);
				__m256 d2 = _mm256_load_ps(&B[(k + 2) * N + j + 8]);
				__m256 d3 = _mm256_load_ps(&B[(k + 3) * N + j + 8]);
				__m256 d4 = _mm256_load_ps(&B[(k + 4) * N + j + 8]);
				__m256 d5 = _mm256_load_ps(&B[(k + 5) * N + j + 8]);
				__m256 d6 = _mm256_load_ps(&B[(k + 6) * N + j + 8]);
				__m256 d7 = _mm256_load_ps(&B[(k + 7) * N + j + 8]);

				c1 = _mm256_fmadd_ps(a0, d0, c1);
				c1 = _mm256_fmadd_ps(a1, d1, c1);
				c1 = _mm256_fmadd_ps(a2, d2, c1);
				c1 = _mm256_fmadd_ps(a3, d3, c1);
				c1 = _mm256_fmadd_ps(a4, d4, c1);
				c1 = _mm256_fmadd_ps(a5, d5, c1);
				c1 = _mm256_fmadd_ps(a6, d6, c1);
				c1 = _mm256_fmadd_ps(a7, d7, c1);

				_mm256_store_ps(&C[i * N + j], c0);
				_mm256_store_ps(&C[i * N + j+8], c1);
			}
		}
}

void mat_mul_tile_kij_avx() {	PRT_FN;
	for (int i0 = 0; i0 < M; i0 += TM)
		for (int j0 = 0; j0 < N; j0 += TN)
			for (int k0 = 0; k0 < K; k0 += TK)
				for (int k = k0; k < k0 + TK; ++k)
					for (int i = i0; i < i0 + TM; ++i) {
						__m256 a0 = _mm256_set1_ps(A[(i + 0) * K + (k + 0)]);
						for (int j = j0; j < j0 + TN; j += 8) {
							__m256 c0 = _mm256_load_ps(&C[(i + 0) * N + j]);
							__m256 b0 = _mm256_load_ps(&B[(k + 0) * N + j]);
							c0 = _mm256_fmadd_ps(a0, b0, c0);
							_mm256_store_ps(&C[i * N + j], c0);
						}
					}
}

void mat_mul_tile_kij_avx2() {	PRT_FN;
	for (int i0 = 0; i0 < M; i0 += TM)
		for (int j0 = 0; j0 < N; j0 += TN)
			for (int k0 = 0; k0 < K; k0 += TK)
				for (int k = k0; k < k0 + TK; k+=2)
					for (int i = i0; i < i0 + TM; ++i) {
						__m256 a0 = _mm256_set1_ps(A[(i + 0) * K + (k + 0)]);
						__m256 a1 = _mm256_set1_ps(A[(i + 0) * K + (k + 1)]);
						for (int j = j0; j < j0 + TN; j += 8) {
							__m256 c0 = _mm256_load_ps(&C[(i + 0) * N + j]);
							__m256 b0 = _mm256_load_ps(&B[(k + 0) * N + j]);
							__m256 b1 = _mm256_load_ps(&B[(k + 1) * N + j]);
							c0 = _mm256_fmadd_ps(a0, b0, c0);
							c0 = _mm256_fmadd_ps(a1, b1, c0);
							_mm256_store_ps(&C[i * N + j], c0);
						}
					}
}

void mat_mul_tile_kij_avx8() {	PRT_FN;
	for (int i0 = 0; i0 < M; i0 += TM)
		for (int j0 = 0; j0 < N; j0 += TN)
			for (int k0 = 0; k0 < K; k0 += TK)
				for (int k = k0; k < k0 + TK; k+=8)
					for (int i = i0; i < i0 + TM; ++i) {
						__m256 a0 = _mm256_set1_ps(A[(i + 0) * K + (k + 0)]);
						__m256 a1 = _mm256_set1_ps(A[(i + 0) * K + (k + 1)]);
						__m256 a2 = _mm256_set1_ps(A[(i + 0) * K + (k + 2)]);
						__m256 a3 = _mm256_set1_ps(A[(i + 0) * K + (k + 3)]);
						__m256 a4 = _mm256_set1_ps(A[(i + 0) * K + (k + 4)]);
						__m256 a5 = _mm256_set1_ps(A[(i + 0) * K + (k + 5)]);
						__m256 a6 = _mm256_set1_ps(A[(i + 0) * K + (k + 6)]);
						__m256 a7 = _mm256_set1_ps(A[(i + 0) * K + (k + 7)]);
						for (int j = j0; j < j0 + TN; j += 16) {
							__m256 c0 = _mm256_load_ps(&C[(i + 0) * N + j]);

							__m256 b0 = _mm256_load_ps(&B[(k + 0) * N + j]);
							__m256 b1 = _mm256_load_ps(&B[(k + 1) * N + j]);
							__m256 b2 = _mm256_load_ps(&B[(k + 2) * N + j]);
							__m256 b3 = _mm256_load_ps(&B[(k + 3) * N + j]);
							__m256 b4 = _mm256_load_ps(&B[(k + 4) * N + j]);
							__m256 b5 = _mm256_load_ps(&B[(k + 5) * N + j]);
							__m256 b6 = _mm256_load_ps(&B[(k + 6) * N + j]);
							__m256 b7 = _mm256_load_ps(&B[(k + 7) * N + j]);

							c0 = _mm256_fmadd_ps(a0, b0, c0);
							c0 = _mm256_fmadd_ps(a1, b1, c0);
							c0 = _mm256_fmadd_ps(a2, b2, c0);
							c0 = _mm256_fmadd_ps(a3, b3, c0);
							c0 = _mm256_fmadd_ps(a4, b4, c0);
							c0 = _mm256_fmadd_ps(a5, b5, c0);
							c0 = _mm256_fmadd_ps(a6, b6, c0);
							c0 = _mm256_fmadd_ps(a7, b7, c0);

							__m256 c1= _mm256_load_ps(&C[(i + 0) * N + j + 8]);

							__m256 d0 = _mm256_load_ps(&B[(k + 0) * N + j + 8]);
							__m256 d1 = _mm256_load_ps(&B[(k + 1) * N + j + 8]);
							__m256 d2 = _mm256_load_ps(&B[(k + 2) * N + j + 8]);
							__m256 d3 = _mm256_load_ps(&B[(k + 3) * N + j + 8]);
							__m256 d4 = _mm256_load_ps(&B[(k + 4) * N + j + 8]);
							__m256 d5 = _mm256_load_ps(&B[(k + 5) * N + j + 8]);
							__m256 d6 = _mm256_load_ps(&B[(k + 6) * N + j + 8]);
							__m256 d7 = _mm256_load_ps(&B[(k + 7) * N + j + 8]);

							c1 = _mm256_fmadd_ps(a0, d0, c1);
							c1 = _mm256_fmadd_ps(a1, d1, c1);
							c1 = _mm256_fmadd_ps(a2, d2, c1);
							c1 = _mm256_fmadd_ps(a3, d3, c1);
							c1 = _mm256_fmadd_ps(a4, d4, c1);
							c1 = _mm256_fmadd_ps(a5, d5, c1);
							c1 = _mm256_fmadd_ps(a6, d6, c1);
							c1 = _mm256_fmadd_ps(a7, d7, c1);

							_mm256_store_ps(&C[i * N + j], c0);
							_mm256_store_ps(&C[i * N + j+8], c1);
						}
					}
}

void test(void (*func)(), int iter = 1) {
	double time_ms = 1.0e99;

	for (int n = 0; n < iter; n++) {
		auto start = std::chrono::high_resolution_clock::now();
		func();
		auto end = std::chrono::high_resolution_clock::now();

		auto time_ms0 = std::chrono::duration<double, std::milli>(end - start).count();
		if (time_ms > time_ms0) time_ms = time_ms0;
		if (iter > 1) puts("");
	}
	auto tflops = 2.0 * M * N * K / time_ms / 1e6;

	printf("%-30s\t%10.1f ms\t%10.1f GFLOPS\n", "", time_ms, tflops);
}


void main() {

	printf("M = %d, N = %d, K = %d\nCalculating...\n\n", M, N, K);

	test(mat_mul_kij);
	test(mat_mul_tile_kij);
	puts("");
	test(mat_mul_kij_avx);
	test(mat_mul_tile_kij_avx);
	puts("");
	test(mat_mul_kij_avx2);
	test(mat_mul_tile_kij_avx2);
	puts("");
	test(mat_mul_kij_avx8);
	test(mat_mul_tile_kij_avx8);

	//delete[] A, B, C;
}
