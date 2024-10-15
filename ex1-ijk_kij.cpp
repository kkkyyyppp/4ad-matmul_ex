#include <stdio.h>
#include <chrono>

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

	test(mat_mul_ijk);
	test(mat_mul_kij);
	test(mat_mul_tile_kij);

	//delete[] A, B, C;
}
