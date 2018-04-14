// Machine Problem 3
// Siqi Dai
// 10183100
// 14sd63

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include <math.h>
#include <device_functions.h>
#include <Windows.h>

// generate a random square matrix
__host__ float* rndMatrix (int size) {
	float* m;
	cudaMallocHost((void**)&m, size*size*sizeof(float));
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++)
			m[i*size + j] = ((float)rand()) / (float)RAND_MAX;
	}
	return m;
}

// show the matrix
void showMatrix (float* m, int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) 
			printf("%.3f ",m[i*size + j]);			
		printf("\n");
	}
	printf("\n");
}

// matrix multiplication using CPU
float* matrixMulCPU(float* M, float* N, int size) {
	float *ans = rndMatrix(size);
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			float pVal = 0;
			for (int k = 0; k < size; k++)
				pVal += M[i*size + k] * N[k*size + j];
			ans[i*size + j] = pVal;
		}
	}
	return ans;
}

bool compare(float* A, float* B, int size) {
	for(int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			if (abs(A[i*size+j] - B[i*size+j]) < 0.005)
				return true;
			return false;
}

// matrix multiplication using GPU: each thread produces one output matrix element
__global__ void matrixMulKernel(float* ans, float* M, float* N, int size) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row < size && col < size) {
		float pVal = 0;
		for (int i = 0; i < size; ++i)
			pVal += M[row*size + i] * N[i*size + col];
		ans[row*size + col] = pVal;
	}
}

int main(int argc, char *argv[]) {

	int sizes[5] = {100, 200, 500, 1500, 5000};
	printf("\n=================== Part 1 & 2 ===================\n");
	printf("*** (assume 1 block and 1 thread per block) ***\n");

	for (int i = 0; i < 5; i++) {

		float time_CPU1 = 0; // data transfer time (host to device)
		float time_CPU2 = 0; // data transfer time (device to host)
		float time_CPU = 0; // matrix multiplication time (CPU)
		float time_GPU = 0; // matrix multiplication time (GPU)

		float *M = rndMatrix(sizes[i]);
		float *N = rndMatrix(sizes[i]);
		float *P = rndMatrix(sizes[i]); 
		float *d_M, *d_N, *d_P;
		float *M_back, *N_back;

		// ================= Part 1 ================= 

		// CPU starts timing
		cudaEvent_t start, stop; 
		cudaEventCreate(&start); 
		cudaEventCreate(&stop); 
		cudaEventRecord(start,0); 

		// allocate memory for input and output matrices
		cudaMalloc((void**)&d_M, sizes[i]*sizes[i]*sizeof(float));
		cudaMalloc((void**)&d_N, sizes[i]*sizes[i]*sizeof(float));
		
		// transfer the two input matrices from the host to the device
		cudaMemcpy((void*)d_M, (void*)M, sizes[i]*sizes[i]*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy((void*)d_N, (void*)N, sizes[i]*sizes[i]*sizeof(float), cudaMemcpyHostToDevice);

		// CPU stops timing
		cudaEventRecord(stop,0); 
		cudaEventSynchronize(stop); 
		cudaEventElapsedTime(&time_CPU1, start, stop); 

		cudaMalloc((void**)&d_P, sizes[i]*sizes[i]*sizeof(float));

		// CPU starts timing 
		cudaEvent_t start1, stop1; 
		cudaEventCreate(&start1); 
		cudaEventCreate(&stop1); 
		cudaEventRecord(start1,0); 

		// allocate memory for input and output matrices
		cudaMallocHost((void**)&M_back, sizes[i]*sizes[i]*sizeof(float));
		cudaMallocHost((void**)&N_back, sizes[i]*sizes[i]*sizeof(float));

		// transfer the two input matrices from the device to the host
		cudaMemcpy((void*)d_M, (void*)M_back, sizes[i]*sizes[i]*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy((void*)d_N, (void*)N_back, sizes[i]*sizes[i]*sizeof(float), cudaMemcpyDeviceToHost);

		// CPU stops timing
		cudaEventRecord(stop1,0); 
		cudaEventSynchronize(stop1); 
		cudaEventElapsedTime(&time_CPU2, start1, stop1); 

		printf("\n%d x %d matrix:\n", sizes[i], sizes[i]);
		printf("     Data transfer time (from host to device): %f ms\n", time_CPU1);
		printf("     Data transfer time (from device to host): %f ms\n", time_CPU2);

		// ================= Part 2 ================= 

		// CPU starts timing		 
		cudaEvent_t start2, stop2; 
		cudaEventCreate(&start2); 
		cudaEventCreate(&stop2); 
		cudaEventRecord(start2,0); 

		float* P_CPU = matrixMulCPU(M, N, sizes[i]);

		// CPU stops timing
		cudaEventRecord(stop2,0); 
		cudaEventSynchronize(stop2); 
		cudaEventElapsedTime(&time_CPU, start2, stop2); 

		// GPU starts timing 
		cudaEvent_t t_start, t_stop; 
		cudaEventCreate(&t_start); 
		cudaEventCreate(&t_stop); 
		cudaEventRecord(t_start,0); 

		// launch the kernel	
		// assume a single block and one thread for the block
		matrixMulKernel<<<1, 1>>>(d_P, d_M, d_N, sizes[i]);
		
		// GPU stops timing
		cudaEventRecord(t_stop,0); 
		cudaEventSynchronize(t_stop); 
		cudaEventElapsedTime(&time_GPU, t_start, t_stop); 

		printf("     Matrix multiplication time (CPU): %f ms\n", time_CPU);
		printf("     Matrix multiplication time (GPU): %f ms\n",time_GPU);
		
		// transfer output data to host
		cudaMemcpy((void*)P, (void*)d_P, sizes[i]*sizes[i]*sizeof(float), cudaMemcpyDeviceToHost);
		printf(compare(P_CPU,P,sizes[i])? "     Test PASSED\n" : "     Test FAILED\n");

		//free the memoy
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		cudaEventDestroy(start1);
		cudaEventDestroy(stop1);
		cudaEventDestroy(start2);
		cudaEventDestroy(stop2);
		cudaEventDestroy(t_start);
		cudaEventDestroy(t_stop);
		cudaFreeHost(M);
		cudaFreeHost(N);
		cudaFreeHost(M_back);
		cudaFreeHost(N_back);
		cudaFreeHost(P);
		cudaFree(d_M);
		cudaFree(d_N);
		cudaFree(d_P);
		cudaDeviceReset();
	}

	// ================= Part 3 =================

	printf("\n\n===================== Part 3 =====================\n");
	for (int i = 0; i < 5; i++) {
		printf("\n%d x %d matrix:\n", sizes[i], sizes[i]);

		int width[5] = {2, 4, 8, 16, 32};
		for (int j = 0; j < 5; j++) {
		float time_GPU = 0; // matrix multiplication time (GPU)

		float *M = rndMatrix(sizes[i]);
		float *N = rndMatrix(sizes[i]);
		float *P = rndMatrix(sizes[i]); 
		float *d_M, *d_N, *d_P;

		float* P_CPU = matrixMulCPU(M, N, sizes[i]); // result of matrix addition computed by CPU

		// allocate memory for input and output matrices
		cudaMalloc((void**)&d_M, sizes[i]*sizes[i]*sizeof(float));
		cudaMalloc((void**)&d_N, sizes[i]*sizes[i]*sizeof(float));
		cudaMalloc((void**)&d_P, sizes[i]*sizes[i]*sizeof(float));

		// transfer the two input matrices from the host to the device
		cudaMemcpy((void*)d_M, (void*)M, sizes[i]*sizes[i]*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy((void*)d_N, (void*)N, sizes[i]*sizes[i]*sizeof(float), cudaMemcpyHostToDevice);
		
		int numBlocks = sizes[i] / width[j];
		if (sizes[i] % width[j])
			numBlocks++;	

		// GPU starts timing 
		cudaEvent_t t_start, t_stop; 
		cudaEventCreate(&t_start); 
		cudaEventCreate(&t_stop);
		cudaEventRecord(t_start,0); 

		// launch the kernel	
		dim3 dimGrid(numBlocks, numBlocks, 1);
		dim3 dimBlock(width[j], width[j], 1);

		matrixMulKernel<<<dimGrid, dimBlock>>>(d_P, d_M, d_N, sizes[i]);
		
		// GPU stops timing
		cudaEventRecord(t_stop,0); 
		cudaEventSynchronize(t_stop); 
		cudaEventElapsedTime(&time_GPU, t_start, t_stop); 

		printf("     Block width: %d\n", width[j]);
		printf("     Number of blocks: %d\n", numBlocks);
		printf("     Kernel time: %f ms\n", time_GPU);

		// transfer output data to host
		cudaMemcpy((void*)P, (void*)d_P, sizes[i]*sizes[i]*sizeof(float), cudaMemcpyDeviceToHost);
		printf(compare(P_CPU,P,sizes[i])? "     Test PASSED\n" : "     Test FAILED\n");
		printf("\n");
	
		//free the memoy
		cudaEventDestroy(t_start);
		cudaEventDestroy(t_stop);
		cudaFreeHost(M);
		cudaFreeHost(N);
		cudaFreeHost(P);
		cudaFree(d_M);
		cudaFree(d_N);
		cudaFree(d_P);
		cudaDeviceReset();
		}
	}
}