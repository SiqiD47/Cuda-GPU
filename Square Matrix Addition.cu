// Machine Problem 2
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

// compute matrix addition using the CPU
float* matrixAddCPU(float* M, float* N, int size) {
	float *ans = rndMatrix(size);
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) 
			ans[i*size+j] = M[i*size+j] + N[i*size+j];
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

// case1: each thread produces one output matrix element
__global__ void matrixAddKernel1(float* ans, float* M, float* N, int size) {
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	if((row < size) && (col < size)) {
		ans[row*size + col] = M[row*size + col] + N[row*size + col];
	}	
}

// case 2: each thread produces one output matrix row
__global__ void matrixAddKernel2(float* ans, float* M, float* N, int size) {
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	if(row < size) { 
		for(int i = 0; i < size; ++i)
			ans[row*size + i] = M[row*size + i] + N[row*size + i];
	}	
}

// case 3: each thread produces one output matrix column
__global__ void matrixAddKernel3(float* ans, float* M, float* N, int size) {
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	if(col < size) {
		for(int i = 0; i < size; ++i)
			ans[i*size + col] = M[i*size + col] + N[i*size + col];
	}
}

int main(int argc, char *argv[]) {

	int sizes[5] = {100, 200, 500, 1500, 5000};
	for (int i = 0; i < 5; i++) {

		printf("******************* Matrix size: %d x %d *******************\n", sizes[i], sizes[i]);

		float time_CPU = 0;
		float time_GPU1 = 0; 
		float time_GPU2 = 0; 
		float time_GPU3 = 0; 

		float *M = rndMatrix(sizes[i]);
		float *N = rndMatrix(sizes[i]);
		float *S1 = rndMatrix(sizes[i]); // for case 1
		float *S2 = rndMatrix(sizes[i]); // for case 2
		float *S3 = rndMatrix(sizes[i]); // for case 3
		float *d_M, *d_N, *d_S1, *d_S2, *d_S3;

		// CPU starts timing
		cudaEvent_t start, stop; 
		cudaEventCreate(&start); 
		cudaEventCreate(&stop); 
		cudaEventRecord(start,0); 

		float* S_CPU = matrixAddCPU(M, N, sizes[i]); // result of matrix addition computed by CPU

		// CPU stops timing
		cudaEventRecord(stop,0); 
		cudaEventSynchronize(stop); 
		cudaEventElapsedTime(&time_CPU, start, stop); 

		// allocate memory for input and output matrices
		cudaMalloc((void**)&d_M, sizes[i]*sizes[i]*sizeof(float));
		cudaMalloc((void**)&d_N, sizes[i]*sizes[i]*sizeof(float));
		cudaMalloc((void**)&d_S1, sizes[i]*sizes[i]*sizeof(float));
		cudaMalloc((void**)&d_S2, sizes[i]*sizes[i]*sizeof(float)); 
		cudaMalloc((void**)&d_S3, sizes[i]*sizes[i]*sizeof(float)); 
	
		// transfer input data to the device
		cudaMemcpy((void*)d_M, (void*)M, sizes[i]*sizes[i]*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy((void*)d_N, (void*)N, sizes[i]*sizes[i]*sizeof(float), cudaMemcpyHostToDevice);
		
		int numBlocks = sizes[i] / 16;
		if (sizes[i] % 16)
			numBlocks++;	

		// === CASE 1: ===
		// GPU starts timing 
		cudaEvent_t t_start, t_stop; 
		cudaEventCreate(&t_start); 
		cudaEventCreate(&t_stop); 
		cudaEventRecord(t_start,0); 

		// launch the kernel	
		dim3 dimGrid1(numBlocks, numBlocks, 1);
		dim3 dimBlock1(16, 16, 1);
		matrixAddKernel1<<<dimGrid1, dimBlock1>>>(d_S1, d_M, d_N, sizes[i]);
		
		// GPU stops timing
		cudaEventRecord(t_stop,0); 
		cudaEventSynchronize(t_stop); 
		cudaEventElapsedTime(&time_GPU1, t_start, t_stop); 

		printf("1) Each thread produces one output matrix element: \n");
		printf("     CPU processing time: %f ms\n", time_CPU);
		printf("     GPU processing time: %f ms\n",time_GPU1);
		
		// transfer output data to host
		cudaMemcpy((void*)S1, (void*)d_S1, sizes[i]*sizes[i]*sizeof(float), cudaMemcpyDeviceToHost);
		printf(compare(S_CPU,S1,sizes[i])? "     Test PASSED\n" : "     Test FAILED\n");
	
		// === CASE 2: ===
		// GPU starts timing 
		cudaEventCreate(&t_start); 
		cudaEventCreate(&t_stop); 
		cudaEventRecord(t_start,0); 

		// launch the kernel	
		dim3 dimGrid2(numBlocks, 1, 1);
		dim3 dimBlock2(16, 1, 1);
		matrixAddKernel2<<<dimGrid2, dimBlock2>>>(d_S2, d_M, d_N, sizes[i]);
		
		// GPU stops timing
		cudaEventRecord(t_stop,0); 
		cudaEventSynchronize(t_stop); 
		cudaEventElapsedTime(&time_GPU2, t_start, t_stop); 
		
		printf("2) Each thread produces one output matrix row: \n");
		printf("     CPU processing time: %f ms\n", time_CPU);
		printf("     GPU processing time: %f ms\n",time_GPU2);

		cudaMemcpy((void*)S2, (void*)d_S2, sizes[i]*sizes[i]*sizeof(float), cudaMemcpyDeviceToHost);
		printf(compare(S_CPU,S2,sizes[i])? "     Test PASSED\n" : "     Test FAILED\n");

		// === CASE 3: ===
		// GPU starts timing 
		cudaEventCreate(&t_start); 
		cudaEventCreate(&t_stop); 
		cudaEventRecord(t_start,0); 

		// launch the kernel	
		dim3 dimGrid3(1, numBlocks, 1);
		dim3 dimBlock3(16, 1, 1);
		matrixAddKernel3<<<dimGrid3, dimBlock3>>>(d_S3, d_M, d_N, sizes[i]);
		
		// GPU stops timing
		cudaEventRecord(t_stop,0); 
		cudaEventSynchronize(t_stop); 
		cudaEventElapsedTime(&time_GPU3, t_start, t_stop); 

		printf("3) Each thread produces one output matrix column: \n");
		printf("     CPU processing time: %f ms\n", time_CPU);
		printf("     GPU processing time: %f ms\n",time_GPU3);

		cudaMemcpy((void*)S3, (void*)d_S3, sizes[i]*sizes[i]*sizeof(float), cudaMemcpyDeviceToHost);
		printf(compare(S_CPU,S3,sizes[i])? "     Test PASSED\n" : "     Test FAILED\n");

		//free the memoy
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		cudaEventDestroy(t_start);
		cudaEventDestroy(t_stop);
		cudaFreeHost(M);
		cudaFreeHost(N);
		cudaFreeHost(S1);
		cudaFreeHost(S2);
		cudaFreeHost(S3);
		cudaFree(d_M);
		cudaFree(d_N);
		cudaFree(d_S1);
		cudaFree(d_S2);
		cudaFree(d_S3);
		cudaDeviceReset();
	}
}
