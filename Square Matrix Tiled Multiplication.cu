// Machine Problem 4
// Siqi Dai
// 10183100
// 14sd63

#ifndef __CUDACC__ 
#define __CUDACC__
#endif

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

// tiled matrix multiplication: tile_width = 2
__global__ void matMulKernel2(float* P, float* M, float* N, int width) {
		__shared__ float Mds[2][2];
		__shared__ float Nds[2][2];
		int bx = blockIdx.x; int by = blockIdx.y;
		int tx = threadIdx.x; int ty = threadIdx.y;
		int row = by*2 + ty; int col = bx*2 + tx;
		float pVal = 0;

		for(int ph = 0; ph < width/2; ++ph) {
			Mds[ty][tx] = M[row*width + ph*2 + tx];
			Nds[ty][tx] = N[(ph*2 + ty)*width + col];
			__syncthreads();
			for(int k = 0; k < 2; ++k) 
				pVal += Mds[ty][k]*Nds[k][tx];
			__syncthreads();
		}
		P[row*width + col] = pVal;
}

// tiled matrix multiplication: tile_width = 4
__global__ void matMulKernel4(float* P, float* M, float* N, int width) {
		__shared__ float Mds4[4][4];
		__shared__ float Nds4[4][4];
		int bx = blockIdx.x; int by = blockIdx.y;
		int tx = threadIdx.x; int ty = threadIdx.y;
		int row = by*4 + ty; int col = bx*4 + tx;
		float pVal = 0;

		for(int ph = 0; ph < width/4; ++ph) {
			Mds4[ty][tx] = M[row*width + ph*4 + tx];
			Nds4[ty][tx] = N[(ph*4 + ty)*width + col];
			__syncthreads();
			for(int k = 0; k < 4; ++k) 
				pVal += Mds4[ty][k]*Nds4[k][tx];
			__syncthreads();
		}
		P[row*width + col] = pVal;
}

// tiled matrix multiplication: tile_width = 10
__global__ void matMulKernel10(float* P, float* M, float* N, int width) {
		__shared__ float Mds10[10][10];
		__shared__ float Nds10[10][10];
		int bx = blockIdx.x; int by = blockIdx.y;
		int tx = threadIdx.x; int ty = threadIdx.y;
		int row = by*10 + ty; int col = bx*10 + tx;
		float pVal = 0;

		for(int ph = 0; ph < width/10; ++ph) {
			Mds10[ty][tx] = M[row*width + ph*10 + tx];
			Nds10[ty][tx] = N[(ph*10 + ty)*width + col];
			__syncthreads();
			for(int k = 0; k < 10; ++k) 
				pVal += Mds10[ty][k]*Nds10[k][tx];
			__syncthreads();
		}
		P[row*width + col] = pVal;
}

// tiled matrix multiplication: tile_width = 20
__global__ void matMulKernel20(float* P, float* M, float* N, int width) {
		__shared__ float Mds20[20][20];
		__shared__ float Nds20[20][20];
		int bx = blockIdx.x; int by = blockIdx.y;
		int tx = threadIdx.x; int ty = threadIdx.y;
		int row = by*20 + ty; int col = bx*20 + tx;
		float pVal = 0;

		for(int ph = 0; ph < width/20; ++ph) {
			Mds20[ty][tx] = M[row*width + ph*20 + tx];
			Nds20[ty][tx] = N[(ph*20 + ty)*width + col];
			__syncthreads();
			for(int k = 0; k < 20; ++k) 
				pVal += Mds20[ty][k]*Nds20[k][tx];
			__syncthreads();
		}
		P[row*width + col] = pVal;
}

// tiled matrix multiplication: tile_width = 25
__global__ void matMulKernel25(float* P, float* M, float* N, int width) {
		__shared__ float Mds25[25][25];
		__shared__ float Nds25[25][25];
		int bx = blockIdx.x; int by = blockIdx.y;
		int tx = threadIdx.x; int ty = threadIdx.y;
		int row = by*25 + ty; int col = bx*25 + tx;
		float pVal = 0;

		for(int ph = 0; ph < width/25; ++ph) {
			Mds25[ty][tx] = M[row*width + ph*25 + tx];
			Nds25[ty][tx] = N[(ph*25 + ty)*width + col];
			__syncthreads();
			for(int k = 0; k < 25; ++k) 
				pVal += Mds25[ty][k]*Nds25[k][tx];
			__syncthreads();
		}
		P[row*width + col] = pVal;
}

int main(int argc, char *argv[]) {

	int sizes[5] = {100, 200, 500, 1500, 5000};
	for (int i = 0; i < 5; i++) {
		
		float time_GPU2 = 0; 
		float time_GPU4 = 0; 
		float time_GPU10 = 0; 
		float time_GPU20 = 0;
		float time_GPU25 = 0;
		float *M = rndMatrix(sizes[i]);
		float *N = rndMatrix(sizes[i]);
		float *P2 = rndMatrix(sizes[i]); 
		float *P4 = rndMatrix(sizes[i]);
		float *P10 = rndMatrix(sizes[i]); 
		float *P20 = rndMatrix(sizes[i]); 
		float *P25 = rndMatrix(sizes[i]); 
		float *d_M, *d_N, *d_P2, *d_P4, *d_P10, *d_P20, *d_P25;

		printf("%d x %d Matrix:\n", sizes[i], sizes[i]);

		// allocate memory for input and output matrices
		cudaMalloc((void**)&d_M, sizes[i]*sizes[i]*sizeof(float));
		cudaMalloc((void**)&d_N, sizes[i]*sizes[i]*sizeof(float));		
		cudaMalloc((void**)&d_P2, sizes[i]*sizes[i]*sizeof(float));
		cudaMalloc((void**)&d_P4, sizes[i]*sizes[i]*sizeof(float));
		cudaMalloc((void**)&d_P10, sizes[i]*sizes[i]*sizeof(float));
		cudaMalloc((void**)&d_P20, sizes[i]*sizes[i]*sizeof(float));
		cudaMalloc((void**)&d_P25, sizes[i]*sizes[i]*sizeof(float));
		
		// transfer the two input matrices from the host to the device
		cudaMemcpy((void*)d_M, (void*)M, sizes[i]*sizes[i]*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy((void*)d_N, (void*)N, sizes[i]*sizes[i]*sizeof(float), cudaMemcpyHostToDevice);

		float* P_CPU = matrixMulCPU(M, N, sizes[i]);

		// ==== block width = 2 ====
		// GPU starts timing 
		cudaEvent_t t_start, t_stop; 
		cudaEventCreate(&t_start); 
		cudaEventCreate(&t_stop); 
		cudaEventRecord(t_start,0); 

		// launch the kernel
		dim3 dimGrid2(sizes[i]/2, sizes[i]/2);
		dim3 dimBlock2(2, 2);
		matMulKernel2<<<dimGrid2, dimBlock2>>>(d_P2, d_M, d_N, sizes[i]);
		
		// GPU stops timing
		cudaEventRecord(t_stop,0); 
		cudaEventSynchronize(t_stop); 
		cudaEventElapsedTime(&time_GPU2, t_start, t_stop); 
		printf("     block width: 2; kernel time: %f ms", time_GPU2);

		// transfer output data to host
		cudaMemcpy((void*)P2, (void*)d_P2, sizes[i]*sizes[i]*sizeof(float), cudaMemcpyDeviceToHost);
		printf(compare(P_CPU,P2,sizes[i])? "     Test PASSED\n" : "     Test FAILED\n");

		// ==== block width = 4 ====
		cudaEventCreate(&t_start); 
		cudaEventCreate(&t_stop); 
		cudaEventRecord(t_start,0); 

		dim3 dimGrid4(sizes[i]/4, sizes[i]/4);
		dim3 dimBlock4(4, 4);
		matMulKernel4<<<dimGrid4, dimBlock4>>>(d_P4, d_M, d_N, sizes[i]);
		
		cudaEventRecord(t_stop,0); 
		cudaEventSynchronize(t_stop); 
		cudaEventElapsedTime(&time_GPU4, t_start, t_stop); 
		printf("     block width: 4; kernel time: %f ms", time_GPU4);

		// transfer output data to host
		cudaMemcpy((void*)P4, (void*)d_P4, sizes[i]*sizes[i]*sizeof(float), cudaMemcpyDeviceToHost);
		printf(compare(P_CPU,P4,sizes[i])? "     Test PASSED\n" : "     Test FAILED\n");

		// ==== block width = 10 ====
		cudaEventCreate(&t_start); 
		cudaEventCreate(&t_stop); 
		cudaEventRecord(t_start,0); 

		dim3 dimGrid10(sizes[i]/10, sizes[i]/10);
		dim3 dimBlock10(10, 10);
		matMulKernel10<<<dimGrid10, dimBlock10>>>(d_P10, d_M, d_N, sizes[i]);
		
		cudaEventRecord(t_stop,0); 
		cudaEventSynchronize(t_stop); 
		cudaEventElapsedTime(&time_GPU10, t_start, t_stop); 
		printf("     block width: 10; kernel time: %f ms", time_GPU10);

		// transfer output data to host
		cudaMemcpy((void*)P10, (void*)d_P10, sizes[i]*sizes[i]*sizeof(float), cudaMemcpyDeviceToHost);
		printf(compare(P_CPU,P10,sizes[i])? "     Test PASSED\n" : "     Test FAILED\n");

		// ==== block width = 20 ====
		cudaEventCreate(&t_start); 
		cudaEventCreate(&t_stop); 
		cudaEventRecord(t_start,0); 

		dim3 dimGrid20(sizes[i]/20, sizes[i]/20);
		dim3 dimBlock20(20, 20);
		matMulKernel20<<<dimGrid20, dimBlock20>>>(d_P20, d_M, d_N, sizes[i]);
		
		cudaEventRecord(t_stop,0); 
		cudaEventSynchronize(t_stop); 
		cudaEventElapsedTime(&time_GPU20, t_start, t_stop); 
		printf("     block width: 20; kernel time: %f ms", time_GPU20);

		// transfer output data to host
		cudaMemcpy((void*)P20, (void*)d_P20, sizes[i]*sizes[i]*sizeof(float), cudaMemcpyDeviceToHost);
		printf(compare(P_CPU,P20,sizes[i])? "     Test PASSED\n" : "     Test FAILED\n");

		// ==== block width = 25 ====
		cudaEventCreate(&t_start); 
		cudaEventCreate(&t_stop); 
		cudaEventRecord(t_start,0); 

		dim3 dimGrid25(sizes[i]/25, sizes[i]/25);
		dim3 dimBlock25(25, 25);
		matMulKernel25<<<dimGrid25, dimBlock25>>>(d_P25, d_M, d_N, sizes[i]);
		
		cudaEventRecord(t_stop,0); 
		cudaEventSynchronize(t_stop); 
		cudaEventElapsedTime(&time_GPU25, t_start, t_stop); 
		printf("     block width: 25; kernel time: %f ms", time_GPU25);

		// transfer output data to host
		cudaMemcpy((void*)P25, (void*)d_P25, sizes[i]*sizes[i]*sizeof(float), cudaMemcpyDeviceToHost);
		printf(compare(P_CPU,P25,sizes[i])? "     Test PASSED\n" : "     Test FAILED\n");

		printf("\n");

		//free the memoy
		cudaEventDestroy(t_start);
		cudaEventDestroy(t_stop);
		cudaFreeHost(M);
		cudaFreeHost(N);
		cudaFreeHost(P2);
		cudaFreeHost(P4);
		cudaFreeHost(P10);
		cudaFreeHost(P20);
		cudaFreeHost(P25);
		cudaFree(d_M);
		cudaFree(d_N);
		cudaFree(d_P2);
		cudaFree(d_P4);
		cudaFree(d_P10);
		cudaFree(d_P20);
		cudaFree(d_P25);
		cudaDeviceReset();
	}

		
}
