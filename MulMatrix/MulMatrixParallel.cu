#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <locale>

#include <stdio.h>

#define BLOCK_SIZE 100

cudaError_t MulMatrixCuda(double* mul_matrix, double* mul_matrix2, double* matrix1, double * matrix2, int n);

__global__ void mtxMult(double *C, double *A, double *B, int n)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float sum = 0.0;

	int ia = n * BLOCK_SIZE * by + n * ty; // A[i,0] - ������ ������ (������� �������� � ������)
	int ib = BLOCK_SIZE * bx + tx;         // B[0,j] - ������ ������� (������� �������� �������)

	for (int k = 0; k < n; k++) // ���������� ��������
	{
		sum += A[ia + k] * B[ib + k * n];
	}

	int ic = n * BLOCK_SIZE*by + BLOCK_SIZE * bx; // ����� ������ ������� � ����� ����������
	C[ic + n * ty + tx] = sum; // ���������� ���������
}

__global__ void mtxMult2(double *C, double *A, double *B, int n)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int aBegin = n * BLOCK_SIZE * by;
	int aEnd = aBegin + n - 1;
	int bBegin = BLOCK_SIZE * bx;
	int aStep = BLOCK_SIZE;
	int bStep = BLOCK_SIZE * n;

	double sum = 0.0;

	for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep)
	{
		__shared__ double as[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ double bs[BLOCK_SIZE][BLOCK_SIZE];

		as[ty][tx] = A[ia + n * ty + tx];
		bs[ty][tx] = B[ib + n * ty + tx];
		__syncthreads();    // ������ ���������������� (���������� ��������� ���������)

		for (int k = 0; k < BLOCK_SIZE; k++)
			sum += as[ty][k] * bs[k][tx];

		__syncthreads(); // ���������� ������ �� �����
	}

	C[n * BLOCK_SIZE * by + BLOCK_SIZE * bx + n * ty + tx] = sum;
}

int main()
{
	setlocale(LC_ALL, "Russian");

	const int k = 1;
	const int n = k * BLOCK_SIZE; // ����������� �������, ������� BLOCK_SIZE

	double matrix1[n*n] = { 0 };
	double matrix2[n*n] = { 0 };
	double mul_matrix[n*n] = { 0 };
	double mul_matrix2[n*n] = { 0 };

	// ������������� ������
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			matrix1[n *i + j] = i * 10 + j;
		}
	}

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			matrix2[n *i + j] = i + 10 * j; // (i == j) ? 1 : 0; ��� ��������� �������
		}
	}


#pragma region ������ ��� ������������ ������

	/*if (n1 != m2)
	{
		printf("����������� ������ �� ���������");
		return 0;
	}


	// ������������� ������
		for (int i = 0; i < m1; i++)
	{
		for (int j = 0; j < n1; j++)
		{
			matrix1[i][j] = 10 * i + j;
		}
	}
	for (int i = 0; i < m1; i++)
	{
		for (int j = 0; j < n1; j++)
		{
			matrix2[i][j] = i + 10 * j;
		}
	}*/
#pragma endregion

	// ����� ������� �� �������
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			double m = matrix1[n * i + j];
			printf("[%g] ", m);
		}
		printf("\n");
	}
	printf("\n");
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			double m = matrix2[n *i + j];
			printf("[%g] ", m);
		}
		printf("\n");
	}

	printf("\n\n");

	cudaError_t cudaStatus = MulMatrixCuda(mul_matrix, mul_matrix2, matrix1, matrix2, n);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	//// ����� ������� �� �������
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			double m = mul_matrix[n *i + j];
			printf("[%g]", m);
		}
		printf("\n");
	}
	printf("\n");
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			double m = mul_matrix2[n *i + j];
			printf("[%g]", m);
		}
		printf("\n");
	}



	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t MulMatrixCuda(double* mul_matrix, double* mul_matrix2, double* matrix1, double * matrix2, int n)
{
	int numBytes = n * n * sizeof(double);
	double *dev_matrix1 = 0;
	double *dev_matrix2 = 0;
	double *dev_mul_matrix = 0;
	double *dev_mul_matrix2 = 0;
	cudaError_t cudaStatus;


	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

#pragma region ��������� ������ � DRAM
	cudaStatus = cudaMalloc((void**)&dev_mul_matrix, numBytes);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_mul_matrix2, numBytes);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_matrix1, numBytes);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_matrix2, numBytes);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
#pragma endregion // ��� ��� ������: 1-��, 2-�� � ��������������

#pragma region ����������� ������ �� CPU � DRAM
	cudaStatus = cudaMemcpy(dev_matrix1, matrix1, numBytes, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_matrix2, matrix2, numBytes, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
#pragma endregion // ��� ���� ��������� ������

	dim3 blocks(n / BLOCK_SIZE, n / BLOCK_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	//������ GPU
	cudaEvent_t start, stop; // ���������� ����������
	float elapsedTimeInMs = 0;
	cudaEventCreate(&start); // �������������
	cudaEventCreate(&stop);  // �������������
	cudaEventRecord(start, 0); // ������ �������

// ������ ����
	mtxMult << <blocks, threads >> > (dev_mul_matrix, dev_matrix1, dev_matrix2, n);

	cudaEventRecord(stop, 0); // ��������� �������
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTimeInMs, start, stop);
	printf("����������� ����� 1-�� ������ GPU: %.8f ��\n\n", elapsedTimeInMs);


	cudaEventCreate(&start); // �������������
	cudaEventCreate(&stop);  // �������������
	cudaEventRecord(start, 0); // ������ �������

	mtxMult2 << <blocks, threads >> > (dev_mul_matrix2, dev_matrix1, dev_matrix2, n);

	cudaEventRecord(stop, 0); // ��������� �������
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTimeInMs, start, stop);
	printf("����������� ����� 2-�� ������ GPU: %.8f ��\n\n", elapsedTimeInMs);



	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(mul_matrix, dev_mul_matrix, numBytes, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(mul_matrix2, dev_mul_matrix2, numBytes, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_mul_matrix);
	cudaFree(dev_mul_matrix2);
	cudaFree(dev_matrix1);
	cudaFree(dev_matrix2);

	return cudaStatus;
}
