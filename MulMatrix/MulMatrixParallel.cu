#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <locale>

#include <stdio.h>

#define BLOCK_SIZE 4

cudaError_t MulMatrixCuda(double* mul_matrix, double* matrix1, double * matrix2, int n);

__global__ void mtxMult(double *C, double *A, double *B, int n)
{
	int bx = blockIdx.x; // == 0
	int by = blockIdx.y; // == 0
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float sum = 0.0;

	int ia = n * BLOCK_SIZE * by + n * ty; // A[i,0] - индекс строки (первого элемента в строке)
	int ib = BLOCK_SIZE * bx + tx;         // B[0,j] - индекс столбца (первого элемента столбца)

	for (int k = 0; k < n; k++) // вычисление элемента
	{
		sum += A[ia + k] * B[ib + k * n];
	}

	int ic = n * BLOCK_SIZE*by + BLOCK_SIZE * bx; // Номер начала столбца в блоке результата
	//C[ic + n * ty + tx] = ib;
	C[ic + n * ty + tx] = sum; // запоминаем разультат
}

int main()
{
	setlocale(LC_ALL, "Russian");

	const int n = 4; // размерность матрицы, кратная BLOCK_SIZE 16

	double matrix1[n*n] = { 0 };
	double matrix2[n*n] = { 0 };
	double mul_matrix[n*n] = { 0 };

	// инициализация матриц
	for (int i = 0; i < n; i++)
	{
		//matrix1[i] = i;
		for (int j = 0; j < n; j++)
		{
			//double v = 10 * i + j;
			matrix1[n *i + j] = i * n + j;
		}
	}

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			matrix2[n *i + j] = i + 10 * j; // (i == j) ? 1 : 0; для единичной матрицы
		}
	}

	// Вывод матрицы на консоль
	/*for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			double m = matrix1[i + n * j];
			printf("[%f] ", m);
		}
		printf("\n");
	}*/

#pragma region Защита для неквадратных матриц

	/*if (n1 != m2)
	{
		printf("Размерности матриц не совпадают");
		return 0;
	}


	// инициализация матриц
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



	cudaError_t cudaStatus = MulMatrixCuda(mul_matrix, matrix1, matrix2, n);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	// Вывод матрицы на консоль
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
	printf("\n");
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			double m = mul_matrix[n *i + j];
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
cudaError_t MulMatrixCuda(double* mul_matrix, double* matrix1, double * matrix2, int n)
{
	int numBytes = n * n * sizeof(double);
	double *dev_matrix1 = 0;
	double *dev_matrix2 = 0;
	double *dev_mul_matrix = 0;
	cudaError_t cudaStatus;


	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

#pragma region Выделение памяти в DRAM
	cudaStatus = cudaMalloc((void**)&dev_mul_matrix, numBytes);
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
#pragma endregion // для трёх матриц: 1-ой, 2-ой и результирующей

#pragma region Копирование данных из CPU в DRAM
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
#pragma endregion // для двух начальных матриц

	dim3 blocks(n / BLOCK_SIZE, n / BLOCK_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	// Запуск ядра
	mtxMult << <blocks, threads >> > (dev_mul_matrix, dev_matrix1, dev_matrix2, n);

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

Error:
	cudaFree(dev_mul_matrix);
	cudaFree(dev_matrix1);
	cudaFree(dev_matrix2);

	return cudaStatus;
}
