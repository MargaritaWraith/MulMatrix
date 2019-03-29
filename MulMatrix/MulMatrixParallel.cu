#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <locale>

#include <stdio.h>

#define BLOCK_SIZE 16

cudaError_t MulMatrixCuda(float* mul_matrix, float* mul_matrix2, float* matrix1, float * matrix2, int n);
void print_matrix(float* mtx, int n);

__global__ void mtxMult(float *C, float *A, float *B, int n)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
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
	C[ic + n * ty + tx] = sum; // запоминаем разультат
}

__global__ void mtxMult2(float *C, float *A, float *B, int n)
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

	float sum = 0.0f;

	for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep)
	{
		__shared__ float as[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float bs[BLOCK_SIZE][BLOCK_SIZE];

		as[ty][tx] = A[ia + n * ty + tx];
		bs[ty][tx] = B[ib + n * ty + tx];
		__syncthreads();    // должно синхронизировать (подматрицы полностью загружены)

		for (int k = 0; k < BLOCK_SIZE; k++)
			sum += as[ty][k] * bs[k][tx];

		__syncthreads(); // подматрицы больше не нужны
	}

	C[n * BLOCK_SIZE * by + BLOCK_SIZE * bx + n * ty + tx] = sum;
}

int main()
{
	setlocale(LC_ALL, "Russian");

	const int k = 400;
	const int n = k * BLOCK_SIZE; // размерность матрицы, кратная BLOCK_SIZE

	/*float matrix1[n*n] = { 0 };
	float matrix2[n*n] = { 0 };
	float mul_matrix[n*n] = { 0 };
	float mul_matrix2[n*n] = { 0 };*/

	float* matrix1;
	matrix1 = new float[n*n];
	float * matrix2;
	matrix2 = new float[n*n];
	float * mul_matrix;
	mul_matrix = new float[n*n];
	float * mul_matrix2;
	mul_matrix2 = new float[n*n];


	// инициализация матриц
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			matrix1[n * i + j] = i * 10 + j;
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
	print_matrix(matrix1, n);
	print_matrix(matrix2, n);
	printf("\n\n");

	cudaError_t cudaStatus = MulMatrixCuda(mul_matrix, mul_matrix2, matrix1, matrix2, n);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	// Вывод матрицы на консоль
	print_matrix(mul_matrix, n);
	print_matrix(mul_matrix2, n);

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
cudaError_t MulMatrixCuda(float* mul_matrix, float* mul_matrix2, float* matrix1, float * matrix2, int n)
{
	int numBytes = n * n * sizeof(float);
	float *dev_matrix1 = 0;
	float *dev_matrix2 = 0;
	float *dev_mul_matrix = 0;
	float *dev_mul_matrix2 = 0;
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

	//Таймер GPU
	cudaEvent_t start, stop; // объявление переменных
	float elapsedTimeInMs = 0;
	cudaEventCreate(&start); // инициализация
	cudaEventCreate(&stop);  // инициализация
	cudaEventRecord(start, 0); // запуск таймера

// Запуск ядра
	mtxMult << <blocks, threads >> > (dev_mul_matrix, dev_matrix1, dev_matrix2, n);

	cudaEventRecord(stop, 0); // остановка времени
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTimeInMs, start, stop);
	printf("Затраченное время 1-го метода GPU: %.8f мс\n\n", elapsedTimeInMs);


	cudaEventCreate(&start); // инициализация
	cudaEventCreate(&stop);  // инициализация
	cudaEventRecord(start, 0); // запуск таймера

	mtxMult2 << <blocks, threads >> > (dev_mul_matrix2, dev_matrix1, dev_matrix2, n);

	cudaEventRecord(stop, 0); // остановка времени
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTimeInMs, start, stop);
	printf("Затраченное время 2-го метода GPU: %.8f мс\n\n", elapsedTimeInMs);



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

void print_matrix(float* mtx, int n)
{
	return;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			float m = mtx[n * i + j];
			printf("%6g|", m);
		}
		printf("\n");
	}
	printf("\n");
}