
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <locale>

#include <stdio.h>

cudaError_t MulMatrixCuda(double **mul_matrix, double **matrix1, double **matrix2, unsigned int m1, unsigned int n1, unsigned int m2, unsigned int n2);

__global__ void mulKernel(double **mul_matrix, double **matrix1, double **matrix2)
{
	int i = threadIdx.x;



	
}

int main()
{
	setlocale(LC_ALL, "Russian");

	const int m1 = 5;
	const int n1 = 9;
	const int m2 = 9;
	const int n2 = 5;

	double matrix1[m1][n1] = { 0 };
	double matrix2[m2][n2] = { 0 };
	double mul_matrix[m1][n2] = { 0 };
	//double **test = &mul_matrix;

	if (n1 != m2)
	{
		printf("Размерности матриц не совпадают");
		return 0;
	}

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
	}


	cudaError_t cudaStatus = MulMatrixCuda((double**)mul_matrix, (double**)matrix1, (double**)matrix2, m1, n1, m2, n2);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	/*printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);*/

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
cudaError_t MulMatrixCuda(double **mul_matrix, double **matrix1, double **matrix2,
	unsigned int m1, unsigned int n1, unsigned int m2, unsigned int n2)
{
	double *dev_matrix1 = 0;
	double *dev_matrix2 = 0;
	double *dev_mul_matrix = 0;
	cudaError_t cudaStatus;


	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}


	cudaStatus = cudaMalloc((void**)&dev_mul_matrix, m1 * n2 * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_matrix1, m1 * n1 * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_matrix2, m2 * n2 * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input matrix from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_matrix1, matrix1, m1*n1 * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_matrix2, matrix2, m2*n2 * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	mulKernel << <1, size >> > (dev_mul_matrix, dev_matrix1, dev_matrix2);

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
	cudaStatus = cudaMemcpy(mul_matrix, dev_mul_matrix, m1*n2 * sizeof(double), cudaMemcpyDeviceToHost);
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
