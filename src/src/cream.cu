#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <cuda.h>
#include "cream.h"

#include <iostream>

void gpu_malloc(int** ptr, const int n)
{
	std::cout << "gpu_malloc: " << n << std::endl;
	cudaMalloc((void**)ptr, n * sizeof(int));
}

void cpu_malloc(int** ptr, const int n)
{
	std::cout << "cpu_malloc: " << n << std::endl;
	cudaMallocHost((void**)ptr, n * sizeof(int));
}

void gpu_free(int* ptr)
{
	std::cout << "gpu_free: " << std::endl;
	cudaFree(ptr);
}

void cpu_free(int* ptr)
{
	std::cout << "gpu_free: " << std::endl;
	cudaFreeHost(ptr);
}


void copy_gpu_to_cpu(int* gpu_ptr, int* cpu_ptr, const int n)
{
	std::cout << "copy_gpu_to_cpu: " << n << std::endl;
	cudaMemcpy(cpu_ptr, gpu_ptr, n * sizeof(int), cudaMemcpyDeviceToHost);
}

void copy_cpu_to_gpu(int* gpu_ptr, int* cpu_ptr, const int n)
{
	std::cout << "copy_cpu_to_gpu: " << n << std::endl;
	cudaMemcpy(gpu_ptr, cpu_ptr, n * sizeof(int), cudaMemcpyHostToDevice);
}

void gpu_fill(int* ptr, int n, int val)
{
	std::cout << "gpu_fill: val=" << val << " n=" << n << std::endl;	
	thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(ptr);
	thrust::fill(dev_ptr, dev_ptr+n, val);
}

void gpu_seq(int* ptr, int n, int val)
{
	std::cout << "gpu_seq: val=" << val << " n=" << n << std::endl;	
	thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(ptr);
	thrust::sequence(dev_ptr, dev_ptr + n, val);
}

int gpu_sum(int* ptr, int n)
{
	std::cout << "gpu_sum "<< std::endl;	
	thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(ptr);
	return thrust::reduce(dev_ptr, dev_ptr+n);
}

void gpu_sums(int* in_ptr, int* out_ptr,int n)
{
	std::cout << "gpu_sum "<< std::endl;	
	thrust::device_ptr<int> in_dev_ptr = thrust::device_pointer_cast(in_ptr);
	thrust::device_ptr<int> out_dev_ptr = thrust::device_pointer_cast(out_ptr);
	thrust::inclusive_scan(in_dev_ptr, in_dev_ptr+n, out_dev_ptr);
}

int gpu_prod(int* ptr, int n)
{
	std::cout << "gpu_prod "<< std::endl;	
	thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(ptr);
	return thrust::reduce(dev_ptr, dev_ptr+n, 1, thrust::multiplies<int>());
}

void gpu_prods(int* in_ptr, int* out_ptr,int n)
{
	std::cout << "gpu_prods "<< std::endl;	
	thrust::device_ptr<int> in_dev_ptr = thrust::device_pointer_cast(in_ptr);
	thrust::device_ptr<int> out_dev_ptr = thrust::device_pointer_cast(out_ptr);
	thrust::inclusive_scan(in_dev_ptr, in_dev_ptr+n, out_dev_ptr, thrust::multiplies<int>());
}

int gpu_get(int* ptr, int n, int i) 
{
	std::cout << "gpu_get "<< i <<std::endl;	
	int v = 0;
	cudaMemcpy(&v, 
		ptr+i, 
		sizeof(int), 
		cudaMemcpyDeviceToHost);
	return v;
}