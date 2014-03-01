#include "gpu_vector.hpp"
#include "cream_cuda.h"

GpuVector::GpuVector(int n) : length(n) {
	gpu_malloc(&ptr_, length);
}

GpuVector::~GpuVector() {
  gpu_free(ptr);
}

int GpuVector::size() {
  return length * sizeof(int);
}