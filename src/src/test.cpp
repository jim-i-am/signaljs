#include <iostream>

#include "gpu_vector.hpp"

int main()
{
	GpuVector v(1000);
	std::cout << "v.size = " v.size() << std::endl;
}