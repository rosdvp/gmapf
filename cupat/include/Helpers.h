#pragma once

#include <iostream>

#include "cuda_runtime.h"

namespace cupat
{
	inline bool TryCatchCudaError(const char* info)
	{
		cudaError_t error = cudaGetLastError();
		if (error == cudaSuccess)
			return false;

		std::cout << "[cupat] cuda error on " << info << std::endl;
		std::cout << cudaGetErrorName(error) << std::endl;
		std::cout << cudaGetErrorString(error) << std::endl;

		return true;
	}
}
