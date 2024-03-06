#pragma once

#include <iostream>

#include <cuda_runtime_api.h>

namespace cupat
{
#define TIME_GET std::chrono::high_resolution_clock::now()
#define TIME_SET(x) (x) = TIME_GET
#define TIME_STAMP(x) auto (x) = TIME_GET
#define TIME_DIFF_MS(start) std::chrono::duration_cast<std::chrono::microseconds>(TIME_GET - (start)).count() / 1000.0f
#define TIME_COUNTER_ADD(start, counter) (counter) += std::chrono::duration_cast<std::chrono::microseconds>(TIME_GET - (start)).count()
#define TIME_COUNTER_GET(counter) ((counter) / 1000.0f)

	inline void CudaCheck(const cudaError_t& result, const char* info)
	{
		if (result == cudaSuccess)
			return;

		std::cout << "[cupat] cuda error, " << info << std::endl;
		std::cout << cudaGetErrorName(result) << std::endl;
		std::cout << cudaGetErrorString(result) << std::endl;
		throw std::exception();
	}

	inline void CudaCatch()
	{
		cudaError_t error = cudaGetLastError();
		if (error == cudaSuccess)
			return;

		std::cout << "[cupat] cuda error" << std::endl;
		std::cout << cudaGetErrorName(error) << std::endl;
		std::cout << cudaGetErrorString(error) << std::endl;
		throw std::exception();
	}

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
