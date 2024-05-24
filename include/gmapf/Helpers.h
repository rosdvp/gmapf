#pragma once

#include <iostream>

#include <cuda_runtime.h>

namespace gmapf
{
#define TIME_GET std::chrono::high_resolution_clock::now()
#define TIME_DIFF_MS(start, end) std::chrono::duration_cast<std::chrono::microseconds>((end) - (start)).count() / 1000.0f
#define TIME_STD_OUT(text, durSum, durMax, count) std::cout << (text) << " avg: " << ((durSum) / (count)) << " max: " << (durMax) << std::endl

	inline void CudaCheck(const cudaError_t& result, const char* info)
	{
		if (result == cudaSuccess)
			return;

		std::cout << "[gmapf] cuda error, " << info << std::endl;
		std::cout << cudaGetErrorName(result) << std::endl;
		std::cout << cudaGetErrorString(result) << std::endl;
		throw std::exception();
	}

	inline void CudaSyncAndCatch()
	{
		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();
		if (error == cudaSuccess)
			return;

		std::cout << "[gmapf] cuda error" << std::endl;
		std::cout << cudaGetErrorName(error) << std::endl;
		std::cout << cudaGetErrorString(error) << std::endl;
		throw std::exception();
	}

	inline void CudaCatch()
	{
		cudaError_t error = cudaGetLastError();
		if (error == cudaSuccess)
			return;

		std::cout << "[gmapf] cuda error" << std::endl;
		std::cout << cudaGetErrorName(error) << std::endl;
		std::cout << cudaGetErrorString(error) << std::endl;
		throw std::exception();
	}

	inline bool TryCatchCudaError(const char* info)
	{
		cudaError_t error = cudaGetLastError();
		if (error == cudaSuccess)
			return false;

		std::cout << "[gmapf] cuda error on " << info << std::endl;
		std::cout << cudaGetErrorName(error) << std::endl;
		std::cout << cudaGetErrorString(error) << std::endl;

		return true;
	}

	inline void CuDriverCatch(CUresult res)
	{
		if (res == CUDA_SUCCESS)
			return;

		const char* errorName;
		cuGetErrorName(res, &errorName);
		const char* errorStr;
		cuGetErrorString(res, &errorStr);

		std::cout << "[gmapf] cuda error" << std::endl;
		std::cout << errorName << std::endl;
		std::cout << errorStr << std::endl;
		throw std::exception();
	}
}
