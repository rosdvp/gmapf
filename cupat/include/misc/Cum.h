#pragma once
#include <assert.h>
#include <cstdlib>
#include <cuda_runtime.h>

namespace cupat
{
	template<typename T>
	class Cum
	{
	public:
		template<typename ...Args>
		void HAllocAndMark(int entriesCount, Args... args)
		{
			assert(_host == nullptr);

			_entriesCount = entriesCount;
			_entrySize = T::EvalSize(args...);
			_host = static_cast<char*>(malloc(_entriesCount * _entrySize));
			for (int i = 0; i < entriesCount; i++)
			{
				T h = H(i);
				h.Mark(args...);
			}
		}

		void HFree()
		{
			if (_host == nullptr)
				return;
			free(_host);
			_host = nullptr;
		}

		template<typename ...Args>
		void DAlloc(int entriesCount, Args... args)
		{
			assert(_device == nullptr);

			_entriesCount = entriesCount;
			_entrySize = T::EvalSize(args...);
			cudaMalloc(&_device, _entriesCount * _entrySize);
		}

		void DFree()
		{
			if (_device == nullptr)
				return;
			cudaFree(_device);
		}

		void CopyToDevice()
		{
			assert(_host != nullptr);

			if (_device == nullptr)
				cudaMalloc(&_device, _entriesCount * _entrySize);
			cudaMemcpy(_device, _host, _entriesCount * _entrySize, cudaMemcpyHostToDevice);
		}

		void CopyToHost()
		{
			assert(_device != nullptr);

			if (_host == nullptr)
				_host = static_cast<char*>(malloc(_entriesCount * _entrySize));
			cudaMemcpy(_host, _device, _entriesCount * _entrySize, cudaMemcpyDeviceToHost);
		}

		T H(int idx)
		{
			assert(idx < _entriesCount);
			return T(_host + idx * _entrySize);
		}

		__host__ __device__ T D(int idx)
		{
			assert(idx < _entriesCount);
			return T(_device + idx * _entrySize);
		}

		__host__ __device__ void* DPtr(int idx)
		{
			assert(idx < _entriesCount);
			return _device + idx * _entrySize;
		}

	private:
		int _entriesCount = 0;
		size_t _entrySize = 0;


		char* _host = nullptr;
		char* _device = nullptr;
	};
}
