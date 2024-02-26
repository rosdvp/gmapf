#pragma once
#include <cassert>
#include <cuda_runtime_api.h>
#include <crt/host_defines.h>

namespace cupat
{
	template<typename T>
	class CumList
	{
	public:
		__host__ static CumList* New(int count, int eachCapacity)
		{
			CumList* ptr;
			cudaMallocManaged(&ptr, count * sizeof(CumList));
			for (int i = 0; i < count; i++)
				new (ptr + i) CumList(eachCapacity);
			return ptr;
		}

		__host__ static void Free(CumList* ptr, int dim)
		{
			for (int i = 0; i < dim; i++)
				ptr[i].~CumList();
			cudaFree(ptr);
		}

		__host__ __device__ explicit CumList(int capacity)
		{
			_count = 0;
			_capacity = capacity;
			cudaMallocManaged(&_data, sizeof(T) * capacity);
		}

		__host__ __device__ ~CumList()
		{
			if (_data != nullptr)
			{
				cudaFree(_data);
				_data = nullptr;
			}
		}

		__host__ void PrefetchOnDevice()
		{
			int device = -1;
			cudaGetDevice(&device);
			cudaMemPrefetchAsync(_data, sizeof(T) * _capacity, device, nullptr);
		}

		__host__ __device__ T& At(int idx)
		{
			assert(idx >= 0 && idx < _count);
			return _data[idx];
		}

		__host__ __device__ void Add(const T& val)
		{
			assert(_count < _capacity);
			_data[_count] = val;
			_count += 1;
		}

		__device__ void AddAtomic(const T& val)
		{
			int idx = atomicAdd(&_count, 1);
			assert(idx < _capacity);
			_data[idx] = val;
		}

		__host__ __device__ void RemoveAll()
		{
			_count = 0;
		}

		__host__ __device__ int Count() const
		{
			return _count;
		}

	private:
		int _count = 0;
		int _capacity = 0;
		T* _data = nullptr;
	};
}
