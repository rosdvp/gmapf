#pragma once

#include <assert.h>
#include <cstdlib>

#include "cuda_runtime.h"

namespace cupat
{
	template<typename T>
	class Array
	{
	public:
		Array() = default;
		~Array() = default;
		Array(const Array& m) = delete;
		Array(Array&& m) = delete;
		Array& operator=(const Array& m) = delete;
		Array& operator=(Array&& m) = delete;

		__host__ void AllocOnHost(int count)
		{
			_size = EvalSize(count);
			void* p = malloc(_size);
			Attach(p);
			*_count = count;
			_size = EvalSize(count);
		}

		__host__ void FreeOnHost()
		{
			free(_ptr);
			_ptr = nullptr;
		}

		__host__ void* AllocOnDeviceAndCopyFromHost()
		{
			void* p;
			cudaMalloc(&p, _size);
			cudaMemcpy(p, _ptr, _size, cudaMemcpyHostToDevice);
			return p;
		}

		__host__ __device__ void Attach(void* p)
		{
			_ptr = p;

			auto ptr = static_cast<char*>(p);
			_count = reinterpret_cast<int*>(ptr);
			ptr += 16;
			_data = reinterpret_cast<T*>(ptr);
		}

		__host__ __device__ T& At(int idx)
		{
			assert(idx >= 0 && idx < *_count);
			return _data[idx];
		}

		__host__ __device__ const T& At(int idx) const
		{
			assert(idx >= 0 && idx < *_count);
			return _data[idx];
		}

		__host__ __device__ int GetCount() const
		{
			return *_count;
		}

		__host__ __device__ void* GetRawPtr()
		{
			return _ptr;
		}

		__host__ __device__ size_t GetRawSize()
		{
			return _size;
		}

		__host__ __device__ constexpr static size_t EvalSize(int count)
		{
			return 16 + sizeof(T) * count;
		}

	private:
		void* _ptr = nullptr;
		int* _count = nullptr;
		T* _data = nullptr;

		size_t _size = 0;
	};
}
