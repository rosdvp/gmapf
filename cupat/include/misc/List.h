#pragma once

#include <cstdlib>
#include <assert.h>
#include <cuda_runtime.h>

namespace cupat
{
	template <typename T>
	class List
	{
	public:
		List() = default;
		~List() = default;
		List(const List& other) = delete;
		List(List&& other) = delete;
		List& operator=(const List& other) = delete;
		List& operator=(List&& other) = delete;

		__host__ void AllocOnHost(int capacity)
		{
			_size = EvalSize(capacity);
			void* p = malloc(_size);
			Attach(p);
			*_capacity = capacity;
			*_count = 0;
			_size = EvalSize(capacity);
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
			_capacity = reinterpret_cast<int*>(ptr);
			ptr += 16;
			_count = reinterpret_cast<int*>(ptr);
			ptr += 16;
			_data = reinterpret_cast<T*>(ptr);

			_size = EvalSize(*_capacity);
		}

		__host__ __device__ T& At(int idx)
		{
			assert(idx >= 0 && idx < *_count);
			return _data[idx];
		}

		__device__ void Add(const T& val)
		{
			*_count += 1;
			assert(*_count < *_capacity);
			_data[*_count] = val;
		}

		__device__ int DAddAtomic(const T& val)
		{
			int idx = atomicAdd(_count, 1);
			assert(idx < *_capacity);
			_data[idx] = val;
			return idx;
		}

		__host__ __device__ void RemoveAll()
		{
			*_count = 0;
		}

		__host__ __device__ int Count() const
		{
			return *_count;
		}

		__host__ __device__ constexpr static size_t EvalSize(int capacity)
		{
			return 16 + 16 + sizeof(T) * capacity;
		}

	private:
		size_t _size;
		void* _ptr = nullptr;

		int* _capacity = nullptr;
		int* _count = nullptr;
		T* _data = nullptr;
	};
}
