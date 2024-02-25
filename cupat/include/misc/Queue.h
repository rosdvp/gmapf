#pragma once
#include <cstdlib>
#include <assert.h>
#include "cuda_runtime.h"

namespace cupat
{
	template<typename T>
	class Queue
	{
	public:
		Queue() = default;
		~Queue() = default;
		Queue(const Queue& other) = delete;
		Queue(Queue&& other) = delete;
		Queue& operator=(const Queue& other) = delete;
		Queue& operator=(Queue&& other) = delete;

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


		__host__ __device__ int Count() const
		{
			return *_count;
		}

		__host__ __device__ void Push(const volatile T& item)
		{
			assert(*_count < *_capacity);

			int idx = 0;
			for (; idx < *_count; idx++)
				if (item.F > _data[idx].F)
					break;

			for (int i = *_count; i > idx; i--)
				_data[i] = _data[i - 1];

			_data[idx] = item;
			*_count += 1;
		}

		__host__ __device__ T Pop()
		{
			assert(*_count > 0);
			*_count -= 1;
			return _data[*_count];
		}

		__host__ __device__ void RemoveAll()
		{
			*_count = 0;
		}

		__host__ __device__ const T& DebugGet(int idx) const
		{
			return _data[idx];
		}

		__host__ __device__ constexpr size_t EvalSize(int capacity)
		{
			return 16 + 16 + sizeof(T) * capacity;
		}

	private:
		size_t _size;
		void* _ptr = nullptr;

		int* _count = nullptr;
		int* _capacity = nullptr;
		T* _data = nullptr;
	};
}