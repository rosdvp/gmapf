#pragma once
#include <cassert>
#include <cuda_runtime_api.h>

namespace cupat
{
	template<typename T>
	class CuQueue
	{
	public:

		__host__ __device__ explicit CuQueue(void* ptr)
		{
			auto p = static_cast<char*>(ptr);
			_capacity = reinterpret_cast<int*>(p);
			p += 8;
			_count = reinterpret_cast<int*>(p);
			p += 8;
			_data = reinterpret_cast<T*>(p);
		}

		__host__ __device__ void Mark(int capacity)
		{
			*_capacity = capacity;
			*_count = 0;
		}

		__host__ __device__ int Count() const
		{
			return *_count;
		}

		__host__ __device__ void Push(const T& item)
		{
			int idx = 0;
			for (; idx < *_count; idx++)
				if (item.F > _data[idx].F)
					break;

			for (int i = *_count; i > idx; i--)
				_data[i] = _data[i - 1];

			_data[idx] = item;
			if (*_count < *_capacity)
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

		__host__ __device__ static size_t EvalSize(int capacity)
		{
			return 8 + 8 + sizeof(T) * capacity;
		}

	private:
		int* _capacity = nullptr;
		int* _count = nullptr;
		T* _data = nullptr;
	};
}
