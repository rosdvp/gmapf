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
			int count = *_count;
			int capacity = *_capacity;

			int idx = -1;
			for (int i = count-1; i >= 0; i--)
				if (item.F < _data[i].F)
				{
					idx = i;
					break;
				}

			if (count < capacity)
			{
				idx += 1;

				for (int i = count; i > idx; i--)
					_data[i] = _data[i - 1];
				*_count += 1;
			}
			else
			{
				if (idx == -1)
					return;

				for (int i = 0; i < idx; i++)
					_data[i] = _data[i + 1];
			}
			_data[idx] = item;

			//printf("queue push (%d, %d):\n", item.Cell.X, item.Cell.Y);
			//for (int i = 0; i < *_count; i++)
			//	printf("[%d] (%d, %d) F=(%f)\n", i, _data[i].Cell.X, _data[i].Cell.Y, _data[i].F);
		}

		__host__ __device__ T Pop()
		{
			assert(*_count > 0);
			*_count -= 1;
			//printf("pop (%d, %d)\n", _data[*_count].Cell.X, _data[*_count].Cell.Y);
			return _data[*_count];
		}

		__host__ __device__ void RemoveAll()
		{
			*_count = 0;
		}

		__host__ __device__ static constexpr size_t EvalSize(int capacity)
		{
			return 8 + 8 + sizeof(T) * capacity;
		}

	private:
		int* _capacity = nullptr;
		int* _count = nullptr;
		T* _data = nullptr;
	};
}
