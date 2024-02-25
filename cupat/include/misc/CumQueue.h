#pragma once
#include <cassert>
#include <cuda_runtime_api.h>

namespace cupat
{
	template<typename T>
	class CumQueue
	{
	public:
		static CumQueue* New(int count, int eachCapacity)
		{
			CumQueue* ptr;
			cudaMallocManaged(&ptr, count * sizeof(CumQueue));
			for (int i = 0; i < count; i++)
				new (ptr + i) CumQueue(eachCapacity);
			return ptr;
		}

		explicit CumQueue(int capacity)
		{
			_count = 0;
			_capacity = capacity;
			cudaMallocManaged(&_data, sizeof(T) * capacity);
		}

		~CumQueue()
		{
			if (_data != nullptr)
			{
				cudaFree(_data);
				_data = nullptr;
			}
		}

		__host__ __device__ int Count() const
		{
			return _count;
		}

		__host__ __device__ void Push(const T& item)
		{
			int idx = 0;
			for (; idx < _count; idx++)
				if (item.F > _data[idx].F)
					break;

			for (int i = _count; i > idx; i--)
				_data[i] = _data[i - 1];

			_data[idx] = item;
			if (_count < _capacity)
				_count += 1;
		}

		__host__ __device__ T Pop()
		{
			assert(_count > 0);
			_count -= 1;
			return _data[_count];
		}

	private:
		int _count;
		int _capacity;
		T* _data = nullptr;
	};
}
