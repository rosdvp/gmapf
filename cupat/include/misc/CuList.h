#pragma once
#include <cassert>
#include <cuda_runtime_api.h>

namespace cupat
{
	template<typename T>
	class CuList
	{
	public:
		__host__ __device__ explicit CuList(void* ptr)
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

		__host__ __device__ T& At(int idx)
		{
			assert(idx >= 0 && idx < *_count);
			return _data[idx];
		}

		__host__ __device__ void Add(const T& val)
		{
			assert(*_count < *_capacity);
			_data[*_count] = val;
			*_count += 1;
		}

		__device__ void AddAtomic(const T& val)
		{
			int idx = atomicAdd(_count, 1);
			assert(idx < *_capacity);
			_data[idx] = val;
		}

		__host__ __device__ void RemoveAll()
		{
			*_count = 0;
		}

		__host__ __device__ int Count() const
		{
			return *_count;
		}

		__host__ __device__ int Capacity() const
		{
			return *_capacity;
		}

		__host__ __device__ void Reverse() const
		{
			int count = *_count;

			for (int i = 0; i < count / 2; i++)
			{
				int r = count - i - 1;
				T temp = _data[i];
				_data[i] = _data[r];
				_data[r] = temp;
			}
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
