#pragma once
#include <cassert>
#include <cuda_runtime.h>

namespace gmapf
{
	template<typename T>
	class CuList
	{
	public:
		__host__ __device__ explicit CuList(void* ptr)
		{
			auto p = static_cast<char*>(ptr);
			_capacity = reinterpret_cast<int*>(p);
			p += sizeof(int*);
			_count = reinterpret_cast<int*>(p);
			p += sizeof(int*);
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

		__host__ __device__ int Add(const T& val)
		{
			int idx = *_count;
			assert(idx < *_capacity);
			_data[idx] = val;
			*_count += 1;
			return idx;
		}

		__device__ void AddAtomic(const T& val)
		{
			int idx = atomicAdd(_count, 1);
			assert(idx < *_capacity);
			_data[idx] = val;
		}

		__host__ __device__ void PreAdd(int count)
		{
			assert(*_count + count <= *_capacity);
			*_count += count;
		}

		__device__ bool TryPopLastAtomic(T& out)
		{
			int idx = atomicSub(_count, 1);
			if (idx <= 0)
			{
				*_count = 0;
				return false;
			}
			out = _data[idx-1];
			return true;
		}

		__device__ T PopLastAtomic()
		{
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
			return sizeof(int*) + sizeof(int*) + sizeof(T) * capacity;
		}

	private:
		int* _capacity = nullptr;
		int* _count = nullptr;
		T* _data = nullptr;
	};
}
