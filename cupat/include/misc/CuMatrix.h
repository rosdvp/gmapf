#pragma once
#include <cassert>
#include <cuda_runtime_api.h>

namespace cupat
{
	template<typename T>
	class CuMatrix
	{
	private:
		struct Entry
		{
			int IsFilled;
			T Value;
		};

	public:
		__host__ __device__ CuMatrix(void* ptr)
		{
			auto p = static_cast<char*>(ptr);
			_countX = reinterpret_cast<int*>(p);
			p += 8;
			_countY = reinterpret_cast<int*>(p);
			p += 8;
			_entries = reinterpret_cast<Entry*>(p);
		}

		__host__ __device__ void Mark(int countX, int countY)
		{
			*_countX = countX;
			*_countY = countY;

			int count = countX * countY;
			for (int i = 0; i < count; i++)
				_entries[i].IsFilled = false;
		}

		__host__ __device__ bool Has(int idx) const
		{
			return _entries[idx].IsFilled == 1;
		}

		__host__ __device__ bool Has(int x, int y) const
		{
			return Has(GetIdx(x, y));
		}

		__device__ bool TryOccupy(int idx) const
		{
			int res = atomicExch(&(_entries[idx].IsFilled), 1);
			return res == 0;
		}

		__host__ __device__ T& At(int idx)
		{
			return _entries[idx].Value;
		}

		__host__ __device__ T& At(int x, int y)
		{
			return At(GetIdx(x, y));
		}

		__host__ __device__ T& At(const V2Int& coord)
		{
			return At(GetIdx(coord.X, coord.Y));
		}

		__host__ __device__ int CountX() const
		{
			return *_countX;
		}

		__host__ __device__ int CountY() const
		{
			return *_countY;
		}

		__host__ __device__ int Count() const
		{
			return (*_countX) * (*_countY);
		}

		__host__ __device__ bool IsValid(int x, int y) const
		{
			return x >= 0 && x < *_countX && y >= 0 && y < *_countY;
		}

		__host__ __device__ bool IsValid(const V2Int& coord) const
		{
			return coord.X >= 0 && coord.X < *_countX && coord.Y >= 0 && coord.Y < *_countY;
		}

		__host__ __device__ int GetIdx(int x, int y) const
		{
			int idx = y * (*_countX) + x;
			assert(idx < Count());
			return idx;
		}


		__host__ __device__ static size_t EvalSize(int countX, int countY)
		{
			return 8 + 8 + sizeof(Entry) * countX * countY;
		}

	private:
		int* _countX = nullptr;
		int* _countY = nullptr;
		Entry* _entries = nullptr;
	};
}
