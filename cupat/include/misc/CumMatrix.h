#pragma once
#include <cassert>
#include <cuda_runtime_api.h>

namespace cupat
{
	template<typename T>
	class CumMatrix
	{
	private:
		struct Entry
		{
			int IsFilled;
			T Value;
		};

	public:
		__host__ __device__ static CumMatrix* New(int dim, int countX, int countY)
		{
			CumMatrix* p;
			cudaMallocManaged(&p, sizeof(CumMatrix) * dim);
			for (int i = 0; i < dim; i++)
				new (p + i) CumMatrix(countX, countY);
			return p;
		}

		__host__ __device__ explicit CumMatrix(int countX, int countY)
		{
			_countX = countX;
			_countY = countY;
			_count = countX * countY;

			cudaMallocManaged(&_entries, sizeof(Entry) * _count);
			for (int i = 0; i < _count; i++)
				_entries[i].IsFilled = 0;
		}

		__host__ __device__ ~CumMatrix()
		{
			if (_entries != nullptr)
			{
				cudaFree(_entries);
				_entries = nullptr;
			}
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

		__host__ __device__ int CountX() const
		{
			return _countX;
		}

		__host__ __device__ int CountY() const
		{
			return _countY;
		}

		__host__ __device__ bool IsValid(int x, int y) const
		{
			return x >= 0 && x < _countX && y >= 0 && y < _countY;
		}

		__host__ __device__ int GetIdx(int x, int y) const
		{
			int idx = y * _countX + x;
			assert(idx < _count);
			return idx;
		}

	private:
		Entry* _entries;
		int _countX;
		int _countY;
		int _count;
	};
}
