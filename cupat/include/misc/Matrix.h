#pragma once
#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>

#include "V2Int.h"
#include "../Helpers.h"

namespace cupat
{
	template<typename T>
	class Matrix
	{
	public:
		Matrix() = default;
		~Matrix() = default;
		Matrix(const Matrix& m) = delete;
		Matrix(Matrix&& m) = delete;
		Matrix& operator=(const Matrix& m) = delete;
		Matrix& operator=(Matrix&& m) = delete;


		__host__ void HAlloc(int countX, int countY)
		{
			_size = EvalSize(countX, countY);
			void* p = malloc(_size);
			Attach(p);
			*_countX = countX;
			*_countY = countY;
			_count = (*_countX) * (*_countY);
		}

		__device__ void DAlloc(int countX, int countY)
		{
			_size = EvalSize(countX, countY);
			void* p;
			cudaMalloc(&p, _size);
			Attach(p);
			*_countX = countX;
			*_countY = countY;
			_count = (*_countX) * (*_countY);
		}

		__host__ void* AllocOnDeviceAndCopyFromHost()
		{
			void* p = nullptr;
			cudaMalloc(&p, _size);
			TryCatchCudaError("Matrix::DAllocCopy malloc");
			cudaMemcpy(p, _ptr, _size, cudaMemcpyHostToDevice);
			TryCatchCudaError("Matrix::DAllocCopy memcpy");
			return p;
		}

		__host__ void HFree()
		{
			free(_ptr);
			_ptr = nullptr;
		}


		__host__ __device__ void Attach(void* p)
		{
			_ptr = p;

			auto ptr = static_cast<char*>(p);
			_countX = reinterpret_cast<int*>(ptr);
			ptr += sizeof(int);
			_countY = reinterpret_cast<int*>(ptr);
			ptr += sizeof(int);
			_data = reinterpret_cast<T*>(ptr);

			_count = (*_countX) * (*_countY);
		}

		__host__ __device__ T& At(const V2Int& coord)
		{
			int idx = CoordToIdx(coord);
			return _data[idx];
		}

		__host__ __device__ bool IsValid(const V2Int& coord)
		{
			return coord.X >= 0 && coord.X < *_countX && coord.Y >= 0 && coord.Y < *_countY;
		}

		__host__ __device__ int CoordToIdx(const V2Int& coord) const
		{
			assert(coord.X >= 0 && coord.X < *_countX);
			assert(coord.Y >= 0 && coord.Y < *_countY);
			return coord.X + coord.Y * (*_countX);
		}

		__host__ __device__ V2Int IdxToCoord(int idx) const
		{
			assert(idx >= 0 && idx < _count);

			int y = idx / *_countX;
			int x = idx % *_countX;
			return { x, y };
		}

		__host__ __device__ int GetCountX() const
		{
			return *_countX;
		}

		__host__ __device__ int GetCountY() const
		{
			return *_countY;
		}

		__host__ __device__ void* GetRawPtr() const
		{
			return _ptr;
		}

		constexpr static size_t EvalSize(int countX, int countY)
		{
			return sizeof(int) + sizeof(int) + sizeof(T) * countX * countY;
		}

	private:
		void* _ptr = nullptr;
		int* _countX = nullptr;
		int* _countY = nullptr;
		T* _data = nullptr;

		size_t _size = 0;
		int _count = -1;
	};
}
