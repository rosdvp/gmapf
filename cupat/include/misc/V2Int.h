#pragma once
#include <sstream>

#include "cuda_runtime.h"

namespace cupat
{
	struct V2Int
	{
		int X;
		int Y;

		__host__ __device__ V2Int(): X(0), Y(0) {}
		__host__ __device__ V2Int(int x, int y) : X(x), Y(y) {}


		__host__ __device__ static float DistSqr(const V2Int& v1, const V2Int& v2)
		{
			int x = v1.X - v2.X;
			int y = v1.Y - v2.Y;
			return x * x + y * y;
		}


		__host__ __device__ size_t GetHash() const
		{
			size_t seed = X;
			seed = seed << 32;
			seed = seed | Y;
			//size_t seed = 0;
			//seed ^= X + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			//seed ^= Y + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			return seed;
		}

		__host__ __device__ friend V2Int operator+(const V2Int& v1, const V2Int& v2)
		{
			return { v1.X + v2.X, v1.Y + v2.Y };
		}


		__host__ __device__ friend bool operator==(const V2Int& v1, const V2Int& v2)
		{
			return v1.X == v2.X && v1.Y == v2.Y;
		}

		__host__ __device__ friend bool operator!=(const V2Int& v1, const V2Int& v2)
		{
			return v1.X != v2.X || v1.Y != v2.Y;
		}

		__host__ __device__ friend V2Int operator-(const V2Int& v1, const V2Int& v2)
		{
			return { v1.X - v2.X, v1.Y - v2.Y };
		}
	};
}

template <>
struct std::hash<cupat::V2Int>
{
	std::size_t operator()(const cupat::V2Int& k) const
	{
		return k.GetHash();
	}
};
