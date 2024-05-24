#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <ostream>

#include "V2Int.h"

namespace gmapf
{
	struct V2Float
	{
		float X;
		float Y;

		__host__ __device__ V2Float() : X(0), Y(0) {}
		__host__ __device__ V2Float(float x, float y) : X(x), Y(y) {}

		__host__ __device__ V2Float(const V2Int& v) : X(v.X), Y(v.Y) {}


		__host__ __device__ float GetLengthSqr() const
		{
			return X * X + Y * Y;
		}

		__host__ __device__ float GetLength() const
		{
			return sqrt(X * X + Y * Y);
		}

		__host__ __device__ V2Float GetNorm() const
		{
			float length = GetLength();
			return { X / length, Y / length };
		}

		__host__ __device__ V2Float GetRotated(float angle) const
		{
			angle = angle * CUDART_PI_F / 180.0f;
			float cs = cos(angle);
			float sn = sin(angle);
			float x = X * cs - Y * sn;
			float y = X * sn + Y * cs;
			return { x, y };
		}


		__host__ __device__ static float DistSqr(const V2Float& v1, const V2Float& v2)
		{
			float x = v1.X - v2.X;
			float y = v1.Y - v2.Y;
			return x * x + y * y;
		}



		__host__ __device__ friend V2Float operator+(const V2Float& v1, const V2Float& v2)
		{
			return { v1.X + v2.X, v1.Y + v2.Y };
		}

		__host__ __device__ V2Float& operator+=(const V2Float& v)
		{
			X += v.X;
			Y += v.Y;
			return *this;
		}

		__host__ __device__ friend V2Float operator-(const V2Float& v)
		{
			return { -v.X, -v.Y };
		}

		__host__ __device__ friend V2Float operator-(const V2Float& v1, const V2Float& v2)
		{
			return { v1.X - v2.X, v1.Y - v2.Y };
		}

		__host__ __device__ friend V2Float operator*(const V2Float& v, float k)
		{
			return { v.X * k, v.Y * k };
		}

		__host__ __device__ friend V2Float operator/(const V2Float& v, float k)
		{
			return { v.X / k, v.Y / k };
		}

		__host__ __device__ friend bool operator==(const V2Float& v1, const V2Float& v2)
		{
			constexpr float epsilon = 0.00001f;
			return fabs(v1.X - v2.X) < epsilon && fabs(v1.Y - v2.Y) < epsilon;
		}

		__host__ __device__ friend bool operator!=(const V2Float& v1, const V2Float& v2)
		{
			return !(v1 == v2);
		}


		__host__ friend std::ostream& operator<<(std::ostream& os, const V2Float& v)
		{
			os << "(" << v.X << ", " << v.Y << ")";
			return os;
		}

		__host__ std::string ToString() const
		{
			return "(" + std::to_string(X) + ", " + std::to_string(Y) + ")";
		}
	};
}
