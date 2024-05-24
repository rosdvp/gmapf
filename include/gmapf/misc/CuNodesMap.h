#pragma once
#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>

namespace gmapf
{
	class CuNodesMap
	{
	public:
		constexpr static int INVALID = -1;
		constexpr static int NEIBS_MAX_COUNT = 3;

		struct __align__(16) Node
		{
			V2Float P1;
			V2Float P2;
			V2Float P3;
			V2Float PCenter;
			int NeibsIdx[NEIBS_MAX_COUNT];
		};

	public:
		__host__ __device__ explicit CuNodesMap(void* ptr)
		{
			auto p = static_cast<char*>(ptr);
			_count = reinterpret_cast<int*>(p);
			p += sizeof(int*);
			_nodes = reinterpret_cast<Node*>(p);
		}

		__host__ __device__ void Mark(int count)
		{
			*_count = count;
			for (int i = 0; i < *_count; i++)
				for (auto& neib : _nodes[i].NeibsIdx)
					neib = INVALID;
		}

		__host__ __device__ Node& At(int idx)
		{
			assert(idx >= 0 && idx < *_count);
			return _nodes[idx];
		}

		__host__ __device__ bool IsInNode(const V2Float& pos, int nodeIdx)
		{
			assert(nodeIdx >= 0 && nodeIdx < *_count);
			auto& node = _nodes[nodeIdx];
			return IsPointInTriangle(pos, node.P1, node.P2, node.P3);
		}

		__host__ __device__ bool TryGetNodeIdx(const V2Float& pos, int* outNodeIdx)
		{
			for (int i = 0; i < *_count; i++)
			{
				auto& node = _nodes[i];
				if (IsPointInTriangle(pos, node.P1, node.P2, node.P3))
				{
					if (outNodeIdx != nullptr)
						*outNodeIdx = i;
					return true;
				}
			}
			return false;
		}

		__host__ __device__ V2Float GetPos(int nodeIdx)
		{
			assert(nodeIdx >= 0 && nodeIdx < *_count);
			return _nodes[nodeIdx].PCenter;
		}

		__host__ __device__ float GetDistSqr(int nodeIdxA, int nodeIdxB)
		{
			assert(nodeIdxA >= 0 && nodeIdxA < *_count && nodeIdxB >= 0 && nodeIdxB < *_count);
			auto a = _nodes[nodeIdxA].PCenter;
			auto b = _nodes[nodeIdxB].PCenter;
			return V2Float::DistSqr(a, b);
		}

		__host__ __device__ int Count()
		{
			return *_count;
		}

		__host__ __device__ static constexpr size_t EvalSize(int count)
		{
			return sizeof(int*) + sizeof(Node) * count;
		}

	private:
		int* _count = nullptr;
		Node* _nodes = nullptr;


		__host__ __device__ bool IsPointInTriangle(V2Float p, V2Float v1, V2Float v2, V2Float v3)
		{
			float d1 = GetSign(p, v1, v2);
			float d2 = GetSign(p, v2, v3);
			float d3 = GetSign(p, v3, v1);

			bool isNeg = (d1 < 0) || (d2 < 0) || (d3 < 0);
			bool isPos = (d1 > 0) || (d2 > 0) || (d3 > 0);

			return !(isNeg && isPos);
		}

		__host__ __device__ float GetSign(V2Float p1, V2Float p2, V2Float p3) const
		{
			return (p1.X - p3.X) * (p2.Y - p3.Y) - (p2.X - p3.X) * (p1.Y - p3.Y);
		}
	};
}
