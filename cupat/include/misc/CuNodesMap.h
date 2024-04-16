#pragma once
#include <cassert>
#include <cstdio>
#include <cuda_runtime_api.h>

#include "../Defines.h"

namespace cupat
{
	class CuNodesMap
	{
	public:
		constexpr static int INVALID = -1;
#ifdef CUPAT_NAV_MESH
		constexpr static int NEIBS_MAX_COUNT = 3;
#else
		constexpr static int NEIBS_MAX_COUNT = 8;
#endif

		struct __align__(16) Desc
		{
#ifndef  CUPAT_NAV_MESH
			float CellSize;
			int CellsCountX;
			int CellsCountY;
#endif
			int Count;
		};

		struct __align__(16) Node
		{
#ifdef CUPAT_NAV_MESH
			V2Float P1;
			V2Float P2;
			V2Float P3;
			V2Float PCenter;
#endif
			int Val;
			int NeibsIdx[NEIBS_MAX_COUNT];
		};

	public:
		__host__ __device__ explicit CuNodesMap(void* ptr)
		{
			auto p = static_cast<char*>(ptr);
			_desc = reinterpret_cast<Desc*>(p);
			p += sizeof(Desc);
			_nodes = reinterpret_cast<Node*>(p);
		}

		__host__ __device__ void Mark(const Desc& desc)
		{
			*_desc = desc;
			for (int i = 0; i < desc.Count; i++)
			{
				_nodes[i].Val = INVALID;
				for (auto& neib : _nodes[i].NeibsIdx)
					neib = INVALID;
			}
		}

		__host__ __device__ Node& At(int idx)
		{
			assert(idx >= 0 && idx < _desc->Count);
			return _nodes[idx];
		}

		__host__ __device__ bool IsInNode(const V2Float& pos, int nodeIdx)
		{
			assert(nodeIdx >= 0 && nodeIdx < _desc->Count);
#ifdef CUPAT_NAV_MESH
			auto& node = _nodes[nodeIdx];
			return IsPointInTriangle(pos, node.P1, node.P2, node.P3);
#else
			int x = static_cast<int>(pos.X / _desc->CellSize);
			if (x < 0 || x >= _desc->CellsCountX)
				return false;
			int y = static_cast<int>(pos.Y / _desc->CellSize);
			if (y < 0 || y >= _desc->CellsCountY)
				return false;
			int idx = y * _desc->CellsCountX + x;
			return nodeIdx == idx;
#endif
		}

		__host__ __device__ bool TryGetNodeIdx(const V2Float& pos, int* outNodeIdx)
		{
#ifdef CUPAT_NAV_MESH
			for (int i = 0; i < _desc->Count; i++)
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
#else
			int x = static_cast<int>(pos.X / _desc->CellSize);
			if (x < 0 || x >= _desc->CellsCountX)
				return false;
			int y = static_cast<int>(pos.Y / _desc->CellSize);
			if (y < 0 || y >= _desc->CellsCountY)
				return false;
			int idx = y * _desc->CellsCountX + x;
			if (outNodeIdx != nullptr)
				*outNodeIdx = idx;
			return _nodes[idx].Val != INVALID;
#endif
		}

		__host__ __device__ V2Float GetPos(int nodeIdx)
		{
			assert(nodeIdx >= 0 && nodeIdx < _desc->Count);
#ifdef CUPAT_NAV_MESH
			return _nodes[nodeIdx].PCenter;
#else
			float x = (nodeIdx % _desc->CellsCountX) * _desc->CellSize + _desc->CellSize / 2.0f;
			float y = (nodeIdx / _desc->CellsCountX) * _desc->CellSize + _desc->CellSize / 2.0f;
			return { x , y };
#endif
		}

		__host__ __device__ float GetDistSqr(int nodeIdxA, int nodeIdxB)
		{
			assert(nodeIdxA >= 0 && nodeIdxA < _desc->Count && nodeIdxB >= 0 && nodeIdxB < _desc->Count);
#ifdef CUPAT_NAV_MESH
			auto a = _nodes[nodeIdxA].PCenter;
			auto b = _nodes[nodeIdxB].PCenter;
			return V2Float::DistSqr(a, b);
#else
			int aX = nodeIdxA % _desc->CellsCountX;
			int aY = nodeIdxA / _desc->CellsCountX;
			int bX = nodeIdxB % _desc->CellsCountX;
			int bY = nodeIdxB / _desc->CellsCountX;
			return (aX - bX) * (aX - bX) + (aY - bY) * (aY - bY);
#endif
		}

		__host__ __device__ int Count()
		{
			return _desc->Count;
		}

		__host__ __device__ static constexpr size_t EvalSize(const Desc& desc)
		{
			return sizeof(Desc) + sizeof(Node) * desc.Count;
		}

	private:
		Node* _nodes = nullptr;
		Desc* _desc = nullptr;


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
