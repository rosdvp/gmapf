#pragma once
#include <cassert>
#include <cstdio>
#include <cuda_runtime_api.h>

namespace cupat
{
	class CuNodesMap
	{
	public:
		constexpr static int INVALID = -1;
		constexpr static int NEIBS_MAX_COUNT = 8;

		struct __align__(4) Desc
		{
			float CellSize;
			int CellsCountX;
			int CellsCountY;
			int Count;
		};
	private:
		struct __align__(4) Node
		{
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

		__host__ __device__ bool TryGetClosest(const V2Float& pos, int* outNodeIdx)
		{
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
		}

		__host__ __device__ V2Float GetPos(int nodeIdx)
		{
			assert(nodeIdx >= 0 && nodeIdx < _desc->Count);
			float x = (nodeIdx % _desc->CellsCountX) * _desc->CellSize + _desc->CellSize / 2.0f;
			float y = (nodeIdx / _desc->CellsCountX) * _desc->CellSize + _desc->CellSize / 2.0f;
			return { x , y };
		}

		__host__ __device__ float GetDistSqr(int nodeIdxA, int nodeIdxB)
		{
			int aX = nodeIdxA % _desc->CellsCountX;
			int aY = nodeIdxA / _desc->CellsCountX;
			int bX = nodeIdxB % _desc->CellsCountX;
			int bY = nodeIdxB / _desc->CellsCountX;
			return (aX - bX) * (aX - bX) + (aY - bY) * (aY - bY);
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
	};
}
