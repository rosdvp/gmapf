#pragma once
#include "misc/CuList.h"
#include "misc/Cum.h"
#include "misc/CuNodesMap.h"

namespace cupat
{
	__global__ static void KernelPreAlloc(
		CuList<void*> paths,
		int count,
		int eachCapacity)
	{
		paths.Mark(count);

		for (int i = 0; i < count; i++)
		{
			void* ptr;
			cudaMalloc(&ptr, CuList<int>::EvalSize(eachCapacity));
			CuList<int> path(ptr);
			path.Mark(eachCapacity);

			paths.Add(ptr);
		}
	}

	class PathAllocator
	{
	public:
		void Init(int agentsCount, int nodesCount)
		{
			_halfPathCapacity = nodesCount / 2;
			_halfPaths.DAlloc(1, agentsCount);
			KernelPreAlloc<<<1,1>>>(_halfPaths.D(0), agentsCount, _halfPathCapacity);

			_fullPathCapacity = nodesCount;
			_fullPaths.DAlloc(1, agentsCount);
			KernelPreAlloc<<<1, 1>>>(_fullPaths.D(0), agentsCount, _fullPathCapacity);
		}


		__device__ void* GetPath(int capacity)
		{
			void* result = nullptr;
			if (capacity <= _halfPathCapacity && _halfPaths.D(0).TryPopLastAtomic(result) ||
				capacity <= _fullPathCapacity && _fullPaths.D(0).TryPopLastAtomic(result))
			{
				return result;
			}
			cudaMalloc(&result, CuList<int>::EvalSize(capacity));
			CuList<int> path(result);
			path.Mark(capacity);
			//printf("[path allocator] new dynamic allocation\n");
			return result;
		}

	private:
		int _halfPathCapacity = 0;
		Cum<CuList<void*>> _halfPaths;

		int _fullPathCapacity = 0;
		Cum<CuList<void*>> _fullPaths;
	};
}
