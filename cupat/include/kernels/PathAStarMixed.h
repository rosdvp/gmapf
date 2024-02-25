#pragma once
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include "../Agent.h"
#include "../misc/Array.h"
#include "../misc/CumList.h"
#include "../misc/CumMatrix.h"
#include "../misc/CumQueue.h"
#include "../misc/Matrix.h"
#include "../misc/V2Int.h"

#define TIME_GET std::chrono::high_resolution_clock::now()
#define TIME_SET(x) (x) = TIME_GET
#define TIME_STAMP(x) auto (x) = TIME_GET
#define TIME_DIFF_MS(start) std::chrono::duration_cast<std::chrono::microseconds>(TIME_GET - (start)).count() / 1000.0f
#define TIME_COUNTER_ADD(start, counter) (counter) += std::chrono::duration_cast<std::chrono::microseconds>(TIME_GET - (start)).count()
#define TIME_COUNTER_GET(counter) ((counter) / 1000.0f)

namespace cupat
{
	struct FindPathAStarMixedInput
	{
		void* Map;
		void* DMap;
		void* DAgents;
		int AgentId;
	};

	struct __align__(8) AStarNodeMixed
	{
		V2Int Cell;
		V2Int PrevCell;
		float F;
		float G;
	};

	__global__ void KernelExpandNode(
		CumMatrix<int>* map,
		CumQueue<AStarNodeMixed>* queues,
		CumList<AStarNodeMixed>* results,
		V2Int targCell)
	{
		constexpr float heuristicK = 1.0f;
		V2Int neibsCellsDeltas[4] =
		{
			{-1, 0},
			{1, 0},
			{0, -1},
			{0, 1},
		};

		int tid = blockIdx.x * blockDim.x + threadIdx.x;

		if (queues[tid].Count() == 0)
			return;

		AStarNodeMixed curr = queues[tid].Pop();

		//printf("expand (%d, %d)\n", curr.Cell.X, curr.Cell.Y);

		for (auto& neibCellDelta : neibsCellsDeltas)
		{
			auto neibCell = curr.Cell + neibCellDelta;
			if (!map->IsValid(neibCell.X, neibCell.Y) || map->At(neibCell.X, neibCell.Y) != 0)
				continue;

			AStarNodeMixed neib;
			neib.Cell = neibCell;
			neib.PrevCell = curr.Cell;
			neib.G = curr.G + 1;
			neib.F = neib.G + V2Int::DistSqr(neibCell, targCell) * heuristicK;
			results[tid].Add(neib);
		}
	}

	__global__ void KernelFillFrontier(
		CumMatrix<AStarNodeMixed>* visited, 
		CumList<AStarNodeMixed>* frontier,
		CumList<AStarNodeMixed>* results,
		V2Int targCell,
		bool* isFound)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		auto& result = results[tid];

		for (int i = 0; i < result.Count(); i++)
		{
			auto& node = result.At(i);

			int idx = visited->GetIdx(node.Cell.X, node.Cell.Y);
			if (visited->TryOccupy(idx))
			{
				visited->At(idx) = node;
				frontier->AddAtomic(node);

				//printf("add to frontier (%d, %d)\n", node.Cell.X, node.Cell.Y);

				if (node.Cell == targCell)
					*isFound = true;
			}
			else
			{
				//printf("already in visited (%d, %d)\n", node.Cell.X, node.Cell.Y);
			}
		}
		result.RemoveAll();
	}

	__global__ void KernelFillQueues(CumList<AStarNodeMixed>* frontier, CumQueue<AStarNodeMixed>* queues)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		int step = gridDim.x * blockDim.x;
		for (int i = tid; i < frontier->Count(); i += step)
			queues[tid].Push(frontier->At(i));
	}

	__global__ void KernelClearFrontier(CumList<AStarNodeMixed>* frontier)
	{
		frontier->RemoveAll();
	}


	inline void FindPathAStarMixed(FindPathAStarMixedInput input)
	{
		constexpr float heuristicK = 1.0f;

		int threadsCount = 512;
		int threadsPerBlock = 128;
		int blocksCount = threadsCount / threadsPerBlock;

		Matrix<int> hMap;
		hMap.Attach(input.Map);
		Array<Agent> agents;
		agents.Attach(input.DAgents);
		Agent& agent = agents.At(input.AgentId);

		auto* map = CumMatrix<int>::New(1, hMap.GetCountX(), hMap.GetCountY());
		auto* visited = CumMatrix<AStarNodeMixed>::New(1, hMap.GetCountX(), hMap.GetCountY());
		auto* frontier = CumList<AStarNodeMixed>::New(1, 5000);
		auto* queues = CumQueue<AStarNodeMixed>::New(threadsCount, 100);
		auto* results = CumList<AStarNodeMixed>::New(threadsCount, 10);
		bool* isFound;
		cudaMallocManaged(&isFound, sizeof(bool));
		*isFound = false;

		for (int x = 0; x < hMap.GetCountX(); x++)
			for (int y = 0; y < hMap.GetCountY(); y++)
				map->At(x, y) = hMap.At({ x, y });

		AStarNodeMixed start;
		start.Cell = agent.TargCell;
		start.F = V2Int::DistSqr(agent.CurrCell, agent.TargCell) * heuristicK;
		start.G = 0;
		queues[0].Push(start);
		
		int* dummy;
		cudaMalloc(&dummy, sizeof(int));

		TIME_STAMP(tExpand);
		long long tExpandCounter = 0;
		TIME_STAMP(tFillFrontier);
		long long tFillFrontierCounter = 0;
		TIME_STAMP(tFillQueues);
		long long tFillQueuesCounter = 0;
		TIME_STAMP(tStart);

		int cyclesCount = 0;

		while (*isFound == false)
		{
			TIME_SET(tExpand);
			KernelExpandNode << <blocksCount, threadsPerBlock >> > (map, queues, results, agent.CurrCell);
			cudaDeviceSynchronize();
			TIME_COUNTER_ADD(tExpand, tExpandCounter);
			if (TryCatchCudaError("kernel expand node"))
				return;

			TIME_SET(tFillFrontier);
			KernelFillFrontier<<<blocksCount, threadsPerBlock >>>(visited, frontier, results, agent.CurrCell, isFound);
			cudaDeviceSynchronize();
			TIME_COUNTER_ADD(tFillFrontier, tFillFrontierCounter);
			if (TryCatchCudaError("kernel expand node"))
				return;

			//printf("frontier:\n");
			//for (int i = 0; i < frontier->Count(); i++)
			//	printf("(%d, %d)\n", frontier->At(i).Cell.X, frontier->At(i).Cell.Y);

			TIME_SET(tFillQueues);
			KernelFillQueues << <blocksCount, threadsPerBlock >> > (frontier, queues);
			cudaDeviceSynchronize();
			TIME_COUNTER_ADD(tFillQueues, tFillQueuesCounter);
			if (TryCatchCudaError("kernel fill queues"))
				return;

			KernelClearFrontier<<<1, 1>>>(frontier);
			cudaDeviceSynchronize();
			if (TryCatchCudaError("kernel clear frontier"))
				return;


			cyclesCount += 1;
		}
		caseEnd:;

		std::cout << "find time: " << TIME_DIFF_MS(tStart) << std::endl;
		std::cout << "expand time: " << TIME_COUNTER_GET(tExpandCounter)
			<< ", avg: " << TIME_COUNTER_GET(tExpandCounter) / cyclesCount << std::endl;
		std::cout << "fill frontier time: " << TIME_COUNTER_GET(tFillFrontierCounter)
			<< ", avg: " << TIME_COUNTER_GET(tFillFrontierCounter) / cyclesCount << std::endl;
		std::cout << "fill queues time: " << TIME_COUNTER_GET(tFillQueuesCounter)
			<< ", avg: " << TIME_COUNTER_GET(tFillQueuesCounter) / cyclesCount << std::endl;

		if (!isFound)
		{
			printf("failed to find path\n");
			return;
		}

		printf("path:\n");

		V2Int iter = agent.CurrCell;
		do
		{
			iter = visited->At(iter.X, iter.Y).PrevCell;
			printf("(%d, %d)\n", iter.X, iter.Y);
		} while (iter != agent.TargCell);

		printf("----------\n");
	}
}
