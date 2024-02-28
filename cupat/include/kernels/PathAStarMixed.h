#pragma once
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include "../Agent.h"
#include "../misc/CuQueue.h"
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
		Cum<CuMatrix<int>> Map;
		Cum<CuList<Agent>> Agents;
		int AgentId;
	};

	struct __align__(8) AStarNodeMixed
	{
		V2Int Cell;
		V2Int PrevCell;
		float F;
		float G;
	};

	__global__ void KernelSetup(
		CuList<AStarNodeMixed> frontier,
		int frontierCapacity,
		Cum<CuQueue<AStarNodeMixed>> queues,
		int queueCapacity,
		AStarNodeMixed initialNode)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;

		queues.D(tid).Mark(queueCapacity);

		if (tid == 0)
		{
			frontier.Mark(frontierCapacity);
			queues.D(tid).Push(initialNode);
		}
	}


	__global__ void KernelExpandNode(
		CuMatrix<int> map,
		Cum<CuQueue<AStarNodeMixed>> queues,
		CuList<AStarNodeMixed> frontier,
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

		auto queue = queues.D(tid);
		if (queue.Count() == 0)
			return;

		AStarNodeMixed curr = queue.Pop();

		//printf("expand (%d, %d)\n", curr.Cell.X, curr.Cell.Y);

		for (auto& neibCellDelta : neibsCellsDeltas)
		{
			auto neibCell = curr.Cell + neibCellDelta;
			if (!map.IsValid(neibCell.X, neibCell.Y) || map.At(neibCell.X, neibCell.Y) != 0)
				continue;

			AStarNodeMixed neib;
			neib.Cell = neibCell;
			neib.PrevCell = curr.Cell;
			neib.G = curr.G + 1;
			neib.F = neib.G + V2Int::DistSqr(neibCell, targCell) * heuristicK;
			frontier.AddAtomic(neib);
		}
	}

	__global__ void KernelFillFrontier(
		CuMatrix<AStarNodeMixed> visited,
		CuList<AStarNodeMixed> frontier,
		Cum<CuList<AStarNodeMixed>> results,
		V2Int targCell,
		bool* isFound)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		auto result = results.D(tid);

		for (int i = 0; i < result.Count(); i++)
		{
			auto& node = result.At(i);

			int idx = visited.GetIdx(node.Cell.X, node.Cell.Y);
			if (visited.TryOccupy(idx))
			{
				visited.At(idx) = node;
				frontier.AddAtomic(node);

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

	__global__ void KernelFillQueues(
		CuMatrix<AStarNodeMixed> visited,
		CuList<AStarNodeMixed> frontier,
		Cum<CuQueue<AStarNodeMixed>> queues,
		V2Int targCell,
		bool* isFound)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		auto queue = queues.D(tid);

		int step = gridDim.x * blockDim.x;
		for (int i = tid; i < frontier.Count(); i += step)
		{
			auto& node = frontier.At(i);
			int idx = visited.GetIdx(node.Cell.X, node.Cell.Y);
			if (visited.TryOccupy(idx))
			{
				visited.At(idx) = node;

				queue.Push(frontier.At(i));

				if (node.Cell == targCell)
					*isFound = true;
			}
		}
	}

	__global__ void KernelClearFrontier(CuList<AStarNodeMixed> frontier)
	{
		frontier.RemoveAll();
	}


	inline void FindPathAStarMixed(FindPathAStarMixedInput input)
	{
		constexpr float heuristicK = 1.0f;

		int threadsCount = 512;
		int threadsPerBlock = 128;
		int blocksCount = threadsCount / threadsPerBlock;

		int frontierCapacity = 5000;
		int queueCapacity = 100;


		Agent& agent = input.Agents.H(0).At(input.AgentId);

		auto hMap = input.Map.H(0);
		auto dMap = input.Map.D(0);

		Cum<CuMatrix<AStarNodeMixed>> cumVisited;
		cumVisited.HAllocAndMark(1, hMap.CountX(), hMap.CountY());
		cumVisited.CopyToDevice();
		auto dVisited = cumVisited.D(0);

		Cum<CuList<AStarNodeMixed>> cumFrontier;
		cumFrontier.DAlloc(1, frontierCapacity);
		auto dFrontier = cumFrontier.D(0);

		Cum<CuQueue<AStarNodeMixed>> queues;
		queues.DAlloc(threadsCount, queueCapacity);

		//Cum<CuList<AStarNodeMixed>> results;
		//results.DAlloc(threadsCount, resultCapacity);

		bool* isFound;
		cudaMallocManaged(&isFound, sizeof(bool));
		*isFound = false;


		AStarNodeMixed initialNode;
		initialNode.Cell = agent.TargCell;
		initialNode.F = V2Int::DistSqr(agent.CurrCell, agent.TargCell) * heuristicK;
		initialNode.G = 0;

		KernelSetup<<<blocksCount, threadsPerBlock>>>(
			dFrontier,
			frontierCapacity,
			queues,
			queueCapacity,
			initialNode
		);
		cudaDeviceSynchronize();
		if (TryCatchCudaError("kernel setup"))
			return;


		TIME_STAMP(tExpand);
		long long tExpandCounter = 0;
		TIME_STAMP(tFillQueues);
		long long tFillQueuesCounter = 0;
		TIME_STAMP(tStart);

		int cyclesCount = 0;

		while (*isFound == false)
		{
			TIME_SET(tExpand);
			KernelExpandNode<<<blocksCount, threadsPerBlock>>>(
				dMap,
				queues,
				dFrontier,
				agent.CurrCell
			);
			cudaDeviceSynchronize();
			TIME_COUNTER_ADD(tExpand, tExpandCounter);
			if (TryCatchCudaError("kernel expand node"))
				return;

			TIME_SET(tFillQueues);
			KernelFillQueues<<<blocksCount, threadsPerBlock>>>(
				dVisited,
				dFrontier, 
				queues,
				agent.CurrCell,
				isFound
			);
			cudaDeviceSynchronize();
			TIME_COUNTER_ADD(tFillQueues, tFillQueuesCounter);
			if (TryCatchCudaError("kernel fill queues"))
				return;

			KernelClearFrontier<<<1, 1>>>(dFrontier);
			cudaDeviceSynchronize();
			if (TryCatchCudaError("kernel clear frontier"))
				return;

			cyclesCount += 1;
		}
		caseEnd:;

		std::cout << "find time: " << TIME_DIFF_MS(tStart) << std::endl;
		std::cout << "expand time: " << TIME_COUNTER_GET(tExpandCounter)
			<< ", avg: " << TIME_COUNTER_GET(tExpandCounter) / cyclesCount << std::endl;
		std::cout << "fill queues time: " << TIME_COUNTER_GET(tFillQueuesCounter)
			<< ", avg: " << TIME_COUNTER_GET(tFillQueuesCounter) / cyclesCount << std::endl;

		if (isFound)
		{
			cumVisited.CopyToHost();
			auto hVisited = cumVisited.H(0);

			printf("path:\n");

			V2Int iter = agent.CurrCell;
			do
			{
				iter = hVisited.At(iter.X, iter.Y).PrevCell;
				printf("(%d, %d)\n", iter.X, iter.Y);
			} while (iter != agent.TargCell);

			printf("----------\n");
		}
		else
		{
			printf("failed to find path\n");
		}

		cumVisited.HFree();
		cumVisited.DFree();
		cumFrontier.DFree();
		queues.DFree();
		cudaFree(isFound);
	}
}
