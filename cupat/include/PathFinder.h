#pragma once

#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Agent.h"
#include "Helpers.h"
#include "PathsStorage.h"
#include "misc/CuList.h"
#include "misc/Cum.h"
#include "misc/CuMatrix.h"
#include "misc/CuQueue.h"
#include "misc/V2Int.h"

namespace cupat
{
	struct __align__(8) PathRequest
	{
		int PathIdx;
		V2Int StartCell;
		V2Int TargetCell;
	};

	struct __align__(8) AStarNode
	{
		V2Int Cell;
		V2Int PrevCell;
		float F;
		float G;
	};

	__global__ void KernelMarkCollections(
		CuMatrix<int> map,
		PathsStorage pathsStorage,
		CuList<int> agentsIndices,
		CuList<PathRequest> requests,
		Cum<CuMatrix<AStarNode>> visiteds,
		Cum<CuList<AStarNode>> frontiers,
		int frontierCapacity,
		Cum<CuQueue<AStarNode>> queues,
		int queueCapacity)
	{
		int tid = threadIdx.x;
		int bid = blockIdx.x;
		int threadsPerAgent = blockDim.x;


		int parallelAgentsCount = gridDim.x;

		if (tid == 0)
		{
			visiteds.D(bid).Mark(map.CountX(), map.CountY());
			frontiers.D(bid).Mark(frontierCapacity);
		}

		if (tid == 0)
			frontiers.D(bid).Mark(frontierCapacity);

		queues.D(bid * threadsPerAgent + tid).Mark(queueCapacity);

		if (tid == 0 && bid == 0)
		{
			pathsStorage.RemoveAll();
			agentsIndices.Mark(parallelAgentsCount);
			requests.Mark(parallelAgentsCount);
		}
	}

	__global__ void KernelClearCollections(
		Cum<CuMatrix<AStarNode>> visiteds,
		Cum<CuQueue<AStarNode>> queues,
		CuList<AStarNode> frontier,
		CuList<int> agentsIndices,
		CuList<PathRequest> requests)
	{
		int tid = threadIdx.x;
		int bid = blockIdx.x;
		int threadsPerAgent = blockDim.x;

		queues.D(bid * threadsPerAgent + tid).RemoveAll();

		auto visited = visiteds.D(bid);
		for (int i = tid; i < visited.Count(); i++)
			visited.UnOccupy(i);

		if (tid == 0 && bid == 0)
		{
			frontier.RemoveAll();
			requests.RemoveAll();
			agentsIndices.RemoveAll();
		}
	}

	__global__ void KernelPrepareSearch(
		PathsStorage pathsStorage,
		CuList<Agent> agents,
		CuList<int> procAgentsIndices,
		CuList<PathRequest> requests)
	{
		int tid = blockDim.x * blockIdx.x + threadIdx.x;
		int threadsCount = gridDim.x * blockDim.x;

		for (int i = tid; i < agents.Count(); i += threadsCount)
		{
			Agent& agent = agents.At(i);
			if (!agent.IsNewPathRequested)
				continue;
			procAgentsIndices.AddAtomic(i);

			bool isPathRequested = false;
			agent.PathIdx = pathsStorage.TryUsePath(agent.CurrCell, agent.TargCell, isPathRequested);
			if (isPathRequested)
				requests.AddAtomic({agent.PathIdx, agent.CurrCell, agent.TargCell});
		}
	}

	__global__ void KernelFindPaths(
		float heuristicK,
		CuMatrix<int> map,
		CuList<PathRequest> requests,
		Cum<CuMatrix<AStarNode>> visiteds,
		Cum<CuList<AStarNode>> frontiers,
		Cum<CuQueue<AStarNode>> queues,
		bool* managedFoundFlags)
	{
		V2Int neibsCellsDeltas[4] =
		{
			{-1, 0},
			{1, 0},
			{0, -1},
			{0, 1},
		};

		int tid = threadIdx.x;
		int bid = blockIdx.x;
		int threadsCount = blockDim.x;

		auto visited = visiteds.D(bid);
		auto frontier = frontiers.D(bid);
		auto queue = queues.D(bid * threadsCount + tid);

		__shared__ bool sharedIsFound;
		sharedIsFound = false;

		PathRequest& request = requests.At(bid);
		V2Int startCell = request.TargetCell;
		V2Int targCell = request.StartCell;

		if (tid == 0)
		{
			AStarNode initialNode;
			initialNode.Cell = startCell;
			initialNode.F = V2Int::DistSqr(startCell, targCell) * heuristicK;
			initialNode.G = 0;
			queue.Push(initialNode);

			int visitedIdx = visited.GetIdx(initialNode.Cell.X, initialNode.Cell.Y);
			visited.TryOccupy(visitedIdx);
			visited.At(visitedIdx) = initialNode;

			managedFoundFlags[bid] = false;
		}

		while (!sharedIsFound)
		{
			if (queue.Count() > 0)
			{
				AStarNode curr = queue.Pop();

				for (auto& neibCellDelta : neibsCellsDeltas)
				{
					auto neibCell = curr.Cell + neibCellDelta;
					if (!map.IsValid(neibCell.X, neibCell.Y) || map.At(neibCell.X, neibCell.Y) != 0)
						continue;

					AStarNode neib;
					neib.Cell = neibCell;
					neib.PrevCell = curr.Cell;
					neib.G = curr.G + 1;
					neib.F = neib.G + V2Int::DistSqr(neibCell, targCell) * heuristicK;
					frontier.AddAtomic(neib);
				}
			}

			__syncthreads();

			for (int i = tid; i < frontier.Count(); i += threadsCount)
			{
				auto& node = frontier.At(i);
				int idx = visited.GetIdx(node.Cell.X, node.Cell.Y);
				if (visited.TryOccupy(idx))
				{
					visited.At(idx) = node;

					queue.Push(frontier.At(i));

					if (node.Cell == targCell)
					{
						sharedIsFound = true;
						managedFoundFlags[bid] = true;
					}
				}
			}

			__syncthreads();
			if (tid == 0)
				frontier.RemoveAll();
			__syncthreads();
		}
	}

	__global__ void KernelBuildPath(
		PathsStorage pathsStorage,
		CuList<PathRequest> requests,
		Cum<CuMatrix<AStarNode>> visiteds,
		bool* managedFoundFlags)
	{
		int tid = blockDim.x * blockIdx.x + threadIdx.x;
		if (tid >= requests.Count())
			return;

		if (managedFoundFlags[tid] == false)
		{
			//TODO set path usage to zero
			printf("path not found");
			return;
		}

		auto visited = visiteds.D(tid);

		PathRequest& request = requests.At(tid);

		int pathLength = 0;
		V2Int iter = request.StartCell;
		do
		{
			iter = visited.At(iter.X, iter.Y).PrevCell;
			pathLength += 1;
		} while (iter != request.TargetCell);

		void* pPath;
		cudaMalloc(&pPath, CuList<V2Int>::EvalSize(pathLength));
		CuList<V2Int> path(pPath);
		path.Mark(pathLength);

		iter = request.StartCell;
		do
		{
			iter = visited.At(iter.X, iter.Y).PrevCell;
			path.Add(iter);
		} while (iter != request.TargetCell);

		pathsStorage.SetUsingPath(request.PathIdx, pPath);
	}

	__global__ void KernelAttachPathsToAgents(
		PathsStorage pathsStorage,
		CuList<Agent> agents,
		CuList<int> agentsIndices
	)
	{
		int tid = blockDim.x * blockIdx.x + threadIdx.x;
		if (tid >= agentsIndices.Count())
			return;

		int agentIdx = agentsIndices.At(tid);
		Agent& agent = agents.At(agentIdx);

		agent.Path = pathsStorage.GetUsingPath(agent.PathIdx);
		agent.PathStepIdx = 0;
		agent.IsNewPathRequested = false;
	}


	__global__ void KernelCheckPaths(
		PathsStorage pathsStorage,
		CuList<Agent> agents)
	{
		for (int i = 0; i < agents.Count(); i++)
		{
			const Agent& agent = agents.At(i);

			int usersCount = pathsStorage.GetPathUsersCount(agent.PathIdx);

			printf("agent %d path (idx %d, users %d):\n", i, agent.PathIdx, usersCount);
			
			//void* p = pathsStorage.GetUsingPath(agent.PathIdx);
			//CuList<V2Int> path(p);
			//for (int step = 0; step < path.Count(); step++)
			//{
			//	const V2Int& cell = path.At(step);
			//	printf("(%d, %d)\n", cell.X, cell.Y);
			//}
			//printf("-------------\n");
		}
	}


	class PathFinder
	{
	public:
		void Init(
			const Cum<CuMatrix<int>>& map,
			const Cum<CuList<Agent>>& agents,
			int parallelAgentsCount,
			int threadsPerAgent,
			int frontierCapacity,
			int queueCapacity,
			float heuristicK)
		{
			_map = map;
			_agents = agents;

			_parallelAgentsCount = parallelAgentsCount;
			_threadsPerAgent = threadsPerAgent;
			_heuristicK = heuristicK;

			_pathsStorage.DAlloc(_agents.H(0).Count() * 2, 16);

			_procAgentsIndices.DAlloc(1, parallelAgentsCount);
			_requests.DAlloc(1, parallelAgentsCount);
				
			auto hMap = _map.H(0);
			_visiteds.DAlloc(parallelAgentsCount, hMap.CountX(), hMap.CountY());

			_frontiers.DAlloc(parallelAgentsCount, frontierCapacity);
			_queues.DAlloc(parallelAgentsCount * threadsPerAgent, queueCapacity);

			cudaMallocManaged(&_managedFoundFlags, sizeof(bool) * parallelAgentsCount);
			for (int i = 0; i < parallelAgentsCount; i++)
				_managedFoundFlags[i] = false;

			KernelMarkCollections<<<parallelAgentsCount, threadsPerAgent>>>(
				_map.D(0),
				_pathsStorage,
				_procAgentsIndices.D(0),
				_requests.D(0),
				_visiteds,
				_frontiers,
				frontierCapacity,
				_queues,
				queueCapacity
			);
			cudaDeviceSynchronize();
			if (TryCatchCudaError("path finder, mark collections"))
				return;

			KernelClearCollections<<<parallelAgentsCount, threadsPerAgent>>>(
				_visiteds,
				_queues,
				_frontiers.D(0),
				_procAgentsIndices.D(0),
				_requests.D(0)
			);
			cudaDeviceSynchronize();
			if (TryCatchCudaError("path finder, clear collections"))
				return;
		}

		~PathFinder()
		{
			_pathsStorage.DFree();

			_procAgentsIndices.DFree();
			_procAgentsIndices.HFree();

			_requests.DFree();
			_requests.HFree();

			_visiteds.DFree();

			_frontiers.DFree();
			_queues.DFree();
		}


		void Find()
		{
			TIME_STAMP(tPrepareSearch);
			PrepareSearch();
			auto durPrepareSearch = TIME_DIFF_MS(tPrepareSearch);

			_procAgentsIndices.CopyToHost();
			_requests.CopyToHost();

			int requestsCount = _requests.H(0).Count();

			TIME_STAMP(tFindPath);
			KernelFindPaths<<<requestsCount, _threadsPerAgent>>>(
				_heuristicK,
				_map.D(0),
				_requests.D(0),
				_visiteds,
				_frontiers,
				_queues,
				_managedFoundFlags
			);
			cudaDeviceSynchronize();
			if (TryCatchCudaError("path finder, find path"))
				return;
			auto durFindPath = TIME_DIFF_MS(tFindPath);

			TIME_STAMP(tBuildPaths);
			BuildPaths();
			auto durBuildPaths = TIME_DIFF_MS(tBuildPaths);

			TIME_STAMP(tAttachPaths);
			AttachPathsToAgents();
			auto durAttachPaths = TIME_DIFF_MS(tAttachPaths);

			KernelCheckPaths<<<1, 1>>>(_pathsStorage, _agents.D(0));
			cudaDeviceSynchronize();
			if (TryCatchCudaError("path finder, check paths"))
				return;

			std::cout << "prepare search ms: " << durPrepareSearch << std::endl;
			std::cout << "find paths ms: " << durFindPath << std::endl;
			std::cout << "build paths ms: " << durBuildPaths << std::endl;
			std::cout << "attach paths ms: " << durAttachPaths << std::endl;
		}



	private:
		int _parallelAgentsCount = 0;
		int _threadsPerAgent = 0;
		float _heuristicK = 1.0f;

		Cum<CuMatrix<int>> _map;
		Cum<CuList<Agent>> _agents;

		PathsStorage _pathsStorage;

		Cum<CuList<int>> _procAgentsIndices;
		Cum<CuList<PathRequest>> _requests;

		Cum<CuMatrix<AStarNode>> _visiteds;
		Cum<CuList<AStarNode>> _frontiers;
		Cum<CuQueue<AStarNode>> _queues;
		bool* _managedFoundFlags = nullptr;


		void PrepareSearch()
		{
			int threadsCount = 512;
			int threadsPerBlock = 128;
			int blocksCount = threadsCount / threadsPerBlock;

			KernelPrepareSearch<<<blocksCount, threadsPerBlock>>>(
				_pathsStorage,
				_agents.D(0),
				_procAgentsIndices.D(0),
				_requests.D(0)
			);
			cudaDeviceSynchronize();

			TryCatchCudaError("path finder, prepare search");
		}

		void BuildPaths()
		{
			int threadsCount = _requests.H(0).Count();
			int threadsPerBlock = 32;
			int blocksCount = threadsCount / threadsPerBlock;
			if (blocksCount * threadsPerBlock < threadsCount)
				blocksCount += 1;

			KernelBuildPath<<<blocksCount, threadsCount>>>(
				_pathsStorage,
				_requests.D(0),
				_visiteds,
				_managedFoundFlags
			);
			cudaDeviceSynchronize();
			TryCatchCudaError("path finder, build paths");
		}

		void AttachPathsToAgents()
		{
			int threadsCount = _procAgentsIndices.H(0).Count();
			int threadsPerBlock = 32;
			int blocksCount = threadsCount / threadsPerBlock;
			if (blocksCount * threadsPerBlock < threadsCount)
				blocksCount += 1;

			KernelAttachPathsToAgents<<<blocksCount, threadsPerBlock>>>(
				_pathsStorage,
				_agents.D(0),
				_procAgentsIndices.D(0)
			);
			cudaDeviceSynchronize();
			TryCatchCudaError("path finder, attach paths to agents");
		}
	};
}
