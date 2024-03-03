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
		CuList<PathRequest> requests,
		bool* dFoundFlags)
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

		if (tid == 0)
			dFoundFlags[bid] = false;
	}

	__global__ void KernelPrepareSearch(
		PathsStorage pathsStorage,
		CuList<Agent> agents,
		CuList<int> procAgentsIndices,
		CuList<PathRequest> requests,
		int* procAgentsCount,
		int* requestsCount)
	{
		int tid = blockDim.x * blockIdx.x + threadIdx.x;
		int threadsCount = gridDim.x * blockDim.x;

		for (int i = tid; i < agents.Count(); i += threadsCount)
		{
			Agent& agent = agents.At(i);
			if (!agent.IsNewPathRequested)
				continue;
			procAgentsIndices.AddAtomic(i);
			atomicAdd(procAgentsCount, 1);

			bool isPathRequested = false;
			agent.PathIdx = pathsStorage.TryUsePath(agent.CurrCell, agent.TargCell, isPathRequested);
			if (isPathRequested)
			{
				PathRequest req;
				req.PathIdx = agent.PathIdx;
				req.StartCell = agent.CurrCell;
				req.TargetCell = agent.TargCell;
				requests.AddAtomic(req);
				atomicAdd(requestsCount, 1);
			}
		}
	}

	__global__ void KernelFindPaths(
		float heuristicK,
		CuMatrix<int> map,
		CuList<PathRequest> requests,
		Cum<CuMatrix<AStarNode>> visiteds,
		Cum<CuList<AStarNode>> frontiers,
		Cum<CuQueue<AStarNode>> queues,
		bool* dFoundFlags)
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

			dFoundFlags[bid] = false;
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
						dFoundFlags[bid] = true;
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
		bool* foundFlags)
	{
		int tid = blockDim.x * blockIdx.x + threadIdx.x;
		if (tid >= requests.Count())
			return;

		if (foundFlags[tid] == false)
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

			_hProcAgentsCount = static_cast<int*>(malloc(sizeof(int)));
			_hRequestsCount = static_cast<int*>(malloc(sizeof(int)));

			cudaMalloc(&_dFoundFlags, sizeof(bool) * parallelAgentsCount);
			cudaMalloc(&_dProcAgentsCount, sizeof(int));
			cudaMalloc(&_dRequestsCount, sizeof(int));

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

			cudaStreamCreate(&_stream);
			cudaEventCreate(&_evClearCollectionsStart);
			cudaEventCreate(&_evClearCollectionsEnd);
			cudaEventCreate(&_evPrepareSearchStart);
			cudaEventCreate(&_evPrepareSearchEnd);
			cudaEventCreate(&_evFindStart);
			cudaEventCreate(&_evFindEnd);
			cudaEventCreate(&_evBuildPathsStart);
			cudaEventCreate(&_evBuildPathsEnd);
			cudaEventCreate(&_evAttachPathsStart);
			cudaEventCreate(&_evAttachPathsEnd);
		}

		~PathFinder()
		{
			_pathsStorage.DFree();
			_procAgentsIndices.DFree();
			_requests.DFree();
			_visiteds.DFree();
			_frontiers.DFree();
			_queues.DFree();

			cudaFree(_dFoundFlags);
			cudaFree(_dProcAgentsCount);
			cudaFree(_dRequestsCount);

			free(_hProcAgentsCount);
			free(_hRequestsCount);
		}


		void Find()
		{
			ClearCollections();
			PrepareSearch();

			cudaStreamSynchronize(_stream);

			int requestsCount = *_hRequestsCount;

			cudaEventRecord(_evFindStart, _stream);
			KernelFindPaths<<<requestsCount, _threadsPerAgent, 0, _stream>>>(
				_heuristicK,
				_map.D(0),
				_requests.D(0),
				_visiteds,
				_frontiers,
				_queues,
				_dFoundFlags
			);
			cudaEventRecord(_evFindEnd, _stream);

			BuildPaths();
			AttachPathsToAgents();

			KernelCheckPaths<<<1, 1, 0, _stream>>>(_pathsStorage, _agents.D(0));

			cudaStreamSynchronize(_stream);

			float durClearCollections, durPrepareSearch, durFind, durBuildPaths, durAttachPaths;

			cudaEventElapsedTime(&durClearCollections, _evClearCollectionsStart, _evClearCollectionsEnd);
			cudaEventElapsedTime(&durPrepareSearch, _evPrepareSearchStart, _evPrepareSearchEnd);
			cudaEventElapsedTime(&durFind, _evFindStart, _evFindEnd);
			cudaEventElapsedTime(&durBuildPaths, _evBuildPathsStart, _evBuildPathsEnd);
			cudaEventElapsedTime(&durAttachPaths, _evAttachPathsStart, _evAttachPathsEnd);

			std::cout << "clear collections ms: " << durClearCollections << std::endl;
			std::cout << "prepare search ms: " << durPrepareSearch << std::endl;
			std::cout << "find paths ms: " << durFind << std::endl;
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
		bool* _dFoundFlags = nullptr;
		int* _hProcAgentsCount = nullptr;
		int* _dProcAgentsCount = nullptr;
		int* _hRequestsCount = nullptr;
		int* _dRequestsCount = nullptr;

		cudaStream_t _stream;
		cudaEvent_t _evClearCollectionsStart;
		cudaEvent_t _evClearCollectionsEnd;
		cudaEvent_t _evPrepareSearchStart;
		cudaEvent_t _evPrepareSearchEnd;
		cudaEvent_t _evFindStart;
		cudaEvent_t _evFindEnd;
		cudaEvent_t _evBuildPathsStart;
		cudaEvent_t _evBuildPathsEnd;
		cudaEvent_t _evAttachPathsStart;
		cudaEvent_t _evAttachPathsEnd;


		void ClearCollections()
		{
			cudaEventRecord(_evClearCollectionsStart, _stream);

			*_hProcAgentsCount = 0;
			cudaMemcpyAsync(_dProcAgentsCount, _hProcAgentsCount, sizeof(int), cudaMemcpyHostToDevice, _stream);
			*_hRequestsCount = 0;
			cudaMemcpyAsync(_dRequestsCount, _hRequestsCount, sizeof(int), cudaMemcpyHostToDevice, _stream);

			KernelClearCollections<<<_parallelAgentsCount, _threadsPerAgent, 0, _stream>>>(
				_visiteds,
				_queues,
				_frontiers.D(0),
				_procAgentsIndices.D(0),
				_requests.D(0),
				_dFoundFlags
			);

			cudaEventRecord(_evClearCollectionsEnd);
		}


		void PrepareSearch()
		{
			cudaEventRecord(_evPrepareSearchStart, _stream);

			int threadsCount = 512;
			int threadsPerBlock = 128;
			int blocksCount = threadsCount / threadsPerBlock;

			KernelPrepareSearch<<<blocksCount, threadsPerBlock, 0, _stream>>>(
				_pathsStorage,
				_agents.D(0),
				_procAgentsIndices.D(0),
				_requests.D(0),
				_dProcAgentsCount,
				_dRequestsCount
			);

			cudaMemcpyAsync(_hProcAgentsCount, _dProcAgentsCount, sizeof(int), cudaMemcpyDeviceToHost, _stream);
			cudaMemcpyAsync(_hRequestsCount, _dRequestsCount, sizeof(int), cudaMemcpyDeviceToHost, _stream);

			cudaEventRecord(_evPrepareSearchEnd, _stream);
		}

		void BuildPaths()
		{
			cudaEventRecord(_evBuildPathsStart, _stream);

			int threadsCount = *_hRequestsCount;
			int threadsPerBlock = 32;
			int blocksCount = threadsCount / threadsPerBlock;
			if (blocksCount * threadsPerBlock < threadsCount)
				blocksCount += 1;

			KernelBuildPath<<<blocksCount, threadsCount, 0, _stream>>>(
				_pathsStorage,
				_requests.D(0),
				_visiteds,
				_dFoundFlags
			);

			cudaEventRecord(_evBuildPathsEnd);
		}

		void AttachPathsToAgents()
		{
			cudaEventRecord(_evAttachPathsStart, _stream);

			int threadsCount = *_hProcAgentsCount;
			int threadsPerBlock = 32;
			int blocksCount = threadsCount / threadsPerBlock;
			if (blocksCount * threadsPerBlock < threadsCount)
				blocksCount += 1;

			KernelAttachPathsToAgents<<<blocksCount, threadsPerBlock, 0, _stream>>>(
				_pathsStorage,
				_agents.D(0),
				_procAgentsIndices.D(0)
			);

			cudaEventRecord(_evAttachPathsEnd);
		}
	};
}