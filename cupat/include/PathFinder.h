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
		int AgentIdx;
		V2Int StartCell;
		V2Int TargetCell;
		V2Int MiddleCellFromStart;
		V2Int MiddleCellFromTarget;
	};

	struct __align__(8) AStarNode
	{
		V2Int Cell;
		V2Int PrevCell;
		float F;
		float G;
		int FromStart;
	};

	__global__ void KernelMarkCollections(
		CuMatrix<int> map,
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
			requests.Mark(parallelAgentsCount);
	}

	__global__ void KernelClearRequests(
		CuList<PathRequest> requests,
		int* requestsCount)
	{
		requests.RemoveAll();
		*requestsCount = 0;
	}

	__global__ void KernelClearCollections(
		Cum<CuMatrix<AStarNode>> visiteds,
		Cum<CuList<AStarNode>> frontiers,
		bool* foundFlags)
	{
		int bid = blockIdx.x;
		int tid = threadIdx.x;
		int threadsCount = blockDim.x;

		auto visited = visiteds.D(bid);
		for (int i = tid; i < visited.Count(); i += threadsCount)
		{
			visited.UnOccupy(i);
			visited.At(i).FromStart = -1;
		}

		if (tid == 0)
		{
			frontiers.D(bid).RemoveAll();
			foundFlags[bid] = false;
		}
	}

	__global__ void KernelPrepareSearch(
		CuList<Agent> agents,
		CuList<PathRequest> requests,
		int* requestsCount)
	{
		int tid = blockDim.x * blockIdx.x + threadIdx.x;
		int threadsCount = gridDim.x * blockDim.x;

		for (int i = tid; i < agents.Count(); i += threadsCount)
		{
			Agent& agent = agents.At(i);
			if (!agent.IsNewPathRequested)
				continue;

			if (agent.CurrCell == agent.TargCell)
			{
				agent.IsTargetReached = true;
				agent.IsNewPathRequested = false;
				continue;
			}

			int oldCount = atomicAdd(requestsCount, 1);
			if (oldCount >= requests.Capacity())
			{
				*requestsCount = requests.Capacity();
				return;
			}
			PathRequest req;
			req.AgentIdx = i;
			req.StartCell = agent.CurrCell;
			req.TargetCell = agent.TargCell;
			requests.AddAtomic(req);
		}
	}

	__global__ void KernelSearch(
		float heuristicK,
		CuMatrix<int> map,
		CuList<PathRequest> requests,
		Cum<CuMatrix<AStarNode>> visiteds,
		Cum<CuList<AStarNode>> frontiers,
		Cum<CuQueue<AStarNode>> queues,
		bool* dFoundFlags)
	{
		V2Int neibsCellsDeltas[8] =
		{
			{-1, 0},
			{1, 0},
			{0, -1},
			{0, 1},
			{1, 1},
			{1, -1},
			{-1, -1},
			{-1, 1}
		};

		int tid = threadIdx.x;
		int bid = blockIdx.x;
		int threadsCount = blockDim.x;

		auto visited = visiteds.D(bid);
		auto frontier = frontiers.D(bid);
		auto queue = queues.D(bid * threadsCount + tid);
		queue.RemoveAll();

		__shared__ int sharedIsFound;
		sharedIsFound = 0;

		//printf("bid %d count %d\n", bid, requests.Count());
		PathRequest& request = requests.At(bid);
		V2Int startCell = request.StartCell;
		V2Int targCell = request.TargetCell;

		if (tid == 0)
		{
			AStarNode startNode;
			startNode.Cell = startCell;
			startNode.F = V2Int::DistSqr(startCell, targCell) * heuristicK;
			startNode.G = 0;
			startNode.FromStart = 1;
			queue.Push(startNode);

			int visitedIdx = visited.GetIdx(startNode.Cell.X, startNode.Cell.Y);
			visited.TryOccupy(visitedIdx);
			visited.At(visitedIdx) = startNode;
		}
		if (tid == 1)
		{
			AStarNode targNode;
			targNode.Cell = targCell;
			targNode.F = V2Int::DistSqr(startCell, targCell) * heuristicK;
			targNode.G = 0;
			targNode.FromStart = 0;
			queue.Push(targNode);

			int visitedIdx = visited.GetIdx(targNode.Cell.X, targNode.Cell.Y);
			visited.TryOccupy(visitedIdx);
			visited.At(visitedIdx) = targNode;
		}

		while (sharedIsFound == 0)
		{
			if (queue.Count() > 0)
			{
				AStarNode curr = queue.Pop();
				//printf("pop (%d, %d) %s\n", curr.Cell.X, curr.Cell.Y, curr.FromStart ? "from start" : "from target");

				for (auto& neibCellDelta : neibsCellsDeltas)
				{
					auto neibCell = curr.Cell + neibCellDelta;
					if (!map.IsValid(neibCell.X, neibCell.Y) || map.At(neibCell.X, neibCell.Y) != 0)
						continue;

					float h = curr.FromStart
						? V2Int::DistSqr(neibCell, targCell)
						: V2Int::DistSqr(neibCell, startCell);

					AStarNode neib;
					neib.Cell = neibCell;
					neib.PrevCell = curr.Cell;
					neib.G = curr.G + 1;
					neib.F = neib.G + h * heuristicK;
					neib.FromStart = curr.FromStart;
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
				}
				else if (visited.At(idx).FromStart != -1 && node.FromStart != visited.At(idx).FromStart)
				{
					if (atomicExch(&sharedIsFound, 1) == 0)
					{
						request.MiddleCellFromStart = node.FromStart == 1 ? node.PrevCell : visited.At(idx).Cell;
						request.MiddleCellFromTarget = node.FromStart == 1 ? visited.At(idx).Cell : node.PrevCell;
						sharedIsFound = 1;
						dFoundFlags[bid] = true;
					}
				}
			}

			int isAnyQueueNotEmpty = __syncthreads_or(queue.Count());
			if (isAnyQueueNotEmpty == 0)
				return;
			if (tid == 0)
				frontier.RemoveAll();
			__syncthreads();
		}
	}

	__global__ void KernelBuildPath(
		CuList<Agent> agents,
		CuList<PathRequest> requests,
		Cum<CuMatrix<AStarNode>> visiteds,
		bool* foundFlags)
	{
		int tid = blockDim.x * blockIdx.x + threadIdx.x;
		if (tid >= requests.Count())
			return;

		PathRequest& request = requests.At(tid);
		Agent& agent = agents.At(request.AgentIdx);

		//printf("middle cells: from start (%d, %d), from target (%d, %d)\n",
		//	request.MiddleCellFromStart.X, request.MiddleCellFromStart.Y,
		//	request.MiddleCellFromTarget.X, request.MiddleCellFromTarget.Y);

		if (foundFlags[tid] == false)
		{
			agent.IsNewPathRequested = false;
			agent.IsTargetReached = true;

			printf("path not found\n");
			return;
		}

		auto visited = visiteds.D(tid);


		int pathLength = 0;
		V2Int iter = request.MiddleCellFromStart;
		while (iter != request.StartCell)
		{
			pathLength += 1;
			iter = visited.At(iter.X, iter.Y).PrevCell;
		}

		iter = request.MiddleCellFromTarget;
		while (iter != request.TargetCell)
		{
			pathLength += 1;
			iter = visited.At(iter.X, iter.Y).PrevCell;
		}
		pathLength += 1; // to add target cell

		void* pPath;
		cudaMalloc(&pPath, CuList<V2Int>::EvalSize(pathLength));
		CuList<V2Int> path(pPath);
		path.Mark(pathLength);

		iter = request.MiddleCellFromStart;
		while (iter != request.StartCell)
		{
			path.Add(iter);
			iter = visited.At(iter.X, iter.Y).PrevCell;
		}

		path.Reverse();

		iter = request.MiddleCellFromTarget;
		while (iter != request.TargetCell)
		{
			path.Add(iter);
			iter = visited.At(iter.X, iter.Y).PrevCell;
		}
		path.Add(request.TargetCell);

		agent.IsNewPathRequested = false;
		agent.IsTargetReached = false;
		agent.Path = pPath;
		agent.PathNextCell = path.At(0);
	}


	__global__ void KernelCheckPaths(
		CuList<Agent> agents)
	{
		for (int agentIdx = 0; agentIdx < agents.Count(); agentIdx++)
		{
			auto& agent = agents.At(agentIdx);
			if (agent.Path == nullptr)
				continue;

			printf("agent %d path:\n", agentIdx);

			CuList<V2Int> path(agent.Path);
			for (int i = 0; i < path.Count(); i++)
			{
				auto& cell = path.At(i);
				printf("(%d, %d)\n", cell.X, cell.Y);
			}

			printf("-------------------------\n");
		}
	}


	class PathFinder
	{
	public:
		bool DebugSyncMode = false;

		float DebugDurClearCollections = 0;
		float DebugDurClearCollectionsMax = 0;
		float DebugDurPrepareSearch = 0;
		float DebugDurPrepareSearchMax = 0;
		float DebugDurSearch = 0;
		float DebugDurSearchMax = 0;
		float DebugDurBuildPaths = 0;
		float DebugDurBuildPathsMax = 0;
		float DebugDurAttachPaths = 0;
		float DebugDurAttachPathsMax = 0;
		int DebugRecordsCount = 0;


		void Init(
			const Cum<CuMatrix<int>>& map,
			const Cum<CuList<Agent>>& agents,
			int parallelAgentsCount,
			int threadsPerAgent,
			int queueCapacity,
			float heuristicK,
			int pathStorageCapacityK)
		{
			_map = map;
			_agents = agents;

			_threadsPerAgent = threadsPerAgent;
			_heuristicK = heuristicK;

			int frontierCapacity = (threadsPerAgent+1) * 8;

			_requests.DAlloc(1, parallelAgentsCount);
				
			auto hMap = _map.H(0);
			_visiteds.DAlloc(parallelAgentsCount, hMap.CountX(), hMap.CountY());

			_frontiers.DAlloc(parallelAgentsCount, frontierCapacity);
			_queues.DAlloc(parallelAgentsCount * threadsPerAgent, queueCapacity);

			_hRequestsCount = static_cast<int*>(malloc(sizeof(int)));

			cudaMalloc(&_dFoundFlags, sizeof(bool) * parallelAgentsCount);
			cudaMalloc(&_dRequestsCount, sizeof(int));

			KernelMarkCollections<<<parallelAgentsCount, threadsPerAgent>>>(
				_map.D(0),
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
			cudaEventCreate(&_evSearchStart);
			cudaEventCreate(&_evSearchEnd);
			cudaEventCreate(&_evBuildPathsStart);
			cudaEventCreate(&_evBuildPathsEnd);
		}

		~PathFinder()
		{
			cudaStreamDestroy(_stream);
			cudaEventDestroy(_evClearCollectionsStart);
			cudaEventDestroy(_evClearCollectionsEnd);
			cudaEventDestroy(_evPrepareSearchStart);
			cudaEventDestroy(_evPrepareSearchEnd);
			cudaEventDestroy(_evSearchStart);
			cudaEventDestroy(_evSearchEnd);
			cudaEventDestroy(_evBuildPathsStart);
			cudaEventDestroy(_evBuildPathsEnd);

			_requests.DFree();
			_visiteds.DFree();
			_frontiers.DFree();
			_queues.DFree();

			cudaFree(_dFoundFlags);
			cudaFree(_dRequestsCount);

			free(_hRequestsCount);

			CudaCatch();
		}

		void AsyncPreFind()
		{
			PrepareSearch();
			if (DebugSyncMode)
				CudaSyncAndCatch();
		}

		void AsyncFind()
		{
			if (*_hRequestsCount == 0)
				return;
			ClearCollections();
			if (DebugSyncMode)
				CudaSyncAndCatch();
			Search();
			if (DebugSyncMode)
				CudaSyncAndCatch();
			BuildPaths();
			if (DebugSyncMode)
				CudaSyncAndCatch();
		}

		void PostFind()
		{
			if (*_hRequestsCount == 0)
				return;

			//KernelCheckPaths<<<1, 1, 0, _stream>>>(
			//	_agents.D(0)
			//);
			//CudaCheck(cudaStreamSynchronize(_stream), "path finder, check paths");

			DebugRecordDurs();
		}

		void Sync()
		{
			CudaCheck(cudaStreamSynchronize(_stream), "path finder");
		}


	private:
		int _threadsPerAgent = 0;
		float _heuristicK = 1.0f;

		Cum<CuMatrix<int>> _map;
		Cum<CuList<Agent>> _agents;

		Cum<CuList<PathRequest>> _requests;

		Cum<CuMatrix<AStarNode>> _visiteds;
		Cum<CuList<AStarNode>> _frontiers;
		Cum<CuQueue<AStarNode>> _queues;
		bool* _dFoundFlags = nullptr;
		int* _hRequestsCount = nullptr;
		int* _dRequestsCount = nullptr;

		cudaStream_t _stream{};
		cudaEvent_t _evClearCollectionsStart{};
		cudaEvent_t _evClearCollectionsEnd{};
		cudaEvent_t _evPrepareSearchStart{};
		cudaEvent_t _evPrepareSearchEnd{};
		cudaEvent_t _evSearchStart{};
		cudaEvent_t _evSearchEnd{};
		cudaEvent_t _evBuildPathsStart{};
		cudaEvent_t _evBuildPathsEnd{};


		void ClearCollections()
		{
			cudaEventRecord(_evClearCollectionsStart, _stream);

			int requestsCount = *_hRequestsCount;
			int threadsPerBlock = 256;

			KernelClearCollections<<<requestsCount, threadsPerBlock, 0, _stream>>>(
				_visiteds,
				_frontiers,
				_dFoundFlags
			);

			cudaEventRecord(_evClearCollectionsEnd);
		}

		void PrepareSearch()
		{
			cudaEventRecord(_evPrepareSearchStart, _stream);

			KernelClearRequests<<<1, 1, 0, _stream>>>(
				_requests.D(0),
				_dRequestsCount
			);

			int threadsCount = _agents.H(0).Count();
			int threadsPerBlock = 128;
			int blocksCount = threadsCount / threadsPerBlock;
			if (blocksCount * threadsPerBlock < threadsCount)
				blocksCount += 1;

			KernelPrepareSearch<<<blocksCount, threadsPerBlock, 0, _stream >>>(
				_agents.D(0),
				_requests.D(0),
				_dRequestsCount
			);

			cudaMemcpyAsync(_hRequestsCount, _dRequestsCount, sizeof(int), cudaMemcpyDeviceToHost, _stream);

			cudaEventRecord(_evPrepareSearchEnd, _stream);
		}

		void Search()
		{
			int requestsCount = *_hRequestsCount;

			//printf("requests count %d\n", requestsCount);

			cudaEventRecord(_evSearchStart, _stream);
			KernelSearch<<<requestsCount, _threadsPerAgent, 0, _stream>>>(
				_heuristicK,
				_map.D(0),
				_requests.D(0),
				_visiteds,
				_frontiers,
				_queues,
				_dFoundFlags
			);
			cudaEventRecord(_evSearchEnd, _stream);
		}

		void BuildPaths()
		{
			cudaEventRecord(_evBuildPathsStart, _stream);

			int threadsCount = *_hRequestsCount;
			int threadsPerBlock = 32;
			int blocksCount = threadsCount / threadsPerBlock;
			if (blocksCount * threadsPerBlock < threadsCount)
				blocksCount += 1;

			KernelBuildPath<<<blocksCount, threadsPerBlock, 0, _stream>>>(
				_agents.D(0),
				_requests.D(0),
				_visiteds,
				_dFoundFlags
			);

			cudaEventRecord(_evBuildPathsEnd, _stream);
		}

		void DebugRecordDurs()
		{
			float t1 = 0;
			cudaEventElapsedTime(&t1, _evClearCollectionsStart, _evClearCollectionsEnd);
			DebugDurClearCollectionsMax = std::max(t1, DebugDurClearCollectionsMax);
			DebugDurClearCollections += t1;

			float t2 = 0;
			cudaEventElapsedTime(&t2, _evPrepareSearchStart, _evPrepareSearchEnd);
			DebugDurPrepareSearchMax = std::max(t2, DebugDurPrepareSearchMax);
			DebugDurPrepareSearch += t2;

			float t3 = 0;
			cudaEventElapsedTime(&t3, _evSearchStart, _evSearchEnd);
			DebugDurSearchMax = std::max(t3, DebugDurSearchMax);
			DebugDurSearch += t3;

			float t4 = 0;
			cudaEventElapsedTime(&t4, _evBuildPathsStart, _evBuildPathsEnd);
			DebugDurBuildPathsMax = std::max(t4, DebugDurBuildPathsMax);
			DebugDurBuildPaths += t4;

			//printf("%f\n", temp);

			DebugRecordsCount += 1;
		}
	};
}
