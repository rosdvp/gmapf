#pragma once

#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Agent.h"
#include "Helpers.h"
#include "PathAllocator.h"
#include "misc/CuList.h"
#include "misc/Cum.h"
#include "misc/CuNodesMap.h"
#include "misc/CuQueue.h"
#include "misc/CuVisitedMap.h"
#include "misc/V2Int.h"

namespace gmapf
{
	struct __align__(8) PathRequest
	{
		int AgentIdx;
		int StartNodeIdx;
		int TargetNodeIdx;
		int FoundFlag;
		int MiddleNodeIdxFromStart;
		int MiddleNodeIdxFromTarget;
	};

	struct __align__(8) PathNode
	{
		int NodeIdx;
		int PrevNodeIdx;
		float F;
		int G;
		int FromStart;
	};

	static __global__ void KernelMarkCollections(
		CuNodesMap map,
		CuList<PathRequest> requests,
		Cum<CuVisitedMap<PathNode>> visiteds,
		Cum<CuList<PathNode>> frontiers,
		int frontierCapacity,
		Cum<CuQueue<PathNode>> queues,
		int queueCapacity)
	{
		int tid = threadIdx.x;
		int bid = blockIdx.x;
		int threadsPerAgent = blockDim.x;

		int parallelAgentsCount = gridDim.x;

		if (tid == 0)
		{
			visiteds.D(bid).Mark(map.Count());
			frontiers.D(bid).Mark(frontierCapacity);
		}

		if (tid == 0)
			frontiers.D(bid).Mark(frontierCapacity);

		queues.D(bid * threadsPerAgent + tid).Mark(queueCapacity);

		if (tid == 0 && bid == 0)
			requests.Mark(parallelAgentsCount);
	}

	static __global__ void KernelClearRequests(
		CuList<PathRequest> requests,
		int* requestsCount)
	{
		requests.RemoveAll();
		*requestsCount = 0;
	}

	static __global__ void KernelFindRequests(
		CuNodesMap map,
		CuList<Agent> agents,
		CuList<PathRequest> requests,
		int* requestsCount)
	{
		int tid = blockDim.x * blockIdx.x + threadIdx.x;
		int threadsCount = gridDim.x * blockDim.x;

		for (int i = tid; i < agents.Count(); i += threadsCount)
		{
			Agent& agent = agents.At(i);
			if (agent.State != EAgentState::Search)
				continue;

			if (agent.CurrNodeIdx == agent.TargNodeIdx)
			{
				agent.State = EAgentState::Idle;
				continue;
			}

			int oldCount = atomicAdd(requestsCount, 1);
			if (oldCount >= requests.Capacity())
			{
				*requestsCount = requests.Capacity();
				return;
			}

			if (agent.Path != nullptr)
			{
				cudaFree(agent.Path);
				agent.Path = nullptr;
			}

			PathRequest req;
			req.AgentIdx = i;
			req.StartNodeIdx = agent.CurrNodeIdx;
			req.TargetNodeIdx = agent.TargNodeIdx;
			req.FoundFlag = 0;
			requests.AddAtomic(req);
		}
	}

	static __global__ void KernelClearCollections(
		Cum<CuVisitedMap<PathNode>> visiteds,
		Cum<CuList<PathNode>> frontiers)
	{
		int bid = blockIdx.x;
		int tid = threadIdx.x;
		int threadsCount = blockDim.x;

		auto visited = visiteds.D(bid);
		for (int i = tid; i < visited.Count(); i += threadsCount)
		{
			visited.UnVisit(i);
			visited.At(i).FromStart = -1;
		}

		if (tid == 0)
			frontiers.D(bid).RemoveAll();
	}

	static __global__ void KernelSearch(
		int queuesPerAgent,
		float heuristicK,
		CuNodesMap map,
		CuList<PathRequest> requests,
		Cum<CuVisitedMap<PathNode>> visiteds,
		Cum<CuList<PathNode>> frontiers,
		Cum<CuQueue<PathNode>> queues)
	{
		int tid = blockDim.x * blockIdx.x + threadIdx.x;
		int reqIdx = tid / queuesPerAgent;
		if (reqIdx >= requests.Count())
			return;
		int requestsInBlock = blockDim.x / queuesPerAgent;
		int requestInBlockIdx = reqIdx % requestsInBlock;
		int queueIdx = tid % queuesPerAgent;
		int queueInBlockIdx = reqIdx * queuesPerAgent + queueIdx;

		PathRequest& request = requests.At(reqIdx);
		int startNodeIdx = request.StartNodeIdx;
		int targetNodeIdx = request.TargetNodeIdx;

		auto visited = visiteds.D(reqIdx);
		auto frontier = frontiers.D(reqIdx);
		auto queue = queues.D(queueInBlockIdx);
		queue.RemoveAll();

		__shared__ int sharedQueuesTotalCounts[64];
		sharedQueuesTotalCounts[requestInBlockIdx] = 0;

		if (queueIdx == 0)
		{
			PathNode startNode;
			startNode.NodeIdx = startNodeIdx;
			startNode.F = map.GetDistSqr(startNodeIdx, targetNodeIdx) * heuristicK;
			startNode.G = 0;
			startNode.FromStart = 1;
			queue.Push(startNode);

			visited.TryVisit(startNodeIdx, startNode);
		}
		if (queueIdx == 1)
		{
			PathNode targNode;
			targNode.NodeIdx = targetNodeIdx;
			targNode.F = map.GetDistSqr(startNodeIdx, targetNodeIdx) * heuristicK;
			targNode.G = 0;
			targNode.FromStart = 0;
			queue.Push(targNode);

			visited.TryVisit(targetNodeIdx, targNode);
		}

		while (request.FoundFlag == 0)
		{
			if (queue.Count() > 0)
			{
				PathNode curr = queue.Pop();
				//printf("pop (%d, %d) %s\n", curr.Cell.X, curr.Cell.Y, curr.FromStart ? "from start" : "from target");

				for (auto& neibNodeIdx : map.At(curr.NodeIdx).NeibsIdx)
				{
					if (neibNodeIdx == CuNodesMap::INVALID)
						break;

					float h = curr.FromStart
						? map.GetDistSqr(neibNodeIdx, targetNodeIdx)
						: map.GetDistSqr(neibNodeIdx, startNodeIdx);

					PathNode neib;
					neib.NodeIdx = neibNodeIdx;
					neib.PrevNodeIdx = curr.NodeIdx;
					neib.G = curr.G + 1;
					neib.F = neib.G + h * heuristicK;
					neib.FromStart = curr.FromStart;

					if (visited.TryVisit(neibNodeIdx, neib))
					{
						frontier.AddAtomic(neib);
					}
					else
					{
						auto& existNode = visited.At(neib.NodeIdx);
						if (existNode.FromStart != -1 && 
							existNode.FromStart != neib.FromStart)
						{
							if (atomicExch(&request.FoundFlag, 1) == 0)
							{
								request.MiddleNodeIdxFromStart = neib.FromStart == 1 ? neib.PrevNodeIdx : existNode.NodeIdx;
								request.MiddleNodeIdxFromTarget = neib.FromStart == 1 ? existNode.NodeIdx : neib.PrevNodeIdx;
							}
						}
					}
				}
			}

			__syncthreads();

			for (int i = queueIdx; i < frontier.Count(); i += queuesPerAgent)
			{
				auto& newNode = frontier.At(i);
				queue.Push(newNode);
			}

			atomicAdd(sharedQueuesTotalCounts + requestInBlockIdx, queue.Count());
			__syncthreads();
			if (sharedQueuesTotalCounts[requestInBlockIdx] == 0)
				return;
			if (queueIdx == 0)
				frontier.RemoveAll();
			__syncthreads();
			sharedQueuesTotalCounts[requestInBlockIdx] = 0;
		}
	}

	static __global__ void KernelBuildPath(
		CuList<Agent> agents,
		CuList<PathRequest> requests,
		Cum<CuVisitedMap<PathNode>> visiteds,
		PathAllocator pathAllocator)
	{
		int tid = blockDim.x * blockIdx.x + threadIdx.x;
		int reqIdx = tid / 2;
		if (reqIdx >= requests.Count())
			return;
		bool isFromStart = tid % 2 == 0;

		PathRequest& request = requests.At(reqIdx);
		Agent& agent = agents.At(request.AgentIdx);

		if (request.FoundFlag == 0)
		{
			agent.State = EAgentState::Idle;
			if (isFromStart)
				printf("path not found\n");
			return;
		}

		auto visited = visiteds.D(reqIdx);
		int lengthFromStart = visited.At(request.MiddleNodeIdxFromStart).G;
		int lengthFromTarget = visited.At(request.MiddleNodeIdxFromTarget).G + 1;

		if (isFromStart)
		{
			agent.State = EAgentState::Move;
			agent.PathStepIdx = -1;

			int pathLength = lengthFromStart + lengthFromTarget;

			agent.Path = pathAllocator.GetPath(pathLength);
			CuList<int> path(agent.Path);
			path.PreAdd(pathLength);
		}
		__syncthreads();

		CuList<int> path(agent.Path);
		int idx = isFromStart ? lengthFromStart - 1 : lengthFromStart;
		int idxStep = isFromStart ? -1 : 1;
		int targNodeIdx = isFromStart ? request.StartNodeIdx : request.TargetNodeIdx;
		int nodeIdx = isFromStart ? request.MiddleNodeIdxFromStart : request.MiddleNodeIdxFromTarget;

		while(nodeIdx != targNodeIdx)
		{
			path.At(idx) = nodeIdx;
			idx += idxStep;
			nodeIdx = visited.At(nodeIdx).PrevNodeIdx;
		}
		if (!isFromStart)
			path.At(idx) = targNodeIdx;
	}


	static __global__ void KernelCheckPaths(
		CuNodesMap map,
		CuList<Agent> agents)
	{
		for (int agentIdx = 0; agentIdx < agents.Count(); agentIdx++)
		{
			auto& agent = agents.At(agentIdx);
			if (agent.Path == nullptr)
				continue;

			printf("agent %d path:\n", agentIdx);

			CuList<int> path(agent.Path);
			for (int i = 0; i < path.Count(); i++)
			{
				auto pos = map.GetPos(path.At(i));
				printf("(%f, %f)\n", pos.X, pos.Y);
			}

			printf("-------------------------\n");
		}
	}


	class PathFinder
	{
	public:
		bool IsDebug = false;
		bool IsProfiler = false;
		float ProfDurClearCollections = 0;
		float ProfDurClearCollectionsSum = 0;
		float ProfDurClearCollectionsMax = 0;
		float ProfDurFindRequests = 0;
		float ProfDurFindRequestsSum = 0;
		float ProfDurFindRequestsMax = 0;
		float ProfDurSearch = 0;
		float ProfDurSearchSum = 0;
		float ProfDurSearchMax = 0;
		float ProfDurBuildPaths = 0;
		float ProfDurBuildPathsSum = 0;
		float ProfDurBuildPathsMax = 0;
		int ProfRecordsCount = 0;


		void Init(
			const Cum<CuNodesMap>& map,
			const Cum<CuList<Agent>>& agents,
			int parallelAgentsCount,
			int threadsPerAgent,
			int queueCapacity,
			float heuristicK)
		{
			_map = map;
			_agents = agents;

			_queuesPerAgent = threadsPerAgent;
			_heuristicK = heuristicK;

			int frontierCapacity = (threadsPerAgent+1) * 8;

			_requests.DAlloc(1, parallelAgentsCount);
				
			auto hMap = _map.H(0);
			_visiteds.DAlloc(parallelAgentsCount, hMap.Count());

			_frontiers.DAlloc(parallelAgentsCount, frontierCapacity);
			_queues.DAlloc(parallelAgentsCount * threadsPerAgent, queueCapacity);

			_hRequestsCount = static_cast<int*>(malloc(sizeof(int)));

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
			if (IsProfiler)
			{
				cudaEventCreate(&_evClearCollectionsStart);
				cudaEventCreate(&_evClearCollectionsEnd);
				cudaEventCreate(&_evFindRequestsStart);
				cudaEventCreate(&_evFindRequestsEnd);
				cudaEventCreate(&_evSearchStart);
				cudaEventCreate(&_evSearchEnd);
				cudaEventCreate(&_evBuildPathsStart);
				cudaEventCreate(&_evBuildPathsEnd);
			}

			_pathAllocator.Init(_agents.H(0).Count(), _map.H(0).Count());
		}

		~PathFinder()
		{
			cudaStreamDestroy(_stream);

			if (IsProfiler)
			{
				cudaEventDestroy(_evClearCollectionsStart);
				cudaEventDestroy(_evClearCollectionsEnd);
				cudaEventDestroy(_evFindRequestsStart);
				cudaEventDestroy(_evFindRequestsEnd);
				cudaEventDestroy(_evSearchStart);
				cudaEventDestroy(_evSearchEnd);
				cudaEventDestroy(_evBuildPathsStart);
				cudaEventDestroy(_evBuildPathsEnd);
			}

			_requests.DFree();
			_visiteds.DFree();
			_frontiers.DFree();
			_queues.DFree();

			cudaFree(_dRequestsCount);

			free(_hRequestsCount);

			CudaCatch();
		}

		void AsyncPreFind()
		{
			FindRequests();
			if (IsDebug)
				CudaSyncAndCatch();
		}

		void AsyncFind()
		{
			if (*_hRequestsCount == 0)
				return;
			Search();
			if (IsDebug)
				CudaSyncAndCatch();
			BuildPaths();
			if (IsDebug)
				CudaSyncAndCatch();
		}

		void PostFind()
		{
			//if (*_hRequestsCount > 0)
			//{
			//	KernelCheckPaths<<<1, 1, 0, _stream>>>(
			//		_map.D(0),
			//		_agents.D(0)
			//	);
			//	CudaCheck(cudaStreamSynchronize(_stream), "path finder, check paths");
			//}

			if (IsProfiler)
				RecordProfiler();
		}

		void Sync()
		{
			CudaCheck(cudaStreamSynchronize(_stream), "path finder");
		}


	private:
		int _queuesPerAgent = 0;
		float _heuristicK = 1.0f;

		Cum<CuNodesMap> _map;
		Cum<CuList<Agent>> _agents;

		Cum<CuList<PathRequest>> _requests;

		Cum<CuVisitedMap<PathNode>> _visiteds;
		Cum<CuList<PathNode>> _frontiers;
		Cum<CuQueue<PathNode>> _queues;
		int* _hRequestsCount = nullptr;
		int* _dRequestsCount = nullptr;

		PathAllocator _pathAllocator;


		cudaStream_t _stream{};
		cudaEvent_t _evClearCollectionsStart{};
		cudaEvent_t _evClearCollectionsEnd{};
		cudaEvent_t _evFindRequestsStart{};
		cudaEvent_t _evFindRequestsEnd{};
		cudaEvent_t _evSearchStart{};
		cudaEvent_t _evSearchEnd{};
		cudaEvent_t _evBuildPathsStart{};
		cudaEvent_t _evBuildPathsEnd{};


		void FindRequests()
		{
			if (IsProfiler)
				cudaEventRecord(_evFindRequestsStart, _stream);

			KernelClearRequests<<<1, 1, 0, _stream>>>(
				_requests.D(0),
				_dRequestsCount
			);
			if (IsDebug)
				CudaSyncAndCatch();

			int threadsCount = _agents.H(0).Count();
			int threadsPerBlock = 128;
			int blocksCount = threadsCount / threadsPerBlock;
			if (blocksCount * threadsPerBlock < threadsCount)
				blocksCount += 1;

			KernelFindRequests<<<blocksCount, threadsPerBlock, 0, _stream >>>(
				_map.D(0),
				_agents.D(0),
				_requests.D(0),
				_dRequestsCount
			);

			cudaMemcpyAsync(_hRequestsCount, _dRequestsCount, sizeof(int), cudaMemcpyDeviceToHost, _stream);

			if (IsProfiler)
				cudaEventRecord(_evFindRequestsEnd, _stream);
		}

		void Search()
		{
			if (IsProfiler)
				cudaEventRecord(_evClearCollectionsStart, _stream);

			int requestsCount = *_hRequestsCount;

			KernelClearCollections << <requestsCount, 256, 0, _stream >> > (
				_visiteds,
				_frontiers
				);
			if (IsDebug)
				CudaSyncAndCatch();

			if (IsProfiler)
				cudaEventRecord(_evClearCollectionsEnd);

			if (IsProfiler)
				cudaEventRecord(_evSearchStart, _stream);

			int threadsCount = *_hRequestsCount * _queuesPerAgent;
			int threadsPerBlock = 128;
			int blocks = threadsCount / threadsPerBlock;
			if (blocks * threadsPerBlock < threadsCount)
				blocks += 1;

			KernelSearch<<<blocks, threadsPerBlock, 0, _stream>>>(
				_queuesPerAgent,
				_heuristicK,
				_map.D(0),
				_requests.D(0),
				_visiteds,
				_frontiers,
				_queues
			);
			if (IsProfiler)
				cudaEventRecord(_evSearchEnd, _stream);
		}

		void BuildPaths()
		{
			if (IsProfiler)
				cudaEventRecord(_evBuildPathsStart, _stream);

			int threadsCount = *_hRequestsCount * 2;
			int threadsPerBlock = 32;
			int blocksCount = threadsCount / threadsPerBlock;
			if (blocksCount * threadsPerBlock < threadsCount)
				blocksCount += 1;

			KernelBuildPath<<<blocksCount, threadsPerBlock, 0, _stream>>>(
				_agents.D(0),
				_requests.D(0),
				_visiteds,
				_pathAllocator
			);

			if (IsProfiler)
				cudaEventRecord(_evBuildPathsEnd, _stream);
		}

		void RecordProfiler()
		{
			cudaEventElapsedTime(&ProfDurFindRequests, _evFindRequestsStart, _evFindRequestsEnd);
			ProfDurFindRequestsMax = std::max(ProfDurFindRequests, ProfDurFindRequestsMax);
			ProfDurFindRequestsSum += ProfDurFindRequests;

			if (*_hRequestsCount == 0)
			{
				ProfDurClearCollections = 0;
				ProfDurSearch = 0;
				ProfDurBuildPaths = 0;
				return;
			}

			cudaEventElapsedTime(&ProfDurClearCollections, _evClearCollectionsStart, _evClearCollectionsEnd);
			ProfDurClearCollectionsMax = std::max(ProfDurClearCollections, ProfDurClearCollectionsMax);
			ProfDurClearCollectionsSum += ProfDurClearCollections;

			cudaEventElapsedTime(&ProfDurSearch, _evSearchStart, _evSearchEnd);
			ProfDurSearchMax = std::max(ProfDurSearch, ProfDurSearchMax);
			ProfDurSearchSum += ProfDurSearch;

			cudaEventElapsedTime(&ProfDurBuildPaths, _evBuildPathsStart, _evBuildPathsEnd);
			ProfDurBuildPathsMax = std::max(ProfDurBuildPaths, ProfDurBuildPathsMax);
			ProfDurBuildPathsSum += ProfDurBuildPaths;

			ProfRecordsCount += 1;
		}
	};
}
