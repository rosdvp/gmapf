#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Agent.h"
#include "Helpers.h"
#include "misc/CuList.h"
#include "misc/Cum.h"
#include "misc/CuQueue.h"
#include "misc/CuNodesMap.h"
#include "misc/CuVisitedMap.h"
#include "misc/V2Int.h"

namespace gmapf
{
	namespace STMA
	{
		struct __align__(8) PathRequest
		{
			int AgentIdx;
			int StartNodeIdx;
			int TargetNodeIdx;
			bool IsFound;
		};

		struct __align__(8) PathNode
		{
			int NodeIdx;
			int PrevNodeIdx;
			float F;
			int G;
		};

		static __global__ void KernelMarkCollections(
			int requestsCapacity,
			CuNodesMap map,
			CuList<PathRequest> requests,
			Cum<CuVisitedMap<PathNode>> visiteds,
			Cum<CuQueue<PathNode>> queues,
			int queueCapacity)
		{
			int tid = blockDim.x * blockIdx.x + threadIdx.x;
			if (tid >= requestsCapacity)
				return;

			if (tid == 0)
			{
				requests.Mark(requestsCapacity);
			}

			visiteds.D(tid).Mark(map.Count());
			queues.D(tid).Mark(queueCapacity);
		}

		__global__ void KernelClearRequests(
			CuList<PathRequest> requests,
			int* requestsCount)
		{
			requests.RemoveAll();
			*requestsCount = 0;
		}

		static __global__ void KernelClearVisiteds(
			Cum<CuVisitedMap<PathNode>> visiteds)
		{
			int bid = blockIdx.x;
			int tid = threadIdx.x;
			int threadsCount = blockDim.x;

			auto visited = visiteds.D(bid);
			for (int i = tid; i < visited.Count(); i += threadsCount)
				visited.UnVisit(i);
		}

		static __global__ void KernelPrepareSearch(
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
				PathRequest req;
				req.AgentIdx = i;
				req.StartNodeIdx = agent.CurrNodeIdx;
				req.TargetNodeIdx = agent.TargNodeIdx;
				requests.AddAtomic(req);
			}
		}

		static __global__ void KernelSearch(
			float heuristicK,
			CuNodesMap map,
			CuList<PathRequest> requests,
			Cum<CuVisitedMap<PathNode>> visiteds,
			Cum<CuQueue<PathNode>> queues)
		{
			int tid = blockDim.x * blockIdx.x + threadIdx.x;
			if (tid >= requests.Count())
				return;

			auto visited = visiteds.D(tid);
			auto queue = queues.D(tid);
			queue.RemoveAll();

			PathRequest& request = requests.At(tid);
			request.IsFound = false;

			int startNodeIdx = request.TargetNodeIdx;
			int targNodeIdx = request.StartNodeIdx;

			PathNode startNode;
			startNode.NodeIdx = startNodeIdx;
			startNode.F = map.GetDistSqr(startNodeIdx, targNodeIdx) * heuristicK;
			startNode.G = 0;
			queue.Push(startNode);

			visited.TryVisit(startNodeIdx, startNode);

			while (true)
			{
				PathNode curr = queue.Pop();

				for (int neibIdx : map.At(curr.NodeIdx).NeibsIdx)
				{
					if (neibIdx == CuNodesMap::INVALID)
						continue;

					PathNode neib;
					neib.NodeIdx = neibIdx;
					neib.PrevNodeIdx = curr.NodeIdx;
					neib.G = curr.G + 1;
					neib.F = neib.G + map.GetDistSqr(neibIdx, targNodeIdx) * heuristicK;

					if (visited.TryVisit(neibIdx, neib))
					{
						if (neibIdx == targNodeIdx)
						{
							request.IsFound = true;
							return;
						}
						queue.Push(neib);
					}
				}

				if (queue.Count() == 0)
					return;
			}
		}

		static __global__ void KernelBuildPath(
			CuList<Agent> agents,
			CuList<PathRequest> requests,
			Cum<CuVisitedMap<PathNode>> visiteds)
		{
			int tid = blockDim.x * blockIdx.x + threadIdx.x;
			if (tid >= requests.Count())
				return;

			PathRequest& request = requests.At(tid);
			Agent& agent = agents.At(request.AgentIdx);

			if (request.IsFound == false)
			{
				agent.State = EAgentState::Idle;
				return;
			}

			auto visited = visiteds.D(tid);

			int pathLength = visited.At(request.StartNodeIdx).G;

			void* pPath;
			cudaMalloc(&pPath, CuList<int>::EvalSize(pathLength));
			CuList<int> path(pPath);
			path.Mark(pathLength);

			int iter = request.StartNodeIdx;
			while (iter != request.TargetNodeIdx)
			{
				iter = visited.At(iter).PrevNodeIdx;
				path.Add(iter);
			}

			agent.State = EAgentState::Move;
			agent.PathStepIdx = -1;
			agent.Path = pPath;
		}


		__global__ void KernelCheckPaths(
			CuNodesMap map,
			CuList<Agent> agents)
		{
			for (int agentIdx = 0; agentIdx < agents.Count(); agentIdx++)
			{
				auto& agent = agents.At(agentIdx);
				if (agent.Path == nullptr)
					continue;
				CuList<int> path(agent.Path);

				printf("agent %d path:\n", agentIdx);
				for (int i = 0; i < path.Count(); i++)
				{
					auto pos = map.GetPos(path.At(i));
					printf("(%f, %f)\n", pos.X, pos.Y);
				}
				printf("-------------------------\n");
			}
		}


		class PathFinderSTMA
		{
		public:
			bool IsDebug = false;
			bool IsProfiler = false;
			float ProfDurClearCollectionsSum = 0;
			float ProfDurClearCollectionsMax = 0;
			float ProfDurPrepareSearchSum = 0;
			float ProfDurPrepareSearchMax = 0;
			float ProfDurSearchSum = 0;
			float ProfDurSearchMax = 0;
			float ProfDurBuildPathsSum = 0;
			float ProfDurBuildPathsMax = 0;
			float ProfDurAttachPathsSum = 0;
			float ProfDurAttachPathsMax = 0;
			int ProfRecordsCount = 0;


			void Init(
				const Cum<CuNodesMap>& map,
				const Cum<CuList<Agent>>& agents,
				int parallelRequestsCount,
				int queueCapacity,
				float heuristicK)
			{
				_map = map;
				_agents = agents;

				_heuristicK = heuristicK;

				_requests.DAlloc(1, parallelRequestsCount);

				auto hMap = _map.H(0);
				_visiteds.DAlloc(parallelRequestsCount, hMap.Count());

				_queues.DAlloc(parallelRequestsCount, queueCapacity);

				_hRequestsCount = static_cast<int*>(malloc(sizeof(int)));

				cudaMalloc(&_dRequestsCount, sizeof(int));

				int threadsPerBlock = 32;
				int blocks = parallelRequestsCount / threadsPerBlock;
				if (blocks * threadsPerBlock < parallelRequestsCount)
					blocks += 1;
				KernelMarkCollections << <blocks, threadsPerBlock >> > (
					parallelRequestsCount,
					_map.D(0),
					_requests.D(0),
					_visiteds,
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
					cudaEventCreate(&_evPrepareSearchStart);
					cudaEventCreate(&_evPrepareSearchEnd);
					cudaEventCreate(&_evSearchStart);
					cudaEventCreate(&_evSearchEnd);
					cudaEventCreate(&_evBuildPathsStart);
					cudaEventCreate(&_evBuildPathsEnd);
				}
			}

			~PathFinderSTMA()
			{
				cudaStreamDestroy(_stream);
				if (IsProfiler)
				{
					cudaEventDestroy(_evClearCollectionsStart);
					cudaEventDestroy(_evClearCollectionsEnd);
					cudaEventDestroy(_evPrepareSearchStart);
					cudaEventDestroy(_evPrepareSearchEnd);
					cudaEventDestroy(_evSearchStart);
					cudaEventDestroy(_evSearchEnd);
					cudaEventDestroy(_evBuildPathsStart);
					cudaEventDestroy(_evBuildPathsEnd);
				}

				_requests.DFree();
				_visiteds.DFree();
				_queues.DFree();

				cudaFree(_dRequestsCount);

				free(_hRequestsCount);

				CudaCatch();
			}

			void AsyncPreFind()
			{
				PrepareSearch();
				if (IsDebug)
					CudaSyncAndCatch();
			}

			void AsyncFind()
			{
				if (*_hRequestsCount == 0)
					return;
				ClearVisiteds();
				if (IsDebug)
					CudaSyncAndCatch();
				Search();
				if (IsDebug)
					CudaSyncAndCatch();
				BuildPaths();
				if (IsDebug)
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

				if (IsProfiler)
					RecordProfiler();
			}

			void Sync()
			{
				CudaCheck(cudaStreamSynchronize(_stream), "path finder");
			}


		private:
			float _heuristicK = 1.0f;

			Cum<CuNodesMap> _map;
			Cum<CuList<Agent>> _agents;

			Cum<CuList<PathRequest>> _requests;

			Cum<CuVisitedMap<PathNode>> _visiteds;
			Cum<CuQueue<PathNode>> _queues;
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


			void ClearVisiteds()
			{
				if (IsProfiler)
					cudaEventRecord(_evClearCollectionsStart, _stream);

				int requestsCount = *_hRequestsCount;
				int threadsPerBlock = 256;

				KernelClearVisiteds << <requestsCount, threadsPerBlock, 0, _stream >> > (
					_visiteds
					);

				if (IsProfiler)
					cudaEventRecord(_evClearCollectionsEnd);
			}

			void PrepareSearch()
			{
				if (IsProfiler)
					cudaEventRecord(_evPrepareSearchStart, _stream);

				KernelClearRequests << <1, 1, 0, _stream >> > (
					_requests.D(0),
					_dRequestsCount
					);

				int threadsCount = _agents.H(0).Count();
				int threadsPerBlock = 128;
				int blocksCount = threadsCount / threadsPerBlock;
				if (blocksCount * threadsPerBlock < threadsCount)
					blocksCount += 1;

				KernelPrepareSearch << <blocksCount, threadsPerBlock, 0, _stream >> > (
					_agents.D(0),
					_requests.D(0),
					_dRequestsCount
					);

				cudaMemcpyAsync(_hRequestsCount, _dRequestsCount, sizeof(int), cudaMemcpyDeviceToHost, _stream);

				if (IsProfiler)
					cudaEventRecord(_evPrepareSearchEnd, _stream);
			}

			void Search()
			{
				if (IsProfiler)
					cudaEventRecord(_evSearchStart, _stream);

				int requestsCount = *_hRequestsCount;
				int threadsPerBlock = 32;
				int blocks = requestsCount / threadsPerBlock;
				if (blocks * threadsPerBlock < requestsCount)
					blocks += 1;

				KernelSearch << <blocks, threadsPerBlock, 0, _stream >> > (
					_heuristicK,
					_map.D(0),
					_requests.D(0),
					_visiteds,
					_queues
					);
				if (IsProfiler)
					cudaEventRecord(_evSearchEnd, _stream);
			}

			void BuildPaths()
			{
				cudaEventRecord(_evBuildPathsStart, _stream);

				int threadsCount = *_hRequestsCount;
				int threadsPerBlock = 128;
				int blocksCount = threadsCount / threadsPerBlock;
				if (blocksCount * threadsPerBlock < threadsCount)
					blocksCount += 1;

				KernelBuildPath << <blocksCount, threadsPerBlock, 0, _stream >> > (
					_agents.D(0),
					_requests.D(0),
					_visiteds
					);

				if (IsProfiler)
					cudaEventRecord(_evBuildPathsEnd, _stream);
			}

			void RecordProfiler()
			{
				float t1 = 0;
				cudaEventElapsedTime(&t1, _evClearCollectionsStart, _evClearCollectionsEnd);
				ProfDurClearCollectionsMax = std::max(t1, ProfDurClearCollectionsMax);
				ProfDurClearCollectionsSum += t1;

				float t2 = 0;
				cudaEventElapsedTime(&t2, _evPrepareSearchStart, _evPrepareSearchEnd);
				ProfDurPrepareSearchMax = std::max(t2, ProfDurPrepareSearchMax);
				ProfDurPrepareSearchSum += t2;

				float t3 = 0;
				cudaEventElapsedTime(&t3, _evSearchStart, _evSearchEnd);
				ProfDurSearchMax = std::max(t3, ProfDurSearchMax);
				ProfDurSearchSum += t3;

				float t4 = 0;
				cudaEventElapsedTime(&t4, _evBuildPathsStart, _evBuildPathsEnd);
				ProfDurBuildPathsMax = std::max(t4, ProfDurBuildPathsMax);
				ProfDurBuildPathsSum += t4;

				ProfRecordsCount += 1;
			}
		};
	}
}
