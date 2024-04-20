#pragma once

#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Agent.h"
#include "Helpers.h"
#include "misc/CuList.h"
#include "misc/Cum.h"
#include "misc/CuNodesMap.h"

namespace cupat
{
	static __global__ void KernelMarkCollections(
		CuList<int> agentsIndices,
		int parallelAgentsCount)
	{
		agentsIndices.Mark(parallelAgentsCount);
	}

	static __global__ void KernelClearCollections(
		CuList<int> agentsIndices,
		int* procAgentsCount)
	{
		agentsIndices.RemoveAll();
		*procAgentsCount = 0;
	}

	static __global__ void KernelFindMovingAgents(
		CuList<Agent> agents,
		CuList<int> agentsIndices,
		int* procAgentsCount)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= agents.Count())
			return;
		int threadsCount = gridDim.x * blockDim.x;

		for (int i = tid; i < agents.Count(); i += threadsCount)
		{
			Agent& agent = agents.At(i);
			if (agent.State != EAgentState::Move)
				continue;
			agentsIndices.AddAtomic(i);
			atomicAdd(procAgentsCount, 1);

			//printf("moving agent idx: %d\n", i);
		}
	}

	static __global__ void KernelMoveAgents(
		CuList<Agent> agents,
		CuList<int> agentsIndices,
		float agentStep
		)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= agentsIndices.Count())
			return;
		int agentIdx = agentsIndices.At(tid);
		Agent& agent = agents.At(agentIdx);
		if (agent.PathStepIdx == -1)
			return;

		V2Float delta = agent.PathStepPos - agent.CurrPos;
		float deltaLen = delta.GetLength();
		if (deltaLen <= agentStep)
			agent.CurrPos = agent.PathStepPos;
		else
			agent.CurrPos += (delta / deltaLen) * agentStep;
	}

	static __global__ void KernelResolveCollisions(
		CuNodesMap map,
		CuList<Agent> agents,
		float collisionDistSqr,
		float collisionDist)
	{
		int tid = threadIdx.x;
		int bid = blockIdx.x;
		int threadsCount = blockDim.x;

		Agent& agent = agents.At(bid);

		__shared__ float shX;
		__shared__ float shY;
		__shared__ int shCount;
		if (tid == 0)
		{
			shX = 0;
			shY = 0;
			shCount = 0;
		}
		__syncthreads();

		for (int i = tid; i < agents.Count(); i += threadsCount)
		{
			if (i == bid)
				continue;

			Agent& other = agents.At(i);
			V2Float delta = agent.CurrPos - other.CurrPos;
			float distSqr = delta.GetLengthSqr();
			if (distSqr < collisionDistSqr)
			{
				float dist = delta.GetLength();
				float shift = collisionDist - dist;
				delta = (delta / dist) * shift;
				atomicAdd(&shX, delta.X);
				atomicAdd(&shY, delta.Y);
				atomicAdd(&shCount, 1);
			}
		}
		__syncthreads();

		if (tid != 0)
			return;
		if (shCount == 0)
			return;

		V2Float avoidStep(shX / shCount, shY / shCount);
		avoidStep = avoidStep.GetRotated(10);

		V2Float newPos = agent.CurrPos + avoidStep;
		if (map.TryGetNodeIdx(newPos, nullptr))
			agent.CurrPos = newPos;

		//printf("avoid step (%f, %f)\n", avoidStep.X, avoidStep.Y);
	}

	__global__ void KernelUpdatePathProgress(
		CuNodesMap map,
		CuList<Agent> agents,
		CuList<int> agentsIndices)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= agentsIndices.Count())
			return;
		int agentIdx = agentsIndices.At(tid);
		Agent& agent = agents.At(agentIdx);

		if (agent.PathStepIdx == -1)
		{
			agent.PathStepIdx = 0;
			CuList<int> path(agent.Path);
			agent.PathStepNode = path.At(0);
			agent.PathStepPos = map.GetPos(agent.PathStepNode);
		}
		else
		{
			if (!map.IsInNode(agent.CurrPos, agent.PathStepNode))
				return;

			agent.CurrNodeIdx = agent.PathStepNode;

			CuList<int> path(agent.Path);
			if (agent.PathStepIdx < path.Count())
				agent.PathStepIdx += 1;
			if (agent.PathStepIdx == path.Count())
			{
				if ((agent.TargPos - agent.CurrPos).GetLength() > FLT_EPSILON)
				{
					agent.PathStepPos = agent.TargPos;
				}
				else
				{
					agent.State = EAgentState::Idle;
				}
			}
			else
			{
				agent.PathStepNode = path.At(agent.PathStepIdx);
				agent.PathStepPos = map.GetPos(agent.PathStepNode);
			}
		}
	}

	__global__ void KernelFreeAgentsPaths(
		CuList<Agent> agents)
	{
		for (int i = 0; i < agents.Count(); i++)
			if (agents.At(i).Path != nullptr)
			{
				cudaFree(agents.At(i).Path);
				agents.At(i).Path = nullptr;
			}
	}


	class AgentsMover
	{
	public:
		bool DebugSyncMode = false;

		float DebugDurFindAgents = 0;
		float DebugDurFindAgentsMax = 0;
		float DebugDurMoveAgents = 0;
		float DebugDurMoveAgentsMax = 0;
		float DebugDurResolveCollisions = 0;
		float DebugDurResolveCollisionsMax = 0;
		float DebugDurUpdateCell = 0;
		float DebugDurUpdateCellMax = 0;
		int DebugRecordsCount = 0;


		void Init(
			const Cum<CuNodesMap>& map,
			const Cum<CuList<Agent>>& agents,
			float moveSpeed,
			float agentRadius,
			int parallelAgentsCount)
		{
			_map = map;
			_agents = agents;

			_moveSpeed = moveSpeed;
			_collisionDistSqr = (agentRadius * 2) * (agentRadius * 2);
			_collisionDist = agentRadius * 2;

			_procAgentsIndices.DAlloc(1, parallelAgentsCount);

			_hProcAgentsCount = static_cast<int*>(malloc(sizeof(int)));

			cudaMalloc(&_dProcAgentsCount, sizeof(int));

			cudaStreamCreate(&_stream);
			cudaEventCreate(&_evFindAgentsStart);
			cudaEventCreate(&_evFindAgentsEnd);
			cudaEventCreate(&_evMoveAgentsStart);
			cudaEventCreate(&_evMoveAgentsEnd);
			cudaEventCreate(&_evResolveCollisionsStart);
			cudaEventCreate(&_evResolveCollisionsEnd);
			cudaEventCreate(&_evUpdateCellsStart);
			cudaEventCreate(&_evUpdateCellsEnd);

			KernelMarkCollections<<<1, 1>>>(
				_procAgentsIndices.D(0), 
				parallelAgentsCount
			);
			cudaDeviceSynchronize();
			if (TryCatchCudaError("agents mover, mark collections"))
				return;
		}

		~AgentsMover()
		{
			//KernelFreeAgentsPaths<<<1, 1>>>(_agents.D(0));
			//CudaSyncAndCatch();

			cudaStreamDestroy(_stream);
			cudaEventDestroy(_evFindAgentsStart);
			cudaEventDestroy(_evFindAgentsEnd);
			cudaEventDestroy(_evMoveAgentsStart);
			cudaEventDestroy(_evMoveAgentsEnd);
			cudaEventDestroy(_evResolveCollisionsStart);
			cudaEventDestroy(_evResolveCollisionsEnd);
			cudaEventDestroy(_evUpdateCellsStart);
			cudaEventDestroy(_evUpdateCellsEnd);

			_procAgentsIndices.DFree();

			cudaFree(_dProcAgentsCount);

			free(_hProcAgentsCount);

			CudaCatch();
		}

		void AsyncPreMove()
		{
			KernelClearCollections<<<1, 1, 0, _stream>>> (
				_procAgentsIndices.D(0),
				_dProcAgentsCount
			);
			if (DebugSyncMode)
				CudaSyncAndCatch();
			FindMovingAgents();
			if (DebugSyncMode)
				CudaSyncAndCatch();
		}

		void AsyncMove(float deltaTime)
		{
			if (*_hProcAgentsCount == 0)
				return;

			MoveAgents(_moveSpeed * deltaTime);
			if (DebugSyncMode)
				CudaSyncAndCatch();
			ResolveCollisions();
			if (DebugSyncMode)
				CudaSyncAndCatch();
			UpdateCells();
			if (DebugSyncMode)
				CudaSyncAndCatch();
		}

		void PostMove()
		{
			if (*_hProcAgentsCount == 0)
				return;

			DebugRecordDurs();
		}

		void Sync()
		{
			CudaCheck(cudaStreamSynchronize(_stream), "agents mover");
		}

	private:
		float _moveSpeed = 0;
		float _collisionDistSqr = 0;
		float _collisionDist = 0;

		Cum<CuNodesMap> _map;
		Cum<CuList<Agent>> _agents;
		Cum<CuList<int>> _procAgentsIndices;

		int* _hProcAgentsCount = nullptr;
		int* _dProcAgentsCount = nullptr;

		cudaStream_t _stream{};
		cudaEvent_t _evFindAgentsStart{};
		cudaEvent_t _evFindAgentsEnd{};
		cudaEvent_t _evMoveAgentsStart{};
		cudaEvent_t _evMoveAgentsEnd{};
		cudaEvent_t _evResolveCollisionsStart{};
		cudaEvent_t _evResolveCollisionsEnd{};
		cudaEvent_t _evUpdateCellsStart{};
		cudaEvent_t _evUpdateCellsEnd{};


		void FindMovingAgents()
		{
			cudaEventRecord(_evFindAgentsStart, _stream);

			int threadsCount = _agents.H(0).Count();
			int threadsPerBlock = 128;
			int blocksCount = threadsCount / threadsPerBlock;
			if (blocksCount * threadsPerBlock < threadsCount)
				blocksCount += 1;

			KernelFindMovingAgents<<<blocksCount, threadsPerBlock, 0, _stream>>>(
				_agents.D(0),
				_procAgentsIndices.D(0),
				_dProcAgentsCount
			);
			cudaMemcpyAsync(_hProcAgentsCount, _dProcAgentsCount, sizeof(int), cudaMemcpyDeviceToHost, _stream);

			cudaEventRecord(_evFindAgentsEnd, _stream);
		}

		void MoveAgents(float agentStep)
		{
			cudaEventRecord(_evMoveAgentsStart, _stream);

			int threadsCount = *_hProcAgentsCount;
			int threadsPerBlock = 128;
			int blocksCount = threadsCount / threadsPerBlock;
			if (blocksCount * threadsPerBlock < threadsCount)
				blocksCount += 1;

			KernelMoveAgents<<<blocksCount, threadsPerBlock, 0, _stream>>>(
				_agents.D(0),
				_procAgentsIndices.D(0),
				agentStep
			);

			cudaEventRecord(_evMoveAgentsEnd, _stream);
		}

		void ResolveCollisions()
		{
			cudaEventRecord(_evResolveCollisionsStart, _stream);

			int blocksCount = _agents.H(0).Count();
			int threadsPerBlock = 128;

			KernelResolveCollisions<<<blocksCount, threadsPerBlock, 0, _stream>>>(
				_map.D(0),
				_agents.D(0),
				_collisionDistSqr,
				_collisionDist
			);

			cudaEventRecord(_evResolveCollisionsEnd, _stream);
		}

		void UpdateCells()
		{
			cudaEventRecord(_evUpdateCellsStart, _stream);

			int blocksCount = *_hProcAgentsCount;
			int threadsPerBlock = 128;

			KernelUpdatePathProgress<<<blocksCount, threadsPerBlock, 0, _stream>>>(
				_map.D(0),
				_agents.D(0),
				_procAgentsIndices.D(0)
			);

			cudaEventRecord(_evUpdateCellsEnd, _stream);
		}

		void DebugRecordDurs()
		{
			float t1 = 0;
			cudaEventElapsedTime(&t1, _evFindAgentsStart, _evFindAgentsEnd);
			DebugDurFindAgentsMax = std::max(t1, DebugDurFindAgentsMax);
			DebugDurFindAgents += t1;

			float t2 = 0;
			cudaEventElapsedTime(&t2, _evMoveAgentsStart, _evMoveAgentsEnd);
			DebugDurMoveAgentsMax = std::max(t2, DebugDurMoveAgentsMax);
			DebugDurMoveAgents += t2;

			float t3 = 0;
			cudaEventElapsedTime(&t3, _evResolveCollisionsStart, _evResolveCollisionsEnd);
			DebugDurResolveCollisionsMax = std::max(t3, DebugDurResolveCollisionsMax);
			DebugDurResolveCollisions += t3;

			float t4 = 0;
			cudaEventElapsedTime(&t4, _evUpdateCellsStart, _evUpdateCellsEnd);
			DebugDurUpdateCellMax = std::max(t4, DebugDurUpdateCellMax);
			DebugDurUpdateCell += t4;

			DebugRecordsCount += 1;
		}
	};
}
