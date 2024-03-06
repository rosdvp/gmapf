#pragma once

#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Agent.h"
#include "Helpers.h"
#include "MapDesc.h"
#include "misc/CuList.h"
#include "misc/Cum.h"
#include "misc/CuMatrix.h"

namespace cupat
{
	__global__ void KernelMarkCollections(
		CuList<int> agentsIndices,
		int parallelAgentsCount)
	{
		agentsIndices.Mark(parallelAgentsCount);
	}

	__global__ void KernelClearCollections(
		CuList<int> agentsIndices,
		int* procAgentsCount)
	{
		agentsIndices.RemoveAll();
		*procAgentsCount = 0;
	}

	__global__ void KernelFindMovingAgents(
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
			if (agent.IsNewPathRequested || agent.IsTargetReached)
				continue;
			agentsIndices.AddAtomic(i);
			atomicAdd(procAgentsCount, 1);

			//printf("moving agent idx: %d\n", i);
		}
	}

	__global__ void KernelMoveAgents(
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

		V2Float delta = agent.PathNextCell - agent.CurrCell;
		delta = delta * agentStep;

		agent.CurrPos += delta;

		//printf("delta (%f, %f)\n", delta.X, delta.Y);
	}

	__global__ void KernelResolveCollisions(
		CuList<Agent> agents,
		CuList<int> agentsIndices,
		float collisionDistSqr)
	{
		int tid = threadIdx.x;
		int bid = blockIdx.x;
		int threadsCount = blockDim.x;

		int agentIdx = agentsIndices.At(bid);
		Agent& agent = agents.At(agentIdx);

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
			if (i == agentIdx)
				continue;

			Agent& other = agents.At(i);
			V2Float delta = agent.CurrPos - other.CurrPos;
			float distSqr = delta.GetLengthSqr();
			if (distSqr < collisionDistSqr)
			{
				delta = delta.GetNorm();
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
		avoidStep = avoidStep.GetRotated(10).GetNorm();

		agent.CurrPos += avoidStep;

		//printf("avoid step (%f, %f)\n", avoidStep.X, avoidStep.Y);
	}

	__global__ void KernelUpdateCells(
		MapDesc mapDesc,
		CuList<Agent> agents,
		CuList<int> agentsIndices)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= agentsIndices.Count())
			return;
		int agentIdx = agentsIndices.At(tid);
		Agent& agent = agents.At(agentIdx);

		agent.CurrCell = mapDesc.PosToCell(agent.CurrPos);
		if (agent.CurrCell == agent.TargCell)
		{
			agent.IsTargetReached = true;
			return;
		}
		if (agent.CurrCell == agent.PathNextCell)
		{
			CuList<V2Int> path(agent.Path);
			agent.PathStepIdx += 1;
			if (agent.PathStepIdx == path.Count())
			{
				agent.IsTargetReached = true;
				return;
			}
			agent.PathNextCell = path.At(agent.PathStepIdx);
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
			const MapDesc& mapDesc,
			Cum<CuMatrix<int>> map,
			Cum<CuList<Agent>> agents,
			float moveSpeed,
			float agentRadius,
			int parallelAgentsCount)
		{
			_mapDesc = mapDesc;
			_map = map;
			_agents = agents;

			_moveSpeed = moveSpeed;
			_collisionDistSqr = (agentRadius * 2) * (agentRadius * 2);

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
			_procAgentsIndices.DFree();

			free(_hProcAgentsCount);
			cudaFree(_dProcAgentsCount);
		}

		void AsyncPreMove()
		{
			KernelClearCollections<<<1, 1, 0, _stream>>> (
				_procAgentsIndices.D(0),
				_dProcAgentsCount
			);
			if (DebugSyncMode)
				CudaCatch();
			FindMovingAgents();
			if (DebugSyncMode)
				CudaCatch();
		}

		void AsyncMove(float deltaTime)
		{
			if (*_hProcAgentsCount == 0)
				return;

			MoveAgents(_moveSpeed * deltaTime);
			if (DebugSyncMode)
				CudaCatch();
			ResolveCollisions();
			if (DebugSyncMode)
				CudaCatch();
			UpdateCells();
			if (DebugSyncMode)
				CudaCatch();
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
		MapDesc _mapDesc;
		float _moveSpeed = 0;
		float _collisionDistSqr = 0;

		Cum<CuMatrix<int>> _map;
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

			int blocksCount = *_hProcAgentsCount;
			int threadsPerBlock = 128;

			KernelResolveCollisions<<<blocksCount, threadsPerBlock, 0, _stream>>>(
				_agents.D(0),
				_procAgentsIndices.D(0),
				_collisionDistSqr
			);

			cudaEventRecord(_evResolveCollisionsEnd, _stream);
		}

		void UpdateCells()
		{
			cudaEventRecord(_evUpdateCellsStart, _stream);

			int blocksCount = *_hProcAgentsCount;
			int threadsPerBlock = 128;

			KernelUpdateCells<<<blocksCount, threadsPerBlock, 0, _stream>>>(
				_mapDesc,
				_agents.D(0),
				_procAgentsIndices.D(0)
			);

			cudaEventRecord(_evUpdateCellsEnd, _stream);
		}

		void DebugRecordDurs()
		{
			float temp = 0;

			cudaEventElapsedTime(&temp, _evFindAgentsStart, _evFindAgentsEnd);
			DebugDurFindAgentsMax = std::max(temp, DebugDurFindAgentsMax);
			DebugDurFindAgents += temp;

			cudaEventElapsedTime(&temp, _evMoveAgentsStart, _evMoveAgentsEnd);
			DebugDurMoveAgentsMax = std::max(temp, DebugDurMoveAgentsMax);
			DebugDurMoveAgents += temp;

			cudaEventElapsedTime(&temp, _evResolveCollisionsStart, _evResolveCollisionsEnd);
			DebugDurResolveCollisionsMax = std::max(temp, DebugDurResolveCollisionsMax);
			DebugDurResolveCollisions += temp;

			cudaEventElapsedTime(&temp, _evUpdateCellsStart, _evUpdateCellsEnd);
			DebugDurUpdateCellMax = std::max(temp, DebugDurUpdateCellMax);
			DebugDurUpdateCell += temp;

			DebugRecordsCount += 1;
		}
	};
}
