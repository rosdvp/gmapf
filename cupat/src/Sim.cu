#include "../include/Sim.h"

#include <chrono>
#include <iostream>

#include "../include/Helpers.h"
#include "../include/kernels/KernelProcessAgent.h"
#include "../include/kernels/PathAStarCPU.h"
#include "../include/kernels/PathAStarMixed.h"

using namespace cupat;

Sim::~Sim()
{
	_map.HFree();
	_agents.FreeOnHost();

	if (_dRawMap != nullptr)
		cudaFree(_dRawMap);
	if (_dRawAgents != nullptr)
		cudaFree(_dRawAgents);

	std::cout << "[cupat] sim destroyed" << std::endl;
}

void Sim::Init(const ConfigSim& config)
{
	_config = config;

	cudaSetDevice(0);
	TryCatchCudaError("set device");
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 32);

	_map.HAlloc(config.MapCountX, config.MapCountY);
	for (int x = 0; x < config.MapCountX; x++)
		for (int y = 0; y < config.MapCountY; y++)
			_map.At({ x, y }) = 0;

	_agents.AllocOnHost(config.AgentsCount);
}

void Sim::SetAgentInitialPos(int agentId, const V2Float& currPos)
{
	Agent& agent = _agents.At(agentId);
	agent.CurrPos = currPos;
	agent.CurrCell = PosToCell(currPos);
}

void Sim::SetAgentTargPos(int agentId, const V2Float& targPos)
{
	Agent& agent = _agents.At(agentId);
	agent.TargPos = targPos;
	agent.TargCell = PosToCell(targPos);
	agent.IsNewPathRequested = true;
	agent.PathCellIdx = 0;
}

void Sim::SetObstacle(const V2Int& cell)
{
	_map.At(cell) = -1;
}

void Sim::Start()
{
	_dRawMap = _map.AllocOnDeviceAndCopyFromHost();
	TryCatchCudaError("allocate map");
	_dRawAgents = _agents.AllocOnDeviceAndCopyFromHost();
	TryCatchCudaError("allocate agents");
}

bool Sim::DoStep(float deltaTime)
{
	int agentsCount = _agents.GetCount();

	KernelProcessAgentInput input;
	input.ConfigSim = _config;
	input.DMap = _dRawMap;
	input.DAgents = _dRawAgents;
	input.DeltaTime = deltaTime;

	//KernelProcessAgent<<<1, agentsCount>>>(input);

	CpuFindPathAStarInput cpuInp;
	cpuInp.Map = _map.GetRawPtr();
	cpuInp.Agents = _agents.GetRawPtr();
	cpuInp.AgentId = 0;

	int threadsCount = 1;

	FindPathAStarMixedInput mixedInp;
	mixedInp.Map = _map.GetRawPtr();
	mixedInp.DMap = _dRawMap;
	mixedInp.DAgents = _agents.GetRawPtr();
	mixedInp.AgentId = 0;

	auto time0 = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < 1; i++)
	{
		//KernelFindPathAStar<<<1, 5>>>(inp);
		//KernelFindPathAStarMono<<<1, 1 >>>(inp);
		FindPathAStarMixed(mixedInp);
		//CpuFindPathAStar(cpuInp);
		//cudaDeviceSynchronize();
	}

	auto time1 = std::chrono::high_resolution_clock::now();
	auto timeDelta = std::chrono::duration_cast<std::chrono::microseconds>(time1 - time0).count();
	std::cout << "time: " << timeDelta / 1000.0f << std::endl;


	cudaMemcpy(_agents.GetRawPtr(), _dRawAgents, _agents.GetRawSize(), cudaMemcpyDeviceToHost);
	if (TryCatchCudaError("copy agents from device"))
		return false;

	return true;
}

const V2Float& Sim::GetAgentPos(int agentId) const
{
	return _agents.At(agentId).CurrPos;
}

V2Int Sim::PosToCell(const V2Float& pos) const
{
	auto x = static_cast<int>(pos.X / _config.MapCellSize);
	auto y = static_cast<int>(pos.Y / _config.MapCellSize);
	return { x, y };
}