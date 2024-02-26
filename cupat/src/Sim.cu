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
	CumMatrix<int>::Free(_map, 1);
	CumList<Agent>::Free(_agents, 1);

	std::cout << "[cupat] sim destroyed" << std::endl;
}

void Sim::Init(const ConfigSim& config)
{
	_config = config;

	cudaSetDevice(0);
	TryCatchCudaError("set device");
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 32);

	_map = CumMatrix<int>::New(1, config.MapCountX, config.MapCountY);
	TryCatchCudaError("allocate map");
	for (int x = 0; x < config.MapCountX; x++)
		for (int y = 0; y < config.MapCountY; y++)
			_map->At(x, y) = 0;

	_agents = CumList<Agent>::New(1, config.AgentsCount);
	for (int i = 0; i < config.AgentsCount; i++)
		_agents->Add({});
	TryCatchCudaError("allocate agents");
}

void Sim::SetAgentInitialPos(int agentId, const V2Float& currPos)
{
	Agent& agent = _agents->At(agentId);
	agent.CurrPos = currPos;
	agent.CurrCell = PosToCell(currPos);
}

void Sim::SetAgentTargPos(int agentId, const V2Float& targPos)
{
	Agent& agent = _agents->At(agentId);
	agent.TargPos = targPos;
	agent.TargCell = PosToCell(targPos);
	agent.IsNewPathRequested = true;
	agent.PathCellIdx = 0;
}

void Sim::SetObstacle(const V2Int& cell)
{
	_map->At(cell) = -1;
}

void Sim::Start()
{
}

bool Sim::DoStep(float deltaTime)
{
	int agentsCount = _agents->Count();

	KernelProcessAgentInput input;

	//KernelProcessAgent<<<1, agentsCount>>>(input);

	CpuFindPathAStarInput cpuInp;
	cpuInp.Map = _map;
	cpuInp.Agents = _agents;
	cpuInp.AgentId = 0;

	int threadsCount = 1;

	FindPathAStarMixedInput mixedInp;
	mixedInp.Map = _map;
	mixedInp.Agents = _agents;
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

	return true;
}

const V2Float& Sim::GetAgentPos(int agentId) const
{
	return _agents->At(agentId).CurrPos;
}

V2Int Sim::PosToCell(const V2Float& pos) const
{
	auto x = static_cast<int>(pos.X / _config.MapCellSize);
	auto y = static_cast<int>(pos.Y / _config.MapCellSize);
	return { x, y };
}