#include "../include/Sim.h"

#include <chrono>
#include <iostream>

#include "AgentsMover.h"
#include "PathAStarCPU.h"
#include "PathFinder.h"
#include "../include/Helpers.h"

using namespace cupat;

Sim::~Sim()
{
	_map.HFree();
	_map.DFree();

	_agents.HFree();
	_agents.DFree();

	std::cout << "[cupat] sim destroyed" << std::endl;
}

void Sim::Init(const ConfigSim& config)
{
	_config = config;

	cudaSetDevice(0);
	TryCatchCudaError("set device");
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 32);

	_mapDesc = MapDesc(config.MapCellSize, config.MapCountX, config.MapCountY);

	_map.HAllocAndMark(1, config.MapCountX, config.MapCountY);
	auto hMap = _map.H(0);
	for (int x = 0; x < config.MapCountX; x++)
		for (int y = 0; y < config.MapCountY; y++)
			hMap.At(x, y) = 0;

	_agents.HAllocAndMark(1, config.AgentsCount);
	auto hAgents = _agents.H(0);
	for (int i = 0; i < config.AgentsCount; i++)
		hAgents.Add({});
}

void Sim::SetAgentInitialPos(int agentId, const V2Float& currPos)
{
	Agent& agent = _agents.H(0).At(agentId);
	agent.CurrPos = currPos;
	agent.CurrCell = PosToCell(currPos);
	agent.IsTargetReached = true;
}

void Sim::SetAgentTargPos(int agentId, const V2Float& targPos)
{
	Agent& agent = _agents.H(0).At(agentId);
	agent.TargPos = targPos;
	agent.TargCell = PosToCell(targPos);
	agent.IsNewPathRequested = true;
	agent.IsTargetReached = false;
}

void Sim::DebugSetAgentPath(int agentId, const std::vector<V2Int>& path)
{
	Cum<CuList<V2Int>> cumPath;
	cumPath.HAllocAndMark(1, path.size());
	for (auto cell : path)
		cumPath.H(0).Add(cell);
	cumPath.CopyToDevice();

	Agent& agent = _agents.H(0).At(agentId);
	agent.Path = cumPath.DPtr(0);
	agent.IsNewPathRequested = false;
	agent.PathIdx = 0;
	agent.PathNextCell = path[0];
}

void Sim::SetObstacle(const V2Int& cell)
{
	_map.H(0).At(cell) = -1;
}

void Sim::Start()
{
	_map.CopyToDevice();
	TryCatchCudaError("allocate map");
	_agents.CopyToDevice();
	TryCatchCudaError("allocate agents");

	_pathFinder = new PathFinder();
	_pathFinder->Init(
		_map,
		_agents,
		_config.PathFinderParallelAgents,
		_config.PathFinderThreadsPerAgents,
		_config.PathFinderEachQueueCapacity,
		_config.PathFinderHeuristicK
	);

	_agentsMover = new AgentsMover();
	_agentsMover->Init(
		_mapDesc,
		_map,
		_agents,
		_config.AgentSpeed,
		_config.AgentRadius,
		_config.AgentsCount
	);
}

void Sim::DoStep(float deltaTime)
{
	_pathFinder->AsyncPreFind();
	_agentsMover->AsyncPreMove();

	_pathFinder->Sync();
	_agentsMover->Sync();

	_pathFinder->AsyncFind();
	_agentsMover->AsyncMove(deltaTime);

	_pathFinder->Sync();
	_agentsMover->Sync();

	_pathFinder->PostFind();
	_agentsMover->PostMove();

	_agents.CopyToHost();
}

void Sim::DoStepOnlyFinder()
{
	_pathFinder->AsyncPreFind();
	_pathFinder->Sync();
	_pathFinder->AsyncFind();
	_pathFinder->Sync();
	_pathFinder->PostFind();
}

void Sim::DoStepOnlyMover(float deltaTime)
{
	_agentsMover->AsyncPreMove();
	_agentsMover->Sync();
	_agentsMover->AsyncMove(deltaTime);
	_agentsMover->Sync();
	_agentsMover->PostMove();

	_agents.CopyToHost();
}

const V2Float& Sim::GetAgentPos(int agentId)
{
	return _agents.H(0).At(agentId).CurrPos;
}

void Sim::DebugDump() const
{
	int count = _pathFinder->DebugRecordsCount;
	printf("path finder:\n");
	printf("clear collections ms: %f (%f)\n", _pathFinder->DebugDurClearCollections / count, _pathFinder->DebugDurClearCollections);
	printf("prepare search ms: %f (%f)\n", _pathFinder->DebugDurPrepareSearch / count, _pathFinder->DebugDurPrepareSearch);
	printf("search ms: %f (%f)\n", _pathFinder->DebugDurSearch / count, _pathFinder->DebugDurSearch);
	printf("build paths ms: %f (%f)\n", _pathFinder->DebugDurBuildPaths / count, _pathFinder->DebugDurBuildPaths);
	printf("attach paths ms: %f (%f)\n", _pathFinder->DebugDurAttachPaths / count, _pathFinder->DebugDurAttachPaths);

	count = _agentsMover->DebugRecordsCount;
	printf("agents mover:\n");
	printf("find moving agents ms: %f (%f)\n", _agentsMover->DebugDurFindAgents / count, _agentsMover->DebugDurFindAgents);
	printf("move agents ms: %f (%f)\n", _agentsMover->DebugDurMoveAgents / count, _agentsMover->DebugDurMoveAgents);
	printf("resolve collisions ms: %f (%f)\n", _agentsMover->DebugDurResolveCollisions / count, _agentsMover->DebugDurResolveCollisions);
	printf("update cells ms: %f (%f)\n", _agentsMover->DebugDurUpdateCell / count, _agentsMover->DebugDurUpdateCell);
}

V2Int Sim::PosToCell(const V2Float& pos) const
{
	auto x = static_cast<int>(pos.X / _config.MapCellSize);
	auto y = static_cast<int>(pos.Y / _config.MapCellSize);
	return { x, y };
}
