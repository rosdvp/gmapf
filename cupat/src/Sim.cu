#include "../include/Sim.h"

#include <chrono>
#include <iostream>

#include "../include/misc/Cum.h"
#include "../include/misc/CuMatrix.h"
#include "../include/misc/CuList.h"
#include "../include/AgentsMover.h"
#include "../include/PathAStarCPU.h"
#include "../include/PathFinder.h"
#include "../include/Helpers.h"

using namespace cupat;

Sim::~Sim()
{
	delete _pathFinder;
	delete _agentsMover;

	_map->HFree();
	_map->DFree();
	delete _map;

	_agents->HFree();
	_agents->DFree();
	delete _agents;

	std::cout << "[cupat] sim destroyed" << std::endl;
}

void Sim::Init(const ConfigSim& config)
{
	_config = config;

	cudaSetDevice(0);
	TryCatchCudaError("set device");
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 32);

	_mapDesc = MapDesc(config.MapCellSize, config.MapCountX, config.MapCountY);

	_map = new Cum<CuMatrix<int>>();
	_map->HAllocAndMark(1, config.MapCountX, config.MapCountY);
	auto hMap = _map->H(0);
	for (int x = 0; x < config.MapCountX; x++)
		for (int y = 0; y < config.MapCountY; y++)
			hMap.At(x, y) = 0;

	_agents = new Cum<CuList<Agent>>();
	_agents->HAllocAndMark(1, config.AgentsCount);
	auto hAgents = _agents->H(0);
	for (int i = 0; i < config.AgentsCount; i++)
		hAgents.Add({});
}

void Sim::SetAgentInitialPos(int agentId, const V2Float& currPos)
{
	if (!_mapDesc.IsValidPos(currPos))
		throw std::exception(("initial pos " + currPos.ToString() + " is invalid").c_str());

	Agent& agent = _agents->H(0).At(agentId);
	agent.CurrPos = currPos;
	agent.CurrCell = _mapDesc.PosToCell(currPos);
	agent.IsTargetReached = true;
}

void Sim::SetAgentTargPos(int agentId, const V2Float& targPos)
{
	if (!_mapDesc.IsValidPos(targPos))
		throw std::exception(("initial pos " + targPos.ToString() + " is invalid").c_str());

	Agent& agent = _agents->H(0).At(agentId);
	agent.TargPos = targPos;
	agent.TargCell = _mapDesc.PosToCell(targPos);
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

	Agent& agent = _agents->H(0).At(agentId);
	agent.Path = cumPath.DPtr(0);
	agent.IsNewPathRequested = false;
	agent.PathStepIdx = 0;
	agent.PathNextCell = path[0];
}

void Sim::SetObstacle(const V2Int& cell)
{
	_map->H(0).At(cell) = -1;
}

void Sim::Start(bool isDebugSyncMode)
{
	_map->CopyToDevice();
	TryCatchCudaError("allocate map");
	_agents->CopyToDevice();
	TryCatchCudaError("allocate agents");

	_pathFinder = new PathFinder();
	_pathFinder->DebugSyncMode = isDebugSyncMode;
	_pathFinder->Init(
		*_map,
		*_agents,
		_config.PathFinderParallelAgents,
		_config.PathFinderThreadsPerAgents,
		_config.PathFinderEachQueueCapacity,
		_config.PathFinderHeuristicK,
		_config.PathStorageCapacityK
	);

	_agentsMover = new AgentsMover();
	_agentsMover->DebugSyncMode = isDebugSyncMode;
	_agentsMover->Init(
		_mapDesc,
		*_map,
		*_agents,
		_config.AgentSpeed,
		_config.AgentRadius,
		_config.AgentsCount
	);
}

void Sim::DoStep(float deltaTime)
{
	TIME_STAMP(tStart);

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

	_agents->CopyToHost();

	auto step = TIME_DIFF_MS(tStart);
	_debugDurStep += step;
	_debugDurStepMax = std::max(step, _debugDurStepMax);
	_debugStepsCount += 1;
}

void Sim::DoStepOnlyFinder()
{
	TIME_STAMP(tStart);

	_pathFinder->AsyncPreFind();
	_pathFinder->Sync();
	_pathFinder->AsyncFind();
	_pathFinder->Sync();
	_pathFinder->PostFind();

	auto step = TIME_DIFF_MS(tStart);
	_debugDurStep += step;
	_debugDurStepMax = std::max(step, _debugDurStepMax);
	_debugStepsCount += 1;
}

void Sim::DoStepOnlyMover(float deltaTime)
{
	TIME_STAMP(tStart);

	_agentsMover->AsyncPreMove();
	_agentsMover->Sync();
	_agentsMover->AsyncMove(deltaTime);
	_agentsMover->Sync();
	_agentsMover->PostMove();

	_agents->CopyToHost();

	auto step = TIME_DIFF_MS(tStart);
	_debugDurStep += step;
	_debugDurStepMax = std::max(step, _debugDurStepMax);
	_debugStepsCount += 1;
}

const V2Float& Sim::GetAgentPos(int agentId)
{
	return _agents->H(0).At(agentId).CurrPos;
}

void Sim::DebugDump() const
{
	printf("----------------------\n");

	printf("sim step ms, avg: %f, max: %f, sum: %f\n", _debugDurStep / _debugStepsCount, _debugDurStepMax, _debugDurStep);

	printf("\n");

	int count = _pathFinder->DebugRecordsCount;
	printf("path finder:\n");
	printf("clear collections ms, avg: %f max: %f, sum: %f\n", _pathFinder->DebugDurClearCollections / count, _pathFinder->DebugDurClearCollectionsMax, _pathFinder->DebugDurClearCollections);
	printf("prepare search ms, avg: %f max: %f, sum: %f\n", _pathFinder->DebugDurPrepareSearch / count, _pathFinder->DebugDurPrepareSearchMax,  _pathFinder->DebugDurPrepareSearch);
	printf("search ms, avg: %f max: %f, sum: %f\n", _pathFinder->DebugDurSearch / count, _pathFinder->DebugDurSearchMax, _pathFinder->DebugDurSearch);
	printf("build paths ms, avg: %f max: %f, sum: %f\n", _pathFinder->DebugDurBuildPaths / count, _pathFinder->DebugDurBuildPathsMax, _pathFinder->DebugDurBuildPaths);
	printf("attach paths ms, avg: %f max: %f, sum: %f\n", _pathFinder->DebugDurAttachPaths / count, _pathFinder->DebugDurAttachPathsMax,  _pathFinder->DebugDurAttachPaths);

	printf("\n");

	count = _agentsMover->DebugRecordsCount;
	printf("agents mover:\n");
	printf("find moving agents ms, avg: %f max: %f, sum: %f\n", _agentsMover->DebugDurFindAgents / count, _agentsMover->DebugDurFindAgentsMax, _agentsMover->DebugDurFindAgents);
	printf("move agents ms, avg: %f max: %f, sum: %f\n", _agentsMover->DebugDurMoveAgents / count, _agentsMover->DebugDurMoveAgentsMax, _agentsMover->DebugDurMoveAgents);
	printf("resolve collisions ms, avg: %f max: %f, sum: %f\n", _agentsMover->DebugDurResolveCollisions / count, _agentsMover->DebugDurResolveCollisionsMax, _agentsMover->DebugDurResolveCollisions);
	printf("update cells ms, avg: %f max: %f, sum: %f\n", _agentsMover->DebugDurUpdateCell / count, _agentsMover->DebugDurUpdateCellMax, _agentsMover->DebugDurUpdateCell);

	printf("----------------------\n");
}
