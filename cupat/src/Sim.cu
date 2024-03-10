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

void Sim::Init(const ConfigSim& config)
{
	_config = config;

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

void Sim::Destroy()
{
	std::cout << "[cupat] sim destroying.." << std::endl;

	CudaSyncAndCatch();

	delete _agentsMover;
	delete _pathFinder;

	_map->HFree();
	_map->DFree();
	delete _map;

	_agents->HFree();
	_agents->DFree();
	delete _agents;

	CudaCatch();

	CuDriverCatch(cuCtxDestroy(_cuContext));

	std::cout << "[cupat] sim destroyed" << std::endl;
}

void Sim::SetAgentInitialPos(int agentId, const V2Float& currPos)
{
	if (!_mapDesc.IsValidPos(currPos))
		throw std::exception(("initial pos " + currPos.ToString() + " is invalid").c_str());

	Agent& agent = _agents->H(0).At(agentId);
	agent.CurrPos = currPos;
	agent.CurrCell = _mapDesc.PosToCell(currPos);
	agent.IsTargetReached = true;

	if (!_map->H(0).IsValid(agent.CurrCell))
		throw std::exception(("initial cell " + agent.CurrCell.ToString() + " is invalid").c_str());
	if (_map->H(0).At(agent.CurrCell) != 0)
		throw std::exception(("initial cell " + agent.CurrCell.ToString() + " is busy").c_str());
}

void Sim::SetAgentTargPos(int agentId, const V2Float& targPos)
{
	if (!_mapDesc.IsValidPos(targPos))
		throw std::exception(("target pos " + targPos.ToString() + " is invalid").c_str());

	Agent& agent = _agents->H(0).At(agentId);
	agent.TargPos = targPos;
	agent.TargCell = _mapDesc.PosToCell(targPos);
	agent.IsNewPathRequested = true;
	agent.IsTargetReached = false;

	if (!_map->H(0).IsValid(agent.CurrCell))
		throw std::exception(("target cell " + agent.TargCell.ToString() + " is invalid").c_str());
	if (_map->H(0).At(agent.CurrCell) != 0)
		throw std::exception(("target cell " + agent.TargCell.ToString() + " is busy").c_str());
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
	cudaSetDevice(0);
	CudaCatch();

	CUdevice device;
	CuDriverCatch(cuDeviceGet(&device, 0));
	CuDriverCatch(cuCtxCreate(&_cuContext, 0, device));
	CuDriverCatch(cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE, (size_t)1024 * (size_t)1024 * (size_t)128));

	//cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 32);
	//CudaCatch();

	CudaCatch();
	_map->CopyToDevice();
	CudaCatch();
	_agents->CopyToDevice();
	CudaCatch();

	_pathFinder = new PathFinder();
	_pathFinder->DebugSyncMode = isDebugSyncMode;
	_pathFinder->Init(
		*_map,
		*_agents,
		_config.PathFinderParallelAgents,
		_config.PathFinderThreadsPerAgents,
		_config.PathFinderQueueCapacity,
		_config.PathFinderHeuristicK
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

	CuDriverCatch(cuCtxPopCurrent(nullptr));
}

void Sim::DoStep(float deltaTime)
{
	TIME_STAMP(tStart);

	CuDriverCatch(cuCtxPushCurrent(_cuContext));

	_pathFinder->AsyncPreFind();
	_agentsMover->AsyncPreMove();

	_pathFinder->Sync();
	_agentsMover->Sync();

	auto durPre = TIME_DIFF_MS(tStart);
	TIME_STAMP(tMain);

	_pathFinder->AsyncFind();
	_agentsMover->AsyncMove(deltaTime);

	_pathFinder->Sync();
	_agentsMover->Sync();

	auto durMain = TIME_DIFF_MS(tMain);
	TIME_STAMP(tPost);

	_pathFinder->PostFind();
	_agentsMover->PostMove();

	auto durPost = TIME_DIFF_MS(tPost);
	TIME_STAMP(tCopy);

	_agents->CopyToHost();

	auto durCopy = TIME_DIFF_MS(tCopy);

	CuDriverCatch(cuCtxPopCurrent(nullptr));

	auto durStep = TIME_DIFF_MS(tStart);

	TIME_APPLY_RECORD(durStep, _debugDurStepSum, _debugDurStepMax);
	TIME_APPLY_RECORD(durPre, _debugDurStepPreSum, _debugDurStepPreMax);
	TIME_APPLY_RECORD(durMain, _debugDurStepMainSum, _debugDurStepMainMax);
	TIME_APPLY_RECORD(durPost, _debugDurStepPostSum, _debugDurStepPostMax);
	TIME_APPLY_RECORD(durCopy, _debugDurStepCopySum, _debugDurStepCopyMax);
	_debugStepsCount += 1;
}

void Sim::DoStepOnlyFinder()
{
	TIME_STAMP(tStart);

	CuDriverCatch(cuCtxPushCurrent(_cuContext));

	_pathFinder->AsyncPreFind();
	_pathFinder->Sync();

	auto durPre = TIME_DIFF_MS(tStart);
	TIME_STAMP(tMain);

	_pathFinder->AsyncFind();
	_pathFinder->Sync();

	auto durMain = TIME_DIFF_MS(tMain);
	TIME_STAMP(tPost);

	_pathFinder->PostFind();

	auto durPost = TIME_DIFF_MS(tPost);

	CuDriverCatch(cuCtxPopCurrent(nullptr));

	auto durStep = TIME_DIFF_MS(tStart);

	TIME_APPLY_RECORD(durStep, _debugDurStepSum, _debugDurStepMax);
	TIME_APPLY_RECORD(durPre, _debugDurStepPreSum, _debugDurStepPreMax);
	TIME_APPLY_RECORD(durMain, _debugDurStepMainSum, _debugDurStepMainMax);
	TIME_APPLY_RECORD(durPost, _debugDurStepPostSum, _debugDurStepPostMax);
	_debugStepsCount += 1;
}

void Sim::DoStepOnlyMover(float deltaTime)
{
	TIME_STAMP(tStart);

	CuDriverCatch(cuCtxPushCurrent(_cuContext));

	_agentsMover->AsyncPreMove();
	_agentsMover->Sync();
	_agentsMover->AsyncMove(deltaTime);
	_agentsMover->Sync();
	_agentsMover->PostMove();

	_agents->CopyToHost();

	CuDriverCatch(cuCtxPopCurrent(nullptr));

	auto step = TIME_DIFF_MS(tStart);
	_debugDurStepSum += step;
	_debugDurStepMax = std::max(step, _debugDurStepMax);
	_debugStepsCount += 1;
}

const V2Float& Sim::GetAgentPos(int agentId)
{
	return _agents->H(0).At(agentId).CurrPos;
}

void Sim::DebugDump() const
{
	std::cout << "----------------------" << std::endl;

	TIME_STD_OUT("sim step", _debugDurStepSum, _debugDurStepMax, _debugStepsCount);
	TIME_STD_OUT("sim step pre", _debugDurStepPreSum, _debugDurStepPreMax, _debugStepsCount);
	TIME_STD_OUT("sim step main", _debugDurStepMainSum, _debugDurStepMainMax, _debugStepsCount);
	TIME_STD_OUT("sim step post", _debugDurStepPostSum, _debugDurStepPostMax, _debugStepsCount);
	TIME_STD_OUT("sim step copy", _debugDurStepCopySum, _debugDurStepCopyMax, _debugStepsCount);

	std::cout << std::endl;

	int count = _pathFinder->DebugRecordsCount;
	printf("path finder:\n");
	printf("prepare search ms, avg: %f max: %f, sum: %f\n", _pathFinder->DebugDurPrepareSearch / count, _pathFinder->DebugDurPrepareSearchMax,  _pathFinder->DebugDurPrepareSearch);
	printf("clear collections ms, avg: %f max: %f, sum: %f\n", _pathFinder->DebugDurClearCollections / count, _pathFinder->DebugDurClearCollectionsMax, _pathFinder->DebugDurClearCollections);
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
