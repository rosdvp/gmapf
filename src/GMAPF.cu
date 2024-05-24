#include "../include/gmapf/GMAPF.h"

#include <chrono>
#include <iostream>

#include "../include/gmapf/misc/Cum.h"
#include "../include/gmapf/misc/CuList.h"
#include "../include/gmapf/AgentsMover.h"
#include "../include/gmapf/PathFinderCPU.h"
#include "../include/gmapf/PathFinder.h"
#include "../include/gmapf/PathFinderSTMA.h"
#include "../include/gmapf/Helpers.h"

using namespace gmapf;

gmapf::GMAPF::GMAPF() {}

gmapf::GMAPF::~GMAPF()
{
	if (!_isInited)
		return;
	std::cout << "[gmapf] destroying.." << std::endl;

	if (_isStepGoing)
		WaitStepEnd();

	if (_isStarted)
		CudaSyncAndCatch();

	delete _agentsMover;
	delete _pathFinder;

	if (_map != nullptr)
	{
		_map->HFree();
		_map->DFree();
		delete _map;
	}
	if (_agents != nullptr)
	{
		_agents->HFree();
		_agents->DFree();
		delete _agents;
	}

	if (_isStarted)
		CudaCatch();

	if (_isStarted)
		CuDriverCatch(cuCtxDestroy(_cuContext));

	_isInited = false;
	_isStarted = false;

	std::cout << "[gmapf] destroyed" << std::endl;
}

void GMAPF::Init(const Config& config)
{
	std::cout << "[gmapf] initializing.." << std::endl;

	_config = config;

	_agents = new Cum<CuList<Agent>>();
	_agents->HAllocAndMark(1, config.AgentsMaxCount);

	_isInited = true;

	std::cout << "[gmapf] initialized" << std::endl;
}

void GMAPF::ManualStart()
{
	if (_isStarted)
		throw std::exception("already started");

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
	_isAnyAgentModifiedFromCpu = false;
	CudaCatch();

	_pathFinder = new PathFinder();
	_pathFinder->IsDebug = _config.IsDebug;
	_pathFinder->IsProfiler = _config.IsProfiler;
	_pathFinder->Init(
		*_map,
		*_agents,
		_config.PathFinderParallelAgents,
		_config.PathFinderThreadsPerAgents,
		_config.PathFinderQueueCapacity,
		_config.PathFinderHeuristicK
	);
	//_pathFinder = new STMA::PathFinderSTMA();
	//_pathFinder->DebugSyncMode = isDebugSyncMode;
	//_pathFinder->Init(
	//	*_map,
	//	*_agents,
	//	_config.PathFinderParallelAgents,
	//	_config.PathFinderQueueCapacity,
	//	_config.PathFinderHeuristicK
	//);

	_agentsMover = new AgentsMover();
	_agentsMover->IsDebug = _config.IsDebug;
	_agentsMover->IsProfiler = _config.IsProfiler;
	_agentsMover->Init(
		*_map,
		*_agents,
		_config.AgentSpeed,
		_config.AgentRadius,
		_config.AgentsMaxCount
	);

	CuDriverCatch(cuCtxPopCurrent(nullptr));

	_isStarted = true;
}


void gmapf::GMAPF::FillMapStart(const V2Float& mapSize)
{
	_cdtVertices.clear();
	_cdtEdges.clear();
	_cdtVertices.push_back({ 0, 0 });
	_cdtVertices.push_back({ 0, mapSize.Y });
	_cdtVertices.push_back({ mapSize.X, mapSize.Y });
	_cdtVertices.push_back({ mapSize.X, 0 });
	_cdtEdges.emplace_back(0, 1);
	_cdtEdges.emplace_back(1, 2);
	_cdtEdges.emplace_back(2, 3);
	_cdtEdges.emplace_back(3, 0);
}

void gmapf::GMAPF::FillMapObstacle(const std::vector<V2Float>& vertices)
{
	int startIdx = _cdtVertices.size();
	int idx = -1;
	for (int i = 0; i < vertices.size(); i++)
	{
		idx = _cdtVertices.size();
		_cdtVertices.push_back({ vertices[i].X, vertices[i].Y });
		if (i > 0)
			_cdtEdges.emplace_back(idx-1, idx);
	}
	_cdtEdges.emplace_back(idx, startIdx);
}

void gmapf::GMAPF::FillMapEnd()
{
	CDT::RemoveDuplicatesAndRemapEdges(_cdtVertices, _cdtEdges);

	CDT::Triangulation<float> cdt(
		CDT::VertexInsertionOrder::Auto,
		CDT::IntersectingConstraintEdges::TryResolve,
		1.0f
	);
	cdt.insertVertices(_cdtVertices);
	cdt.insertEdges(_cdtEdges);
	cdt.eraseOuterTrianglesAndHoles();

	std::vector<CuNodesMap::Node> nodes;
	for (auto& tr : cdt.triangles)
	{
		auto v1 = cdt.vertices[tr.vertices[0]];
		auto v2 = cdt.vertices[tr.vertices[1]];
		auto v3 = cdt.vertices[tr.vertices[2]];

		CuNodesMap::Node node;
		node.P1 = { v1.x, v1.y };
		node.P2 = { v2.x, v2.y };
		node.P3 = { v3.x, v3.y };
		int neibCounter = 0;
		for (int i = 0; i < 3; i++)
			if (tr.neighbors[i] != UINT_MAX)
				node.NeibsIdx[neibCounter++] = tr.neighbors[i];
		for (int i = neibCounter; i < 3; i++)
			node.NeibsIdx[i] = CuNodesMap::INVALID;

		nodes.push_back(node);
	}

	FillMapManually(nodes);
}


void GMAPF::FillMapManually(const std::vector<CuNodesMap::Node>& nodes)
{
	if (!_isInited)
		throw std::exception("not initialized");
	if (_isStarted)
		throw std::exception("cannot modify map after start");
	if (_map != nullptr)
		throw std::exception("map is already filled");

	_map = new Cum<CuNodesMap>();
	_map->HAllocAndMark(1, nodes.size());
	auto hMap = _map->H(0);

	for (int i = 0; i < nodes.size(); i++)
	{
		hMap.At(i) = nodes[i];
		hMap.At(i).PCenter = (nodes[i].P1 + nodes[i].P2 + nodes[i].P3) / 3.0f;
	}
}

int GMAPF::AddAgent(const V2Float& currPos)
{
	if (!_isInited)
		throw std::exception("not initialized");
	if (_isStarted)
		throw std::exception("cannot add agents after start");

	int currNodeIdx = -1;
	if (!_map->H(0).TryGetNodeIdx(currPos, &currNodeIdx))
		throw std::exception(("initial pos " + currPos.ToString() + " is invalid").c_str());

	Agent agent;
	agent.State = EAgentState::Idle;
	agent.CurrPos = currPos;
	agent.CurrNodeIdx = currNodeIdx;

	return _agents->H(0).Add(agent);
}

void GMAPF::SetAgentTargPos(int agentId, const V2Float& targPos)
{
	if (!_isInited)
		throw std::exception("not initialized");

	int targNodeIdx = -1;
	if (!_map->H(0).TryGetNodeIdx(targPos, &targNodeIdx))
		throw std::exception(("target pos " + targPos.ToString() + " is invalid").c_str());

	Agent& agent = _agents->H(0).At(agentId);
	agent.State = EAgentState::Search;
	agent.TargPos = targPos;
	agent.TargNodeIdx = targNodeIdx;

	_isAnyAgentModifiedFromCpu = true;
}

void GMAPF::AsyncStep(float deltaTime)
{
	if (!_isInited)
		throw std::exception("not initialized");

	if (_isStepGoing)
		return;

	if (!_isStarted)
		ManualStart();

	CuDriverCatch(cuCtxPushCurrent(_cuContext));

	if (_config.IsProfiler)
		_profIsPreCopyPerformed = _isAnyAgentModifiedFromCpu;
	if (_isAnyAgentModifiedFromCpu)
	{
		if (_config.IsProfiler)
			_profPreCopyStart = TIME_GET;
		_agents->CopyToDevice();
		_isAnyAgentModifiedFromCpu = false;
		if (_config.IsProfiler)
			_profPreCopyEnd = TIME_GET;
	}
	_pathFinder->AsyncPreFind();
	_agentsMover->AsyncPreMove();

	_pathFinder->Sync();
	_agentsMover->Sync();

	_pathFinder->AsyncFind();
	_agentsMover->AsyncMove(deltaTime);

	_isStepGoing = true;
}

void GMAPF::WaitStepEnd()
{
	if (!_isStepGoing)
		return;

	_pathFinder->Sync();
	_agentsMover->Sync();

	_pathFinder->PostFind();
	_agentsMover->PostMove();

	if (_config.IsProfiler)
		_profPostCopyStart = TIME_GET;
	_agents->CopyToHost();
	if (_config.IsProfiler)
		_profPostCopyEnd = TIME_GET;

	CuDriverCatch(cuCtxPopCurrent(nullptr));

	_isStepGoing = false;

	if (_config.IsProfiler)
		RecordProfiler();
}

const V2Float& GMAPF::GetAgentPos(int agentId)
{
	return _agents->H(0).At(agentId).CurrPos;
}

EAgentState GMAPF::GetAgentState(int agentId)
{
	return _agents->H(0).At(agentId).State;
}

void GMAPF::ProfilerDump() const
{
	if (!_config.IsProfiler)
		throw std::exception("profiler disabled");

	std::cout << "[gmapf]----------------------" << std::endl;

	TIME_STD_OUT("[gmapf] step", _profDurStepSum, _profDurStepMax, _profRecordsCount);
	TIME_STD_OUT("[gmapf] ---step pre copy", _profDurPreCopySum, _profDurPreCopyMax, _profPreCopiesCount);
	TIME_STD_OUT("[gmapf] ---step post copy", _profDurPostCopySum, _profDurPostCopyMax, _profRecordsCount);

	std::cout << std::endl;

	auto pf = _pathFinder;
	std::cout << "[gmapf] path finder" << std::endl;
	TIME_STD_OUT("[gmapf] find requests", pf->ProfDurFindRequestsSum, pf->ProfDurFindRequestsMax, _profRecordsCount);
	TIME_STD_OUT("[gmapf] clear collections", pf->ProfDurClearCollectionsSum, pf->ProfDurClearCollectionsMax, pf->ProfRecordsCount);
	TIME_STD_OUT("[gmapf] search", pf->ProfDurSearchSum, pf->ProfDurSearchMax, pf->ProfRecordsCount);
	TIME_STD_OUT("[gmapf] build paths", pf->ProfDurBuildPathsSum, pf->ProfDurBuildPathsMax, pf->ProfRecordsCount);

	std::cout << std::endl;

	auto am = _agentsMover;
	std::cout << "[gmapf] agents mover" << std::endl;
	TIME_STD_OUT("[gmapf] find", am->ProfDurFindAgentsSum, am->ProfDurFindAgentsMax, _profRecordsCount);
	TIME_STD_OUT("[gmapf] move", am->ProfDurMoveAgentsSum, am->ProfDurMoveAgentsMax, am->ProfRecordsCount);
	TIME_STD_OUT("[gmapf] collisions", am->ProfDurResolveCollisionsSum, am->ProfDurResolveCollisionsMax, am->ProfRecordsCount);
	TIME_STD_OUT("[gmapf] update path progress", am->ProfDurUpdatePathProgressSum, am->ProfDurUpdatePathProgressMax, am->ProfRecordsCount);

	std::cout << "[gmapf]----------------------" << std::endl;
}

void GMAPF::RecordProfiler()
{
	float durPreCopy = 0;
	if (_profIsPreCopyPerformed)
	{
		durPreCopy = TIME_DIFF_MS(_profPreCopyStart, _profPreCopyEnd);
		_profDurPreCopySum += durPreCopy;
		_profDurPreCopyMax = std::max(_profDurPreCopyMax, durPreCopy);
		_profPreCopiesCount += 1;
	}

	float durPostCopy = TIME_DIFF_MS(_profPostCopyStart, _profPostCopyEnd);
	_profDurPostCopySum += durPostCopy;
	_profDurPostCopyMax = std::max(_profDurPostCopyMax, durPostCopy);

	float durStep =
		durPreCopy +
		durPostCopy +
		_pathFinder->ProfDurFindRequests +
		_pathFinder->ProfDurClearCollections +
		_pathFinder->ProfDurSearch +
		_pathFinder->ProfDurBuildPaths +
		_agentsMover->ProfDurFindAgents +
		_agentsMover->ProfDurMoveAgents +
		_agentsMover->ProfDurResolveCollisions +
		_agentsMover->ProfDurUpdatePathProgress;
	_profDurStepSum += durStep;
	_profDurStepMax = std::max(_profDurStepMax, durStep);
	_profRecordsCount += 1;
}
