#pragma once
#include <vector>
#include <cuda.h>
#include <chrono>

#include "../CDT/CDT.h"

#include "Agent.h"
#include "Config.h"
#include "misc/CuNodesMap.h"
#include "misc/V2Float.h"
#include "misc/V2Int.h"

namespace gmapf
{
	template<typename T>
	class Cum;
	class CuNodesMap;
	template<typename T>
	class CuList;
	class AgentsMover;
	class PathFinder;

	namespace stma
	{
		class PathFinderSTMA;
	}


	class GMAPF
	{
	public:
		GMAPF();
		GMAPF(const GMAPF& gmapf) = delete;
		GMAPF(GMAPF&& gmapf) = delete;
		GMAPF& operator=(GMAPF& gmapf) = delete;
		GMAPF& operator=(GMAPF&& gmapf) = delete;
		~GMAPF();

		void Init(const Config& config);
		void ManualStart();

		int AddAgent(const V2Float& currPos);
		void SetAgentTargPos(int agentId, const V2Float& targPos);

		void FillMapStart(const V2Float& mapSize);
		void FillMapObstacle(const std::vector<V2Float>& vertices);
		void FillMapEnd();

		void FillMapManually(const std::vector<CuNodesMap::Node>& nodes);

		void AsyncStep(float deltaTime);
		void WaitStepEnd();

		const V2Float& GetAgentPos(int agentId);
		EAgentState GetAgentState(int agentId);

		void ProfilerDump() const;

	private:
		bool _isInited = false;
		bool _isStarted = false;
		bool _isStepGoing = false;

		Config _config;

		CUcontext _cuContext;

		std::vector<CDT::V2d<float>> _cdtVertices;
		std::vector<CDT::Edge> _cdtEdges;

		Cum<CuNodesMap>* _map = nullptr;

		Cum<CuList<Agent>>* _agents = nullptr;
		bool _isAnyAgentModifiedFromCpu = false;

		PathFinder* _pathFinder = nullptr;
		//STMA::PathFinderSTMA* _pathFinder = nullptr;
		
		AgentsMover* _agentsMover = nullptr;

		std::chrono::steady_clock::time_point _profPreCopyStart;
		std::chrono::steady_clock::time_point _profPreCopyEnd;
		std::chrono::steady_clock::time_point _profPostCopyStart;
		std::chrono::steady_clock::time_point _profPostCopyEnd;

		float _profDurStepSum = 0;
		float _profDurStepMax = 0;
		float _profDurPreCopySum = 0;
		float _profDurPreCopyMax = 0;
		bool _profIsPreCopyPerformed;
		bool _profPreCopiesCount;
		float _profDurPostCopySum = 0;
		float _profDurPostCopyMax = 0;
		int _profRecordsCount = 0;

		void RecordProfiler();
	};
}
