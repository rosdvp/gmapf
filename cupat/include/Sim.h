#pragma once
#include <vector>

#include "Agent.h"
#include "ConfigSim.h"
#include "MapDesc.h"
#include "misc/V2Float.h"
#include "misc/V2Int.h"

namespace cupat
{
	template<typename T>
	class Cum;
	template<typename T>
	class CuMatrix;
	template<typename T>
	class CuList;
	class AgentsMover;
	class PathFinder;

	class Sim
	{
	public:
		void Init(const ConfigSim& config);

		void Destroy();

		void SetAgentInitialPos(int agentId, const V2Float& currPos);
		void SetAgentTargPos(int agentId, const V2Float& targPos);
		void DebugSetAgentPath(int agentId, const std::vector<V2Int>& path);

		void SetObstacle(const V2Int& cell);

		void Start(bool isDebugSyncMode);

		void DoStep(float deltaTime);
		void DoStepOnlyFinder();
		void DoStepOnlyMover(float deltaTime);

		const V2Float& GetAgentPos(int agentId);

		void DebugDump() const;

	private:
		ConfigSim _config;

		MapDesc _mapDesc;

		Cum<CuMatrix<int>>* _map = nullptr;
		Cum<CuList<Agent>>* _agents = nullptr;

		PathFinder* _pathFinder = nullptr;
		AgentsMover* _agentsMover = nullptr;

		float _debugDurStep = 0;
		float _debugDurStepMax = 0;
		int _debugStepsCount = 0;
	};
}
