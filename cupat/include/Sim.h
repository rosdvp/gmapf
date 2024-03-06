#pragma once
#include <vector>

#include "Agent.h"
#include "ConfigSim.h"
#include "MapDesc.h"
#include "misc/CuList.h"
#include "misc/Cum.h"
#include "misc/CuMatrix.h"
#include "misc/V2Float.h"
#include "misc/V2Int.h"

namespace cupat
{
	class AgentsMover;
	class PathFinder;

	class Sim
	{
	public:
		~Sim();

		void Init(const ConfigSim& config);

		void SetAgentInitialPos(int agentId, const V2Float& currPos);
		void SetAgentTargPos(int agentId, const V2Float& targPos);
		void DebugSetAgentPath(int agentId, const std::vector<V2Int>& path);

		void SetObstacle(const V2Int& cell);

		void Start();

		void DoStep(float deltaTime);
		void DoStepOnlyFinder();
		void DoStepOnlyMover(float deltaTime);

		const V2Float& GetAgentPos(int agentId);

		void DebugDump() const;

	private:
		ConfigSim _config;

		MapDesc _mapDesc;

		Cum<CuMatrix<int>> _map;
		Cum<CuList<Agent>> _agents;

		PathFinder* _pathFinder;
		AgentsMover* _agentsMover;


		V2Int PosToCell(const V2Float& pos) const;
	};
}
