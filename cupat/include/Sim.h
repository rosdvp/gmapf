#pragma once
#include <cuda.h>
#include <vector>

#include "Agent.h"
#include "ConfigSim.h"
#include "misc/Array.h"
#include "misc/Matrix.h"
#include "misc/V2Float.h"
#include "misc/V2Int.h"

namespace cupat
{
	class Sim
	{
	public:
		~Sim();

		void Init(const ConfigSim& config);

		void SetAgentInitialPos(int agentId, const V2Float& currPos);
		void SetAgentTargPos(int agentId, const V2Float& targPos);

		void SetObstacle(const V2Int& cell);

		void Start();

		bool DoStep(float deltaTime);

		const V2Float& GetAgentPos(int agentId) const;

	private:
		ConfigSim _config;

		Matrix<int> _map;
		void* _dRawMap = nullptr;

		Array<Agent> _agents;
		void* _dRawAgents = nullptr;


		V2Int PosToCell(const V2Float& pos) const;
	};
}
