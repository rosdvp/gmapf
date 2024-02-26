#pragma once
#include <cuda.h>
#include <vector>

#include "Agent.h"
#include "ConfigSim.h"
#include "misc/CuList.h"
#include "misc/Cum.h"
#include "misc/CuMatrix.h"
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

		const V2Float& GetAgentPos(int agentId);

	private:
		ConfigSim _config;

		Cum<CuMatrix<int>> _map;
		Cum<CuList<Agent>> _agents;


		V2Int PosToCell(const V2Float& pos) const;
	};
}
