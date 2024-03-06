#pragma once

namespace cupat
{
	struct ConfigSim
	{
		int MapCountX;
		int MapCountY;
		float MapCellSize;

		int AgentsCount;
		float AgentRadius;
		float AgentSpeed;

		int PathFinderParallelAgents;
		int PathFinderThreadsPerAgents;
		int PathFinderEachQueueCapacity;
		float PathFinderHeuristicK;

		int PathStorageBinsK;
		int PathStorageBinSize;
	};
}