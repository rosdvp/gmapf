#pragma once

namespace cupat
{
	struct ConfigSim
	{
		int AgentsMaxCount;
		float AgentRadius;
		float AgentSpeed;

		int PathFinderParallelAgents;
		int PathFinderThreadsPerAgents;
		int PathFinderQueueCapacity;
		float PathFinderHeuristicK;
	};
}