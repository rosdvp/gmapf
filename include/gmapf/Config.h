#pragma once

namespace gmapf
{
	struct Config
	{
		int AgentsMaxCount;
		float AgentRadius;
		float AgentSpeed;

		int PathFinderParallelAgents;
		int PathFinderThreadsPerAgents;
		int PathFinderQueueCapacity;
		float PathFinderHeuristicK;

		bool IsDebug = false;
		bool IsProfiler = false;
	};
}