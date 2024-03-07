#include <iostream>

#include "cupat/include/Sim.h"

void PlaceObstacles(cupat::Sim& sim, int width, int height);

void TestFinder()
{
	cupat::ConfigSim config;
	config.MapCountX = 100;
	config.MapCountY = 100;
	config.MapCellSize = 10;
	config.AgentsCount = 100;
	config.AgentSpeed = 100;
	config.AgentRadius = 100;
	config.PathFinderParallelAgents = 128;
	config.PathFinderThreadsPerAgents = 128;
	config.PathFinderEachQueueCapacity = 32;
	config.PathFinderHeuristicK = 1;
	config.PathStorageCapacityK = 2;

	cupat::Sim sim;
	sim.Init(config);

	for (int i = 0; i < config.AgentsCount; i++)
	{
		sim.SetAgentInitialPos(i, { static_cast<float>(i * 5), 0 });
		sim.SetAgentTargPos(i, { static_cast<float>(i * 5), 900 });
	}

	PlaceObstacles(sim, config.MapCountX, config.MapCountY);
	//for (int x = 0; x < config.MapCountX; x++)
	//	sim.SetObstacle({ x, 50});

	sim.Start(true);
	sim.DoStepOnlyFinder();

	sim.DebugDump();
}

void TestMover()
{
	cupat::ConfigSim config;
	config.MapCountX = 100;
	config.MapCountY = 100;
	config.MapCellSize = 10;
	config.AgentsCount = 2;
	config.AgentSpeed = 1;
	config.AgentRadius = 5;
	config.PathFinderParallelAgents = 128;
	config.PathFinderThreadsPerAgents = 128;
	config.PathFinderEachQueueCapacity = 32;
	config.PathFinderHeuristicK = 1;
	config.PathStorageCapacityK = 2;

	cupat::Sim sim;
	sim.Init(config);

	sim.SetAgentInitialPos(0, { 10, 5 });
	sim.SetAgentTargPos(0, { 10, 50});
	sim.DebugSetAgentPath(0,
		{
			{1, 1},
			{1, 2},
			{1, 3},
			{1, 4},
			{1, 5},
		}
	);

	sim.SetAgentInitialPos(1, { 10, 50 });
	sim.SetAgentTargPos(1, { 10, 5 });
	sim.DebugSetAgentPath(1,
		{
			{1, 4},
			{1, 3},
			{1, 2},
			{1, 1},
			{1, 0},
		}
	);

	sim.Start(false);

	int stepsCount = 100;
	for (int i = 0; i < stepsCount; i++)
	{
		sim.DoStepOnlyMover(1);
		std::cout << "0: " << sim.GetAgentPos(0) << ", 1: " << sim.GetAgentPos(1) << std::endl;
	}

	std::cout << "agents final poses:" << std::endl;
	for (int i = 0; i < config.AgentsCount; i++)
	{
		std::cout << sim.GetAgentPos(i) << std::endl;
	}

	sim.DebugDump();
}

void TestFull()
{
	cupat::ConfigSim config;
	config.MapCountX = 100;
	config.MapCountY = 100;
	config.MapCellSize = 10;
	config.AgentsCount = 128;
	config.AgentSpeed = 1;
	config.AgentRadius = 2;
	config.PathFinderParallelAgents = 128;
	config.PathFinderThreadsPerAgents = 128;
	config.PathFinderEachQueueCapacity = 32;
	config.PathFinderHeuristicK = 1;
	config.PathStorageCapacityK = 4;

	cupat::Sim sim;
	sim.Init(config);

	for (int i = 0; i < config.AgentsCount; i++)
	{
		sim.SetAgentInitialPos(i, { static_cast<float>(i * 5), 0 });
		sim.SetAgentTargPos(i, { static_cast<float>(i * 5), 50 });
	}

	sim.Start(false);

	int stepsCount = 100;
	for (int i = 0; i < stepsCount; i++)
	{
		sim.DoStep(1);
	}

	std::cout << "agents final poses:" << std::endl;
	for (int i = 0; i < config.AgentsCount; i++)
	{
		std::cout << sim.GetAgentPos(i) << std::endl;
	}

	sim.DebugDump();
}


int main()
{
	TestFinder();
	//TestMover();
	//TestFull();

	std::cout << "test done" << std::endl;
	return 0;
}


void PlaceObstacles(cupat::Sim& sim, int width, int height)
{
	for (int x = 0; x < width - 1; x++)
		sim.SetObstacle({ x, 10 });

	for (int x = 10; x < width; x++)
		sim.SetObstacle({ x, 20 });

	for (int x = 0; x < width - 1; x++)
		sim.SetObstacle({ x, 30 });

	for (int x = 10; x < width; x++)
		sim.SetObstacle({ x, 40 });

	for (int x = 0; x < width - 1; x++)
		sim.SetObstacle({ x, 50 });

	for (int x = 10; x < width; x++)
		sim.SetObstacle({ x, 60 });

	for (int x = 0; x < width - 1; x++)
		sim.SetObstacle({ x, 70 });

	for (int x = 10; x < width; x++)
		sim.SetObstacle({ x, 80 });

	for (int x = 0; x < width - 1; x++)
		sim.SetObstacle({ x, 90 });
}