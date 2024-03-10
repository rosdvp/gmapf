#include <iostream>

#include "cupat/include/Sim.h"

void PlaceObstaclesLines(cupat::Sim& sim);
void PlaceObstaclesZigZag(cupat::Sim& sim);

void TestFinder()
{
	cupat::ConfigSim config;
	config.MapCountX = 100;
	config.MapCountY = 100;
	config.MapCellSize = 10;
	config.AgentsCount = 2048;
	config.AgentSpeed = 100;
	config.AgentRadius = 100;
	config.PathFinderParallelAgents = 2048;
	config.PathFinderThreadsPerAgents = 32;
	config.PathFinderEachQueueCapacity = 16;
	config.PathFinderHeuristicK = 1;
	config.PathStorageCapacityK = 2;

	cupat::Sim sim;
	sim.Init(config);

	for (int i = 0; i < config.AgentsCount; i++)
	{
		sim.SetAgentInitialPos(i, { static_cast<float>(i / 5.0f), 0 });
		sim.SetAgentTargPos(i, { static_cast<float>(i / 5.0f), 950 });
	}

	PlaceObstaclesLines(sim);
	//PlaceObstaclesZigZag(sim);
	//for (int x = 0; x < config.MapCountX; x++)
	//	sim.SetObstacle({ x, 50});

	sim.Start(true);

	sim.DoStepOnlyFinder();

	sim.DebugDump();

	sim.Destroy();
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
	config.PathFinderThreadsPerAgents = 32;
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
	sim.Destroy();
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
	sim.Destroy();
}


int main()
{
	TestFinder();
	//TestMover();
	//TestFull();

	std::cout << "test done" << std::endl;
	return 0;
}


void PlaceObstaclesLinesSub(cupat::Sim& sim, int offsetX, int lineY, int lineWidth, int space)
{
	int x = offsetX;
	while (x < 100)
	{
		for (int i = 0; i < lineWidth; i++)
		{
			if (x >= 0 && x < 100)
			{
				sim.SetObstacle({ x, lineY });
				//printf("X");
			}
			x += 1;
		}
		for (int i = 0; i < space; i++)
		{
			//if (x >= 0 && x < 100)
			//	printf("-");
			x += 1;
		}
	}
	//printf("\n");
}

void PlaceObstaclesLines(cupat::Sim& sim)
{
	PlaceObstaclesLinesSub(sim, 0, 20, 20, 10);
	PlaceObstaclesLinesSub(sim, -18, 30, 20, 10);
	PlaceObstaclesLinesSub(sim, -18 * 2, 40, 20, 10);
	PlaceObstaclesLinesSub(sim, -18 * 3, 50, 20, 10);
	PlaceObstaclesLinesSub(sim, -18 * 4, 60, 20, 10);
	PlaceObstaclesLinesSub(sim, -18 * 5, 70, 20, 10);
	PlaceObstaclesLinesSub(sim, -18 * 6, 80, 20, 10);
}

void PlaceObstaclesZigZag(cupat::Sim& sim)
{
	for (int x = 10; x < 100; x++)
		sim.SetObstacle({ x, 20 });

	for (int x = 0; x < 90; x++)
		sim.SetObstacle({ x, 30 });

	for (int x = 10; x < 100; x++)
		sim.SetObstacle({ x, 40 });

	for (int x = 0; x < 90; x++)
		sim.SetObstacle({ x, 50 });

	for (int x = 10; x < 100; x++)
		sim.SetObstacle({ x, 60 });

	for (int x = 0; x < 90; x++)
		sim.SetObstacle({ x, 70 });

	for (int x = 10; x < 100; x++)
		sim.SetObstacle({ x, 80 });
}