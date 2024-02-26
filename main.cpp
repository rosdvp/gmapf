#include <iostream>

#include "cupat/include/Sim.h"

void PlaceObstacles(cupat::Sim& sim, int width, int height);

void TestThread()
{
	cupat::ConfigSim config;
	config.MapCountX = 100;
	config.MapCountY = 100;
	config.MapCellSize = 10;
	config.AgentsCount = 1;
	config.AgentSpeed = 1;
	config.AgentRadius = 100;

	cupat::Sim sim;
	sim.Init(config);

	for (int i = 0; i < config.AgentsCount; i++)
	{
		sim.SetAgentInitialPos(i, { static_cast<float>(i), 0 });
		sim.SetAgentTargPos(i, { static_cast<float>(i), 900 });
	}

	PlaceObstacles(sim, config.MapCountX, config.MapCountY);

	sim.Start();

	int stepsCount = 1;
	for (int i = 0; i < stepsCount; i++)
	{
		sim.DoStep(2);
		std::cout << sim.GetAgentPos(0) << std::endl;
	}

	std::cout << "agents final poses:" << std::endl;
	for (int i = 0; i < config.AgentsCount; i++)
	{
		std::cout << sim.GetAgentPos(i) << std::endl;
	}
}


int main()
{
	TestThread();

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