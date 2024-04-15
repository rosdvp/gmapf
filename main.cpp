#include <iostream>
#include <fstream>

#include "cupat/include/Sim.h"

std::vector<cupat::V2Int> GetObstaclesLines();
std::vector<cupat::V2Int> GetObstaclesZigZag();
std::vector<int> ConvertObstacles(const std::vector<cupat::V2Int>& obstacles);
void FillFromFile(cupat::Sim& sim, int agentsCount);

void TestFinder()
{
	cupat::ConfigSim config;
	config.AgentsMaxCount = 1024;
	config.AgentSpeed = 100;
	config.AgentRadius = 100;
	config.PathFinderParallelAgents = 1024;
	config.PathFinderThreadsPerAgents = 32;
	config.PathFinderQueueCapacity = 16;
	config.PathFinderHeuristicK = 1;

	cupat::Sim sim;
	sim.Init(config);

	//std::vector<cupat::V2Int> obstacles;
	//std::vector<cupat::V2Int> obstacles = GetObstaclesLines();
	//std::vector<cupat::V2Int> obstacles = GetObstaclesZigZag();
	//for (int x = 0; x < 100; x++)
	//	obstacles.emplace_back(x, 50);
	//std::vector<int> cells = ConvertObstacles(obstacles);
	//sim.FillMap(cells.data(), 10, 100, 100);

	//for (int i = 0; i < config.AgentsMaxCount; i++)
	//{
	//	sim.AddAgent({ static_cast<float>(i / 5.0f), 0 });
	//	sim.SetAgentTargPos(i, { static_cast<float>(i / 5.0f), 950 });
	//}

	FillFromFile(sim, config.AgentsMaxCount);

	sim.Start(true);

	sim.DoStepOnlyFinder();

	sim.DebugDump();

	sim.Destroy();
}

void TestMover()
{
	cupat::ConfigSim config;
	config.AgentsMaxCount = 2;
	config.AgentSpeed = 1;
	config.AgentRadius = 5;
	config.PathFinderParallelAgents = 128;
	config.PathFinderThreadsPerAgents = 32;
	config.PathFinderQueueCapacity = 32;
	config.PathFinderHeuristicK = 1;

	cupat::Sim sim;
	sim.Init(config);

	std::vector<cupat::V2Int> obstacles;
	std::vector<int> cells = ConvertObstacles(obstacles);
	sim.FillMap(cells.data(), 10, 100, 100);

	sim.AddAgent({ 10, 5 });
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

	sim.AddAgent({ 10, 50 });
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
	for (int i = 0; i < config.AgentsMaxCount; i++)
	{
		std::cout << sim.GetAgentPos(i) << std::endl;
	}

	sim.DebugDump();
	sim.Destroy();
}

void TestFull()
{
	cupat::ConfigSim config;
	config.AgentsMaxCount = 4;
	config.AgentSpeed = 1;
	config.AgentRadius = 2;
	config.PathFinderParallelAgents = 1024;
	config.PathFinderThreadsPerAgents = 2;
	config.PathFinderQueueCapacity = 32;
	config.PathFinderHeuristicK = 1;

	cupat::Sim sim;
	sim.Init(config);

	//std::vector<cupat::V2Int> obstacles;
	//obstacles = GetObstaclesLines();
	//obstacles = GetObstaclesZigZag();
	//for (int x = 0; x < 100; x++)
	//	obstacles.emplace_back(x, 50);
	//std::vector<int> cells = ConvertObstacles(obstacles);
	//sim.FillMap(cells.data(), 10, 100, 100);
	//
	//for (int i = 0; i < config.AgentsMaxCount; i++)
	//{
	//	sim.AddAgent(cupat::V2Float(i * 5, 0));
	//	sim.SetAgentTargPos(i, cupat::V2Float(i * 5, 950));
	//}

	FillFromFile(sim, config.AgentsMaxCount);

	sim.Start(false);

	int stepsCount = 10000;
	for (int i = 0; i < stepsCount; i++)
	{
		sim.DoStep(1);
	}

	std::cout << "agents final poses:" << std::endl;
	for (int i = 0; i < config.AgentsMaxCount; i++)
	{
		std::cout << sim.GetAgentPos(i) << std::endl;
	}

	sim.DebugDump();
	sim.Destroy();
}


int main()
{
	//TestFinder();
	//TestMover();
	TestFull();

	std::cout << "test done" << std::endl;
	return 0;
}


void PlaceObstaclesLinesSub(
	std::vector<cupat::V2Int>& obstacles, 
	int offsetX, 
	int lineY, 
	int lineWidth, 
	int space)
{
	int x = offsetX;
	while (x < 100)
	{
		for (int i = 0; i < lineWidth; i++)
		{
			if (x >= 0 && x < 100)
				obstacles.emplace_back(x, lineY);
			x += 1;
		}
		for (int i = 0; i < space; i++)
			x += 1;
	}
}

std::vector<cupat::V2Int> GetObstaclesLines()
{
	std::vector<cupat::V2Int> obstacles;
	PlaceObstaclesLinesSub(obstacles, 0, 20, 20, 10);
	PlaceObstaclesLinesSub(obstacles, -18, 30, 20, 10);
	PlaceObstaclesLinesSub(obstacles, -18 * 2, 40, 20, 10);
	PlaceObstaclesLinesSub(obstacles, -18 * 3, 50, 20, 10);
	PlaceObstaclesLinesSub(obstacles, -18 * 4, 60, 20, 10);
	PlaceObstaclesLinesSub(obstacles, -18 * 5, 70, 20, 10);
	PlaceObstaclesLinesSub(obstacles, -18 * 6, 80, 20, 10);
	return obstacles;
}

std::vector<cupat::V2Int> GetObstaclesZigZag()
{
	std::vector<cupat::V2Int> obstacles;

	for (int x = 10; x < 100; x++)
		obstacles.emplace_back(x, 20);

	for (int x = 0; x < 90; x++)
		obstacles.emplace_back(x, 30);

	for (int x = 10; x < 100; x++)
		obstacles.emplace_back(x, 40);

	for (int x = 0; x < 90; x++)
		obstacles.emplace_back(x, 50);

	for (int x = 10; x < 100; x++)
		obstacles.emplace_back(x, 60);

	for (int x = 0; x < 90; x++)
		obstacles.emplace_back(x, 70);

	for (int x = 10; x < 100; x++)
		obstacles.emplace_back(x, 80);

	return obstacles;
}

std::vector<int> ConvertObstacles(const std::vector<cupat::V2Int>& obstacles)
{
	std::vector<int> result;
	for (int x = 0; x < 100; x++)
		for (int y = 0; y < 100; y++)
			result.push_back(0);

	for (auto& obstacle : obstacles)
	{
		int i = obstacle.Y * 100 + obstacle.X;
		result[i] = -1;
	}

	return result;
}

void FillFromFile(cupat::Sim& sim, int agentsCount)
{
	float cellSize = 200;
	int cellsCountX = 200;
	int cellsCountY = 200;

	std::vector<int> cells;

	std::ifstream file;
	file.open("D:\\Projects\\cpp\\map.txt");
	for (int i = 0; i < cellsCountX * cellsCountY; i++)
	{
		char v;
		file >> v;
		if (v == 'o')
			cells.push_back(-1);
		else
			cells.push_back(0);
	}
	file.close();

	sim.FillMap(cells.data(), cellSize, cellsCountX, cellsCountY);

	file.open("D:\\Projects\\cpp\\agents.txt");
	for (int i = 0; i < agentsCount; i++)
	{
		float initPosX, initPosY, targPosX, targPosY;
		file >> initPosX >> initPosY >> targPosX >> targPosY;

		int idx = sim.AddAgent(cupat::V2Float(initPosX, initPosY));
		sim.SetAgentTargPos(idx, cupat::V2Float(targPosX, targPosY));
	}
	file.close();
}