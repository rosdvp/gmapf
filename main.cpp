#include <iostream>
#include <fstream>

#include "include/CDT/CDT.h"
#include "include/gmapf/GMAPF.h"

using namespace gmapf;

void AddObstaclesNavMeshEmpty(GMAPF& gmapf);
void AddObstaclesNavMeshLines(GMAPF& gmapf);
void AddObstaclesNavMeshZigZag(GMAPF& gmapf);


void TestFinder()
{
	Config config;
	config.AgentsMaxCount = 2048;
	config.AgentSpeed = 100;
	config.AgentRadius = 100;
	config.PathFinderParallelAgents = 2048;
	config.PathFinderThreadsPerAgents = 2;
	config.PathFinderQueueCapacity = 16;
	config.PathFinderHeuristicK = 1;
	config.IsProfiler = true;

	GMAPF gmapf;
	gmapf.Init(config);

	//AddObstaclesNavMeshEmpty(gmapf);
	//AddObstaclesNavMeshLines(gmapf);
	AddObstaclesNavMeshZigZag(gmapf);

	for (int i = 0; i < config.AgentsMaxCount; i++)
	{
		gmapf.AddAgent({ static_cast<float>(i / 5.0f), 0 });
		gmapf.SetAgentTargPos(i, { static_cast<float>(i / 5.0f), 950 });
	}

	gmapf.ManualStart();
	gmapf.AsyncStep(1);
	gmapf.WaitStepEnd();
	gmapf.ProfilerDump();
}

void TestFull()
{
	Config config;
	config.AgentsMaxCount = 2048;
	config.AgentSpeed = 1;
	config.AgentRadius = 2;
	config.PathFinderParallelAgents = 2048;
	config.PathFinderThreadsPerAgents = 2;
	config.PathFinderQueueCapacity = 16;
	config.PathFinderHeuristicK = 1;
	config.IsProfiler = true;

	GMAPF gmapf;
	gmapf.Init(config);

	//AddObstaclesNavMeshEmpty(gmapf);
	//AddObstaclesNavMeshLines(gmapf);
	AddObstaclesNavMeshZigZag(gmapf);

	for (int i = 0; i < config.AgentsMaxCount; i++)
	{
		gmapf.AddAgent({ static_cast<float>(i / 5.0f), 0 });
		gmapf.SetAgentTargPos(i, { static_cast<float>(i / 5.0f), 950 });
	}

	gmapf.ManualStart();

	int stepsCount = 10000;
	for (int i = 0; i < stepsCount; i++)
	{
		gmapf.AsyncStep(1);
		gmapf.WaitStepEnd();
	}

	//std::cout << "agents final poses:" << std::endl;
	//for (int i = 0; i < config.AgentsMaxCount; i++)
	//{
	//	std::cout << gmapf.GetAgentPos(i) << std::endl;
	//}

	gmapf.ProfilerDump();
}


int main()
{
	TestFinder();
	//TestFull();

	std::cout << "test done" << std::endl;
	return 0;
}


void AddObstaclesNavMeshEmpty(GMAPF& gmapf)
{
	gmapf.FillMapStart({ 1000, 1000 });
	gmapf.FillMapEnd();
}

void AddObstaclesNavMeshLine(
	GMAPF& gmapf,
	float mapSizeX,
	float mapSizeY,
	float offsetX,
	float posY,
	float lineSizeX,
	float lineSizeY,
	float space)
{
	std::vector<V2Float> temp;

	float x = offsetX;
	while (x < mapSizeX)
	{
		float x0 = std::clamp(x, 0.0f, mapSizeX);
		float x1 = std::clamp(x + lineSizeX, 0.0f, mapSizeX);
		float y0 = std::clamp(posY, 0.0f, mapSizeY);
		float y1 = std::clamp(posY + lineSizeY, 0.0f, mapSizeY);

		temp.emplace_back(x0, y0);
		temp.emplace_back(x0, y1);
		temp.emplace_back(x1, y1);
		temp.emplace_back(x1, y0);
		gmapf.FillMapObstacle(temp);

		x += space;
	}
}

void AddObstaclesNavMeshLines(GMAPF& gmapf)
{
	float mapSizeX = 1000;
	float mapSizeY = 1000;
	gmapf.FillMapStart({mapSizeX, mapSizeY});

	AddObstaclesNavMeshLine(gmapf, mapSizeX, mapSizeY, 0, 200, 200, 10, 100);
	AddObstaclesNavMeshLine(gmapf, mapSizeX, mapSizeY, -18, 300, 200, 10, 100);
	AddObstaclesNavMeshLine(gmapf, mapSizeX, mapSizeY, -18 * 2, 400, 200, 10, 100);
	AddObstaclesNavMeshLine(gmapf, mapSizeX, mapSizeY, -18 * 3, 500, 200, 10, 100);
	AddObstaclesNavMeshLine(gmapf, mapSizeX, mapSizeY, -18 * 4, 600, 200, 10, 100);
	AddObstaclesNavMeshLine(gmapf, mapSizeX, mapSizeY, -18 * 5, 700, 200, 10, 100);
	AddObstaclesNavMeshLine(gmapf, mapSizeX, mapSizeY, -18 * 6, 800, 200, 10, 100);

	gmapf.FillMapEnd();
}

void AddObstaclesNavMeshZigZag(GMAPF& gmapf)
{
	float mapSizeX = 1000;
	float mapSizeY = 1000;
	gmapf.FillMapStart({ mapSizeX, mapSizeY });

	std::vector<V2Float> temp;
	for (int i = 2; i <= 8; i++)
	{
		float y0 = i * 100;
		float y1 = y0 + 10;
		float x0 = i % 2 == 0 ? 100 : 0;
		float x1 = i % 2 == 0 ? 1000 : 100;

		temp.clear();
		temp.emplace_back(x0, y0);
		temp.emplace_back(x0, y1);
		temp.emplace_back(x1, y1);
		temp.emplace_back(x1, y0);

		gmapf.FillMapObstacle(temp);
	}

	gmapf.FillMapEnd();
}