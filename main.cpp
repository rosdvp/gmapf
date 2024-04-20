#include <iostream>
#include <fstream>

#include "cupat/CDT/CDT.h"
#include "cupat/include/Sim.h"

void AddObstaclesGridLines(cupat::Sim& sim);
void AddObstaclesGridZigZag(cupat::Sim& sim);
void AddObstaclesNavMeshEmpty(cupat::Sim& sim, float mapSizeX, float mapSizeY);
void AddObstaclesNavMeshLines(cupat::Sim& sim);
void AddObstaclesNavMeshZigZag(cupat::Sim& sim);
void FillFromFile(cupat::Sim& sim, int agentsCount);

void TestFinder()
{
	cupat::ConfigSim config;
	config.AgentsMaxCount = 2048;
	config.AgentSpeed = 100;
	config.AgentRadius = 100;
	config.PathFinderParallelAgents = 2048;
	config.PathFinderThreadsPerAgents = 2;
	config.PathFinderQueueCapacity = 8;
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

	//AddObstaclesNavMeshEmpty(sim, 1000, 1000);
	//AddObstaclesNavMeshLines(sim);
	AddObstaclesNavMeshZigZag(sim);

	for (int i = 0; i < config.AgentsMaxCount; i++)
	{
		sim.AddAgent({ static_cast<float>(i / 5.0f), 0 });
		sim.SetAgentTargPos(i, { static_cast<float>(i / 5.0f), 950 });
	}

	//FillFromFile(sim, config.AgentsMaxCount);

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

	AddObstaclesGridLines(sim);

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
	TestFinder();
	//TestMover();
	//TestFull();

	std::cout << "test done" << std::endl;
	return 0;
}


void AddObstaclesGrid(cupat::Sim& sim, const std::vector<cupat::V2Int>& obstacles)
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

	sim.FillMap(result.data(), 10, 100, 100);
}

void AddObstaclesGridLine(
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

void AddObstaclesGridLines(cupat::Sim& sim)
{
	std::vector<cupat::V2Int> obstacles;
	AddObstaclesGridLine(obstacles, 0, 20, 20, 10);
	AddObstaclesGridLine(obstacles, -18, 30, 20, 10);
	AddObstaclesGridLine(obstacles, -18 * 2, 40, 20, 10);
	AddObstaclesGridLine(obstacles, -18 * 3, 50, 20, 10);
	AddObstaclesGridLine(obstacles, -18 * 4, 60, 20, 10);
	AddObstaclesGridLine(obstacles, -18 * 5, 70, 20, 10);
	AddObstaclesGridLine(obstacles, -18 * 6, 80, 20, 10);
	AddObstaclesGrid(sim, obstacles);
}

void AddObstaclesGridZigZag(cupat::Sim& sim)
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

	AddObstaclesGrid(sim, obstacles);
}


void AddObstaclesNavMesh(cupat::Sim& sim, const CDT::Triangulation<float>& cdt)
{
	std::vector<cupat::CuNodesMap::Node> nodes;
	for (auto& tr : cdt.triangles)
	{
		auto v1 = cdt.vertices[tr.vertices[0]];
		auto v2 = cdt.vertices[tr.vertices[1]];
		auto v3 = cdt.vertices[tr.vertices[2]];

		cupat::CuNodesMap::Node node;
		node.P1 = { v1.x, v1.y };
		node.P2 = { v2.x, v2.y };
		node.P3 = { v3.x, v3.y };
		node.PCenter = (node.P1 + node.P2 + node.P3) / 3.0f;
		node.Val = 0;

		int neibCounter = 0;
		for (int i = 0; i < 3; i++)
			if (tr.neighbors[i] != UINT_MAX)
				node.NeibsIdx[neibCounter++] = tr.neighbors[i];
		for (int i = neibCounter; i < 3; i++)
			node.NeibsIdx[i] = cupat::CuNodesMap::INVALID;

		nodes.push_back(node);
	}

	sim.FillMap(nodes);
}

void AddObstaclesNavMeshEmpty(cupat::Sim& sim, float mapSizeX, float mapSizeY)
{
	std::vector<CDT::V2d<float>> vertices;

	vertices.push_back({ 0, 0 });
	vertices.push_back({ 0, mapSizeY });
	vertices.push_back({ mapSizeX, mapSizeY });
	vertices.push_back({ mapSizeX, 0 });

	vertices.push_back({ 0, mapSizeY / 2 });
	vertices.push_back({ mapSizeX / 2, 0});
	vertices.push_back({ mapSizeX / 2, mapSizeY / 2 });

	CDT::Triangulation<float> cdt;
	cdt.insertVertices(vertices);
	cdt.eraseSuperTriangle();

	AddObstaclesNavMesh(sim, cdt);
}



void AddObstaclesNavMeshLine(
	std::vector<CDT::V2d<float>>& vertices,
	std::vector<CDT::Edge> edges,
	float mapSizeX,
	float mapSizeY,
	float offsetX,
	float posY,
	float lineSizeX,
	float lineSizeY,
	float space)
{
	float x = offsetX;
	while (x < mapSizeX)
	{
		float x0 = std::clamp(x, 0.0f, mapSizeX);
		float x1 = std::clamp(x + lineSizeX, 0.0f, mapSizeX);
		float y0 = std::clamp(posY, 0.0f, mapSizeY);
		float y1 = std::clamp(posY + lineSizeY, 0.0f, mapSizeY);

		int i = edges.size();
		vertices.push_back({ x0, y0 });
		vertices.push_back({ x0, y1 });
		vertices.push_back({ x1, y1 });
		vertices.push_back({ x1, y0 });

		edges.emplace_back(i, i + 1);
		edges.emplace_back(i + 1, i + 2);
		edges.emplace_back(i + 2, i + 3);
		edges.emplace_back(i + 3, i);

		x += space;
	}
}

void AddObstaclesNavMeshLines(cupat::Sim& sim)
{
	float mapSizeX = 1000;
	float mapSizeY = 1000;

	std::vector<CDT::V2d<float>> vertices;
	vertices.push_back({ 0, 0 });
	vertices.push_back({ 0, mapSizeY });
	vertices.push_back({ mapSizeX, mapSizeY });
	vertices.push_back({ mapSizeX, 0 });

	std::vector<CDT::Edge> edges;
	edges.emplace_back(0, 1);
	edges.emplace_back(1, 2);
	edges.emplace_back(2, 3);
	edges.emplace_back(3, 0);

	AddObstaclesNavMeshLine(vertices, edges, mapSizeX, mapSizeY, 0, 200, 200, 10, 100);
	AddObstaclesNavMeshLine(vertices, edges, mapSizeX, mapSizeY, -18, 300, 200, 10, 100);
	AddObstaclesNavMeshLine(vertices, edges, mapSizeX, mapSizeY, -18 * 2, 400, 200, 10, 100);
	AddObstaclesNavMeshLine(vertices, edges, mapSizeX, mapSizeY, -18 * 3, 500, 200, 10, 100);
	AddObstaclesNavMeshLine(vertices, edges, mapSizeX, mapSizeY, -18 * 4, 600, 200, 10, 100);
	AddObstaclesNavMeshLine(vertices, edges, mapSizeX, mapSizeY, -18 * 5, 700, 200, 10, 100);
	AddObstaclesNavMeshLine(vertices, edges, mapSizeX, mapSizeY, -18 * 6, 800, 200, 10, 100);

	CDT::RemoveDuplicatesAndRemapEdges(vertices, edges);

	CDT::Triangulation<float> cdt(
		CDT::VertexInsertionOrder::Auto, 
		CDT::IntersectingConstraintEdges::TryResolve, 
		1.0f
	);
	cdt.insertVertices(vertices);
	cdt.insertEdges(edges);
	cdt.eraseOuterTrianglesAndHoles();

	AddObstaclesNavMesh(sim, cdt);
}

void AddObstaclesNavMeshZigZag(cupat::Sim& sim)
{
	float mapSizeX = 1000;
	float mapSizeY = 1000;

	std::vector<CDT::V2d<float>> vertices;
	vertices.push_back({ 0, 0 });
	vertices.push_back({ 0, mapSizeY });
	vertices.push_back({ mapSizeX, mapSizeY });
	vertices.push_back({ mapSizeX, 0 });

	std::vector<CDT::Edge> edges;
	edges.emplace_back(0, 1);
	edges.emplace_back(1, 2);
	edges.emplace_back(2, 3);
	edges.emplace_back(3, 0);

	for (int i = 2; i <= 8; i++)
	{
		float y0 = i * 100;
		float y1 = y0 + 10;
		float x0 = i % 2 == 0 ? 100 : 0;
		float x1 = i % 2 == 0 ? 1000 : 100;

		int n = vertices.size();
		vertices.push_back({ x0, y0 });
		vertices.push_back({ x0, y1 });
		vertices.push_back({ x1, y1 });
		vertices.push_back({ x1, y0 });

		edges.emplace_back(n, n+1);
		edges.emplace_back(n + 1, n + 2);
		edges.emplace_back(n+2, n+3);
		edges.emplace_back(n + 3, n);
	}

	CDT::RemoveDuplicatesAndRemapEdges(vertices, edges);

	CDT::Triangulation<float> cdt(
		CDT::VertexInsertionOrder::Auto,
		CDT::IntersectingConstraintEdges::TryResolve,
		1.0f
	);
	cdt.insertVertices(vertices);
	cdt.insertEdges(edges);
	cdt.eraseOuterTrianglesAndHoles();

	AddObstaclesNavMesh(sim, cdt);
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
