#pragma once
#include <queue>
#include <unordered_map>

#include "../Agent.h"
#include "../ConfigSim.h"
#include "../misc/Array.h"
#include "../misc/Matrix.h"
#include "../misc/V2Int.h"

namespace cupat
{
	struct CpuFindPathAStarInput
	{
		void* Map;
		void* Agents;
		int AgentId;
	};

	struct __align__(16) AStarNode
	{
		V2Int Cell;
		V2Int PrevCell;
		float F;
		float G;
	};


	inline void CpuFindPathAStar(CpuFindPathAStarInput input)
	{
		constexpr float heuristicK = 1.0f;
		V2Int neibsCellsDeltas[4] =
		{
			{-1, 0},
			{1, 0},
			{0, -1},
			{0, 1},
		};

		int expandedNodesCount = 1;

		Matrix<int> map;
		map.Attach(input.Map);
		Array<Agent> agents;
		agents.Attach(input.Agents);
		Agent& agent = agents.At(input.AgentId);

		std::unordered_map<V2Int, AStarNode> visited;

		struct Comparer
		{
			bool operator()(const AStarNode& a, const AStarNode& b)
			{
				return a.F > b.F;
			}
		} comparer;
		std::priority_queue<AStarNode, std::vector<AStarNode>, Comparer> frontier(comparer);


		AStarNode start;
		start.Cell = agent.TargCell;
		start.F = V2Int::DistSqr(agent.CurrCell, agent.TargCell) * heuristicK;
		start.G = 0;

		visited[start.Cell] = start;
		frontier.push(start);


		bool isFound = false;
		while (!isFound && !frontier.empty())
		{
			auto curr = frontier.top();
			frontier.pop();

			for (auto& neibCellDelta : neibsCellsDeltas)
			{
				auto neibCell = curr.Cell + neibCellDelta;
				if (!map.IsValid(neibCell) || map.At(neibCell) != 0)
					continue;

				AStarNode neib;
				neib.Cell = neibCell;
				neib.PrevCell = curr.Cell;
				neib.G = curr.G + 1;
				neib.F = neib.G + V2Int::DistSqr(neibCell, agent.CurrCell) * heuristicK;

				if (visited.find(neibCell) == visited.end() || visited[neibCell].F > neib.F)
				{
					visited[neibCell] = neib;
					frontier.push(neib);
					expandedNodesCount += 1;
				}

				if (neibCell == agent.CurrCell)
				{
					isFound = true;
					break;
				}
			}
		}

		if (!isFound)
		{
			printf("failed to find path\n");
			return;
		}

		printf("expanded nodes: %d\n", expandedNodesCount);
		printf("path:\n");

		V2Int iter = agent.CurrCell;
		do
		{
			iter = visited[visited[iter].PrevCell].Cell;
			printf("(%d, %d)\n", iter.X, iter.Y);
		} while (iter != agent.TargCell);

		printf("----------\n");
	}
}
