#pragma once
#include <queue>
#include <unordered_map>

#include "../Agent.h"
#include "../misc/V2Int.h"

namespace cupat
{
	struct CpuFindPathAStarInput
	{
		Cum<CuMatrix<int>> Map;
		Cum<CuList<Agent>> Agents;
	};

	struct __align__(16) AStarNodeCpu
	{
		V2Int Cell;
		V2Int PrevCell;
		float F;
		float G;
	};

	struct CpuPriorityQueueComparer
	{
		bool operator()(const AStarNodeCpu& a, const AStarNodeCpu& b)
		{
			return a.F > b.F;
		}
	};

	inline void CpuFind(
		Agent& agent,
		CuMatrix<int> map,
		std::unordered_map<V2Int, AStarNodeCpu> visited,
		std::priority_queue<AStarNodeCpu, std::vector<AStarNodeCpu>, CpuPriorityQueueComparer>& frontier
		)
	{
		constexpr float heuristicK = 1.0f;
		V2Int neibsCellsDeltas[4] =
		{
			{-1, 0},
			{1, 0},
			{0, -1},
			{0, 1},
		};

		AStarNodeCpu start;
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

				AStarNodeCpu neib;
				neib.Cell = neibCell;
				neib.PrevCell = curr.Cell;
				neib.G = curr.G + 1;
				neib.F = neib.G + V2Int::DistSqr(neibCell, agent.CurrCell) * heuristicK;

				if (visited.find(neibCell) == visited.end() || visited[neibCell].F > neib.F)
				{
					visited[neibCell] = neib;
					frontier.push(neib);
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

		//printf("path:\n");

		V2Int iter = agent.CurrCell;
		do
		{
			iter = visited[visited[iter].PrevCell].Cell;
			//printf("(%d, %d)\n", iter.X, iter.Y);
		} while (iter != agent.TargCell);

		//printf("----------\n");
	}

	inline void CpuFindPathAStar(CpuFindPathAStarInput input)
	{
		auto map = input.Map.H(0);
		auto agents = input.Agents.H(0);

		std::unordered_map<V2Int, AStarNodeCpu> visited;

		CpuPriorityQueueComparer comparer;

		for (int i = 0; i < agents.Count(); i++)
		{
			visited.erase(visited.begin(), visited.end());
			std::priority_queue<AStarNodeCpu, std::vector<AStarNodeCpu>, CpuPriorityQueueComparer> frontier(comparer);
			CpuFind(
				agents.At(i),
				map,
				visited,
				frontier
			);
		}
	}
}
