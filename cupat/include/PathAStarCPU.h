#pragma once
#include <queue>
#include <unordered_map>

#include "Agent.h"
#include "misc/CuList.h"
#include "misc/Cum.h"
#include "misc/CuNodesMap.h"

namespace cupat
{
	struct CpuFindPathAStarInput
	{
		Cum<CuNodesMap> Map;
		Cum<CuList<Agent>> Agents;
	};

	struct __align__(16) AStarNodeCpu
	{
		int NodeIdx;
		int PrevNodeIdx;
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
		CuNodesMap map,
		std::unordered_map<int, AStarNodeCpu> visited,
		std::priority_queue<AStarNodeCpu, std::vector<AStarNodeCpu>, CpuPriorityQueueComparer>& frontier
		)
	{
		constexpr float heuristicK = 1.0f;

		int startNodeIdx = -1;
		if (!map.TryGetNodeIdx(agent.CurrPos, &startNodeIdx))
		{
			printf("agent curr pos is invalid node");
			agent.State = EAgentState::Idle;
			return;
		}
		int targNodeIdx = -1;
		if (!map.TryGetNodeIdx(agent.TargPos, &targNodeIdx))
		{
			printf("agent targ pos is invalid node");
			agent.State = EAgentState::Idle;
			return;
		}
		if (startNodeIdx == targNodeIdx)
		{
			agent.State = EAgentState::Idle;
			return;
		}

		AStarNodeCpu targNode;
		targNode.NodeIdx = targNodeIdx;
		targNode.F = map.GetDistSqr(startNodeIdx, targNodeIdx) * heuristicK;
		targNode.G = 0;

		visited[targNodeIdx] = targNode;
		frontier.push(targNode);


		bool isFound = false;
		while (!isFound && !frontier.empty())
		{
			auto curr = frontier.top();
			frontier.pop();

			for (auto& neibNodeIdx : map.At(curr.NodeIdx).NeibsIdx)
			{
				AStarNodeCpu neib;
				neib.NodeIdx = neibNodeIdx;
				neib.PrevNodeIdx = curr.NodeIdx;
				neib.G = curr.G + 1;
				neib.F = neib.G + map.GetDistSqr(neibNodeIdx, startNodeIdx) * heuristicK;

				if (visited.find(neibNodeIdx) == visited.end() || visited[neibNodeIdx].F > neib.F)
				{
					visited[neibNodeIdx] = neib;
					frontier.push(neib);
				}

				if (neibNodeIdx == startNodeIdx)
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

		int nodeIdx = startNodeIdx;
		do
		{
			nodeIdx = visited[visited[nodeIdx].PrevNodeIdx].NodeIdx;
			//printf("(%d, %d)\n", iter.X, iter.Y);
		} while (nodeIdx != targNodeIdx);

		//printf("----------\n");
	}

	inline void CpuFindPathAStar(CpuFindPathAStarInput input)
	{
		auto map = input.Map.H(0);
		auto agents = input.Agents.H(0);

		std::unordered_map<int, AStarNodeCpu> visited;

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
