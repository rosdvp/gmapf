#include "../../include/kernels/KernelProcessAgent.h"

#include <device_launch_parameters.h>

#include "../../include/Agent.h"
#include "../../include/misc/Array.h"
#include "../../include/misc/Matrix.h"

using namespace cupat;

__global__ void cupat::KernelProcessAgent(KernelProcessAgentInput input)
{
	Matrix<int> map;
	map.Attach(input.DMap);
	Array<Agent> agents;
	agents.Attach(input.DAgents);

	int agentId = threadIdx.x;
	Agent& agent = agents.At(agentId);
	if (V2Float::DistSqr(agent.CurrPos, agent.TargPos) <= 0.001f)
		return;

	if (agent.IsNewPathRequested)
	{
		//TODO
		agent.IsNewPathRequested = false;
	}
	if (agent.DPath == nullptr)
		return;

	Array<V2Int> path;
	path.Attach(agent.DPath);

	bool isLastCell = agent.PathCellIdx + 1 == path.GetCount();
	V2Int nextCell;
	V2Float targPos;
	if (isLastCell)
	{
		targPos = agent.TargPos;
	}
	else
	{
		nextCell = path.At(agent.PathCellIdx);
		targPos.X = (nextCell.X + 0.5f) * input.ConfigSim.MapCellSize;
		targPos.Y = (nextCell.Y + 0.5f) * input.ConfigSim.MapCellSize;
	}

	V2Float delta = targPos - agent.CurrPos;
	float deltaLength = delta.GetLength();
	V2Float step = delta / delta.GetLength() * input.ConfigSim.AgentSpeed * input.DeltaTime;
	if (step.GetLength() >= deltaLength)
		agent.CurrPos = targPos;
	else
		agent.CurrPos += step;

	agent.CurrCell.X = static_cast<int>(agent.CurrPos.X / input.ConfigSim.MapCellSize);
	agent.CurrCell.Y = static_cast<int>(agent.CurrPos.Y / input.ConfigSim.MapCellSize);

	if (!isLastCell && agent.CurrCell == nextCell)
		agent.PathCellIdx += 1;
}
