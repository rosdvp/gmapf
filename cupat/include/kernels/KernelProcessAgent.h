#pragma once

#include "cuda_runtime.h"
#include "../ConfigSim.h"


namespace cupat
{
	struct KernelProcessAgentInput
	{
		ConfigSim ConfigSim;
		void* DMap;
		void* DAgents;

		float DeltaTime;
	};

	__global__ void KernelProcessAgent(KernelProcessAgentInput input);
}
