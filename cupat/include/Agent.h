#pragma once
#include "misc/V2Float.h"
#include "misc/V2Int.h"

namespace cupat
{
	enum class EAgentState
	{
		Idle,
		Search,
		Move,
	};

	struct __align__(16) Agent
	{
		V2Float CurrPos;
		V2Float TargPos;

		EAgentState State;

		void* Path;
		int PathStepIdx;
		int PathStepNode;
		V2Float PathStepPos;
	};
}
