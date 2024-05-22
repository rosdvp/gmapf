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
		int CurrNodeIdx;
		V2Float TargPos;
		int TargNodeIdx;

		EAgentState State;

		void* Path = nullptr;
		int PathStepIdx;
		int PathStepNode;
		V2Float PathStepPos;
	};
}
