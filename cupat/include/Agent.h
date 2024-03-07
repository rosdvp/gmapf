#pragma once
#include "misc/V2Float.h"
#include "misc/V2Int.h"

namespace cupat
{
	struct __align__(16) Agent
	{
		int Priority;

		V2Float CurrPos;
		V2Float TargPos;

		V2Int CurrCell;
		V2Int TargCell;

		bool IsTargetReached;

		bool IsNewPathRequested;
		int PathIdx;
		void* Path;
		int PathStepIdx;
		V2Int PathNextCell;
	};
}
