#pragma once
#include "misc/V2Float.h"
#include "misc/V2Int.h"

namespace cupat
{
	struct __align__(16) Agent
	{
		V2Float CurrPos;
		V2Float NextPos;
		V2Float TargPos;
		V2Int CurrCell;
		V2Int TargCell;

		bool IsNewPathRequested;
		int PathCellIdx;
		void* DPath;
	};
}
