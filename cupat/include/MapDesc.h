#pragma once
#include "misc/V2Float.h"
#include "misc/V2Int.h"


namespace cupat
{
	class MapDesc
	{
	public:
		MapDesc() {}

		MapDesc(float cellSize, int countX, int countY)
		{
			_cellSize = cellSize;
			_countX = countX;
			_countY = countY;
		}

		__host__ __device__ bool IsValidPos(const V2Float& pos) const
		{
			return pos.X >= 0 && pos.X < _cellSize * _countX &&
				pos.Y >= 0 && pos.Y < _cellSize * _countY;
		}

		__host__ __device__ V2Float CellToCenterPos(const V2Int& cell) const
		{
			float x = (cell.X + 0.5f) * _cellSize;
			float y = (cell.Y + 0.5f) * _cellSize;
			return { x, y };
		}

		__host__ __device__ V2Int PosToCell(const V2Float& pos) const
		{
			int x = static_cast<int>(pos.X / _cellSize);
			int y = static_cast<int>(pos.Y / _cellSize);
			return { x ,y };
		}

	private:
		int _countX = 0;
		int _countY = 0;
		float _cellSize = 0;
	};
}
