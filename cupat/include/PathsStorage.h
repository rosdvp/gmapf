#pragma once

#include <cuda_runtime.h>

#include "misc/V2Int.h"

namespace cupat
{
	class PathsStorage
	{
		struct __align__(16) Path
		{
			int UsersCount;
			V2Int Start;
			V2Int Target;
			void* DPath;
		};

	public:
		__host__ void DAlloc(int binsCount, int pathsPerBin)
		{
			_binsCount = binsCount;
			_pathsPerBin = pathsPerBin;

			cudaMalloc(&_paths, sizeof(Path) * binsCount * pathsPerBin);
		}

		__host__ void DFree()
		{
			if (_paths == nullptr)
				return;
			cudaFree(_paths);
			_paths = nullptr;
		}

		__device__ void* GetUsingPath(int idx) const
		{
			return _paths[idx].DPath;
		}

		__device__ void SetUsingPath(int idx, void* path)
		{
			_paths[idx].DPath = path;
		}

		__device__ int GetPathUsersCount(int idx) const
		{
			return _paths[idx].UsersCount;
		}

		__device__ int TryUsePath(const V2Int& start, const V2Int& target, bool& outIsRequested)
		{
			int startIdx;
			int endIdx;
			FindBin(start, target, startIdx, endIdx);

			while (true)
			{
				bool isSetFailed = false;
				for (int i = startIdx; i <= endIdx; i++)
				{
					Path& path = _paths[i];

					if (path.UsersCount == 0)
					{
						if (atomicExch(&path.UsersCount, 1) == 0)
						{
							path.Start = start;
							path.Target = target;
							if (path.DPath != nullptr)
							{
								cudaFree(path.DPath);
								path.DPath = nullptr;
							}
							outIsRequested = true;
							return i;
						}
						isSetFailed = true;
						break;
					}
					if (path.Start == start && path.Target == target)
					{
						atomicAdd(&path.UsersCount, 1);
						outIsRequested = false;
						return i;
					}
				}
				if (!isSetFailed)
				{
					int filledCount = 0;
					for (int i = startIdx; i <= endIdx; i++)
					{
						Path& path = _paths[i];
						if (path.UsersCount > 0)
							filledCount += 1;
					}
					printf("path storage not enough space for (%d, %d) -> (%d, %d), bin fill %d/%d\n",
						start.X, start.Y,
						target.X, target.Y,
						filledCount, _pathsPerBin
					);
					for (int a = startIdx; a <= endIdx; a++)
					{
						Path& pA = _paths[a];
						for (int b = startIdx; b <= endIdx; b++)
						{
							if (a == b)
								continue;

							Path& pB = _paths[b];
							if (pA.Start == pB.Start && pA.Target == pB.Target)
								printf("path storage duplicate detected\n");
						}
					}
					break;
				}
			}
			return -1;
		}

		__device__ void UnUsePath(int idx)
		{
			atomicSub(&_paths[idx].UsersCount, 1);
		}

		__device__ void RemoveAll()
		{
			for (int i = 0; i < _binsCount * _pathsPerBin; i++)
			{
				_paths[i].UsersCount = 0;
				_paths[i].DPath = nullptr;
			}
		}

	private:
		int _binsCount = 0;
		int _pathsPerBin = 0;
		Path* _paths = nullptr;


		__device__ void FindBin(const V2Int& start, const V2Int& target, int& outStartIdx, int& outEndIdx)
		{
			size_t hash = GetHash(start, target);
			int binIdx = hash % _binsCount;
			outStartIdx = binIdx * _pathsPerBin;
			outEndIdx = outStartIdx + _pathsPerBin - 1;
		}

		__device__ size_t GetHash(const V2Int& a, const V2Int& b)
		{
			size_t seed = 0;
			seed ^= a.GetHash() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= b.GetHash() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			return seed;
		}
	};
}
