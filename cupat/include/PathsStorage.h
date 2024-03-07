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
		__host__ void DAlloc(int capacity)
		{
			_capacity = capacity;
			cudaMalloc(&_paths, sizeof(Path) * capacity);
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
			int startIdx = FindStartIdx(start, target);

			while (true)
			{
				bool isSetFailed = false;
				for (int offset = 0; offset < _capacity; offset++)
				{
					int idx = (startIdx + offset) % _capacity;

					Path& path = _paths[idx];

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
							return idx;
						}
						isSetFailed = true;
						break;
					}
					if (path.Start == start && path.Target == target)
					{
						atomicAdd(&path.UsersCount, 1);
						outIsRequested = false;
						return idx;
					}
				}
				if (!isSetFailed)
				{
					printf("paths storage is full\n");
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
			for (int i = 0; i < _capacity; i++)
			{
				_paths[i].UsersCount = 0;
				_paths[i].DPath = nullptr;
			}
		}

	private:
		int _capacity = 0;
		Path* _paths = nullptr;

		__device__ int FindStartIdx(const V2Int& start, const V2Int& target)
		{
			size_t hash = GetHash(start, target);
			return hash % _capacity;
		}

		__device__ size_t GetHash(const V2Int& a, const V2Int& b)
		{
			size_t seed = 0;
			seed ^= a.GetHash1() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= b.GetHash1() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			return seed;
		}
	};
}
