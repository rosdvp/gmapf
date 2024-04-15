#pragma once
#include <cassert>
#include <crt/host_defines.h>

namespace cupat
{
	template<typename T>
	class CuVisitedMap
	{
	private:
		struct __align__(8) Entry
		{
			int IsVisited;
			T Val;
		};

	public:
		__host__ __device__ explicit CuVisitedMap(void* ptr)
		{
			auto p = static_cast<char*>(ptr);
			_count = reinterpret_cast<int*>(p);
			p += 8;
			_entries = reinterpret_cast<Entry*>(p);
		}

		__host__ __device__ void Mark(int count)
		{
			*_count = count;
			for (int i = 0; i < count; i++)
				_entries[i].IsVisited = 0;
		}

		__host__ __device__ T& At(int idx)
		{
			assert(idx >= 0 && idx < *_count);
			return _entries[idx].Val;
		}

		__host__ __device__ bool IsVisited(int idx) const
		{
			assert(idx >= 0 && idx < *_count);
			return _entries[idx].IsVisited == 1;
		}

		__host__ __device__ bool TryVisit(int idx, T& val)
		{
			assert(idx >= 0 && idx < *_count);
			Entry& entry = _entries[idx];
			int old = atomicExch(&(entry.IsVisited), 1);
			if (old == 1)
				return false;
			entry.Val = val;
			return true;
		}

		__host__ __device__ void UnVisit(int idx)
		{
			assert(idx >= 0 && idx < *_count);
			_entries[idx].IsVisited = 0;
		}

		__host__ __device__ int Count() const
		{
			return *_count;
		}

		__host__ __device__ static constexpr size_t EvalSize(int count)
		{
			return 8 + sizeof(Entry) * count;
		}

	private:
		Entry* _entries = nullptr;
		int* _count = nullptr;
	};
}
