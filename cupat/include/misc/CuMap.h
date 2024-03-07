#pragma once
#include <cassert>
#include <cstdio>
#include <cuda_runtime_api.h>

namespace cupat
{
	template<typename TKey, typename TVal>
	class CuMap
	{
	private:
		struct __align__(16) Entry
		{
			int IsOccupied;
			int IsModifying;
			TKey Key;
			TVal Val;
		};

	public:
		__host__ __device__ explicit CuMap(void* ptr)
		{
			auto p = static_cast<char*>(ptr);
			_capacity = reinterpret_cast<int*>(p);
			p += 8;
			_entries = reinterpret_cast<Entry*>(p);
		}

		__host__ __device__ void Mark(int capacity)
		{
			*_capacity = capacity;

			for (int i = 0; i < capacity; i++)
			{
				_entries[i].IsOccupied = 0;
				_entries[i].IsModifying = 0;
			}
		}

		__host__ __device__ bool TryAdd(const TKey& key, const TVal& val)
		{
			int debugStatus[2];
			debugStatus[0] = 0;
			debugStatus[1] = 0;

			TKey debugPrevKeys[2];

			for (int hashFuncIdx = 0; hashFuncIdx < HASH_FUNCS_COUNT; hashFuncIdx++)
			{
				int idx = GetIdx(key, hashFuncIdx);
				Entry& entry = _entries[idx];

				if (atomicExch(&entry.IsModifying, 1) == 0)
				{
					if (entry.IsOccupied == 0)
					{
						entry.IsOccupied = 1;
						entry.Key = key;
						entry.Val = val;
						atomicExch(&entry.IsModifying, 0);
						return true;
					}
					if (entry.Key == key)
					{
						atomicExch(&entry.IsModifying, 0);
						return false;
					}
				}
				else
				{
					//retry until no one is modifying
					hashFuncIdx -= 1;
				}
			}

			{
				int idx0 = GetIdx(key, 0);
				Entry& entry0 = _entries[idx0];
				int idx1 = GetIdx(key, 1);
				Entry& entry1 = _entries[idx1];

				printf("CuMap, TryAdd, miss for key (%d, %d), 0: (%d, %d)[%d], 1: (%d, %d)[%d], ds: %d%d, dk: (%d, %d) (%d, %d)\n",
					key.X, key.Y,
					entry0.Key.X, entry0.Key.Y, entry0.IsOccupied,
					entry1.Key.X, entry1.Key.Y, entry1.IsOccupied,
					debugStatus[0], debugStatus[1],
					debugPrevKeys[0].X, debugPrevKeys[0].Y, debugPrevKeys[1].X, debugPrevKeys[1].Y
				);
				assert(0);
			}
			return false;
		}

		__host__ __device__ bool Has(const TKey& key) const
		{
			for (int i = 0; i < HASH_FUNCS_COUNT; i++)
			{
				int idx = GetIdx(key, i);
				Entry& entry = _entries[idx];
				if (entry.IsOccupied == 1 && entry.Key == key)
					return true;
			}
			return false;
		}

		__host__ __device__ const TVal& Get(const TKey& key) const
		{
			for (int i = 0; i < HASH_FUNCS_COUNT; i++)
			{
				int idx = GetIdx(key, i);
				Entry& entry = _entries[idx];
				if (entry.IsOccupied == 1 && entry.Key == key)
					return entry.Val;
			}
			printf("CuMap, Get, key is not present\n");
			assert(0);
			return {};
		}

		__host__ __device__ void RemoveByIdx(int idx)
		{
			_entries[idx].IsOccupied = 0;
		}

		__host__ __device__ int Capacity()
		{
			return *_capacity;
		}


		__host__ __device__ static constexpr size_t EvalSize(int capacity)
		{
			return 8 + sizeof(Entry) * capacity;
		}

	private:
		static constexpr int HASH_FUNCS_COUNT = 3;

		int* _capacity = nullptr;
		Entry* _entries = nullptr;

		__host__ __device__ int GetIdx(const TKey& key, int hashFuncId) const
		{
			size_t hash = 0;
			if (hashFuncId == 0)
				hash = key.GetHash1();
			else if (hashFuncId == 1)
				hash = key.GetHash2();
			else if (hashFuncId == 2)
				hash = key.GetHash3();
			return hash % (*_capacity);
		}
	};
}
