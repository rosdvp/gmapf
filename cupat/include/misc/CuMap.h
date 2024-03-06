#pragma once
#include <cuda_runtime_api.h>

namespace cupat
{
	template<typename TKey, typename TVal>
	class CuMap
	{
	private:
		struct __align__(16) Entry
		{
			bool IsOccupied;
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
				_entries[i].IsOccupied = false;
		}

		__host__ __device__ bool Has(const TKey& key) const
		{
			Entry& entry = GetEntry(key);
			return !entry.IsEmpty && entry.Key == key;
		}

		__host__ __device__ const TVal& Get(const TKey& key) const
		{
			Entry& entry = GetEntry(key);
			assert(!entry.IsEmpty && entry.Key == key);
			return entry.Val;
		}

		__host__ __device__ void Set(const TKey& key, TVal& val)
		{
			Entry& entry = GetEntry(key);
			entry.Key = key;
			entry.Val = val;
			entry.IsEmpty = false;
		}

	private:
		int* _capacity = nullptr;
		Entry* _entries = nullptr;


		__host__ __device__ Entry& GetEntry(const TKey& key) const
		{
			int capacity = *_capacity;

			size_t hash = key.GetHash();
			int startIdx = hash % capacity;
			int idx = startIdx;
			for (int i = 0; i < capacity; i++)
			{
				idx = (startIdx + i) % capacity;
				if (!_entries[idx].IsOccupied || _entries[idx].Key == key)
					return _entries[idx];
			}
			printf("CuMap miss\n", key.X, key.Y);
			return _entries[idx];
		}
	};
}
