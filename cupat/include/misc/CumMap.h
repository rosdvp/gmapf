#pragma once
#include <cuda_runtime_api.h>
#include <crt/host_defines.h>

namespace cupat
{
	template<typename TKey, typename TVal>
	class CumMap
	{
	private:
		struct __align__(16) Entry
		{
			bool IsEmpty;
			TKey Key;
			TVal Val;
		};

	public:
		static CumMap* New(int count, int eachCapacity)
		{
			CumMap* ptr;
			cudaMallocManaged(&ptr, count * sizeof(CumMap));
			for (int i = 0; i < count; i++)
				new (ptr + i) CumMap(eachCapacity);
			return ptr;
		}

		explicit CumMap(int capacity)
		{
			_capacity = capacity;
			cudaMallocManaged(&_entries, sizeof(Entry) * capacity);
			for (int i = 0; i < capacity; i++)
				_entries[i].IsEmpty = true;
		}

		~CumMap()
		{
			if (_entries != nullptr)
			{
				cudaFree(_entries);
				_entries = nullptr;
			}
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
		int _capacity;
		Entry* _entries = nullptr;


		__host__ __device__ Entry& GetEntry(const TKey& key) const
		{
			size_t hash = key.GetHash();
			int startIdx = hash % _capacity;
			int idx = startIdx;
			for (int i = 0; i < 10; i++)
			{
				idx = (startIdx + i) % _capacity;
				if (_entries[idx].IsEmpty || _entries[idx].Key == key)
					return _entries[idx];
			}
			printf("map miss (%d, %d)\n", key.X, key.Y);
			return _entries[idx];
		}
	};
}
