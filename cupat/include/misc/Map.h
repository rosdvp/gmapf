#pragma once

#include <cuda_runtime.h>
#include <assert.h>
#include <cstdlib>

namespace cupat
{
	template<typename TKey, typename TVal>
	class Map
	{
	private:
		
		struct __align__(16) Entry
		{
			bool IsEmpty;
			TKey Key;
			TVal Val;
		};

	public:
		Map() = default;
		~Map() = default;
		Map(const Map& other) = delete;
		Map(Map&& other) = delete;
		Map& operator=(const Map& other) = delete;
		Map& operator=(Map&& other) = delete;

		__host__ void AllocOnHost(int capacity)
		{
			_size = EvalSize(capacity);
			void* p = malloc(_size);
			Attach(p);
			*_capacity = capacity;
			for (int i = 0; i < capacity; i++)
				_entries[i].IsEmpty = true;
			_size = EvalSize(capacity);
		}

		__host__ void FreeOnHost()
		{
			free(_ptr);
			_ptr = nullptr;
		}

		__host__ void* AllocOnDeviceAndCopyFromHost()
		{
			void* p;
			cudaMalloc(&p, _size);
			cudaMemcpy(p, _ptr, _size, cudaMemcpyHostToDevice);
			return p;
		}

		__host__ __device__ void Attach(void* p)
		{
			_ptr = p;

			auto ptr = static_cast<char*>(p);
			_capacity = reinterpret_cast<int*>(ptr);
			ptr += 16;
			_entries = reinterpret_cast<Entry*>(ptr);

			_size = EvalSize(*_capacity);
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

		__host__ __device__ constexpr size_t EvalSize(int capacity)
		{
			return 16 + sizeof(Entry) * capacity;
		}

	private:
		size_t _size;
		void* _ptr = nullptr;
		
		Entry* _entries = nullptr;
		int* _capacity = nullptr;


		__host__ __device__ Entry& GetEntry(const TKey& key) const
		{
			size_t hash = key.GetHash();
			int startIdx = hash % *_capacity;
			int idx = startIdx;
			for (int i = 0; i < 10; i++)
			{
				idx = (startIdx + i) % *_capacity;
				if (_entries[idx].IsEmpty || _entries[idx].Key == key)
					return _entries[idx];
			}
			printf("map miss\n");
			return _entries[idx];
		}
	};
}