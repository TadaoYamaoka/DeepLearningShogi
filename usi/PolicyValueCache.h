#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <vector>

#include "common.hpp"

class PolicyValueCache {
public:
	struct Result {
		float value = 0.0f;
		std::vector<float> policy;
	};

	// 探索ワーカーが停止している間に設定する。
	void SetCapacity(const size_t capacity)
	{
		capacity_ = capacity;
		shard_count_ = std::min(capacity, kShardCount);

		for (size_t i = 0; i < kShardCount; ++i) {
			auto& shard = shards_[i];
			const std::unique_lock<std::shared_mutex> lock(shard.mutex);
			std::vector<Entry>().swap(shard.entries);
			std::vector<Index>().swap(shard.buckets);
			std::vector<float>().swap(shard.pending_policy);
			shard.newest = shard.oldest = kNoEntry;
			shard.capacity = i < shard_count_
				? capacity / shard_count_ + (i < capacity % shard_count_ ? 1 : 0)
				: 0;
		}
	}

	bool IsEnabled() const { return capacity_ != 0; }

	template <typename PolicySetter>
	bool Lookup(const Key key, const size_t policy_size, float& value, PolicySetter&& set_policy)
	{
		if (!IsEnabled())
			return false;

		auto& shard = GetShard(key);
		Index index;
		uint64_t generation;
		{
			const std::shared_lock<std::shared_mutex> lock(shard.mutex);
			index = shard.Find(key);
			if (index == kNoEntry)
				return false;
			const auto& entry = shard.entries[index];
			if (entry.result.policy.size() != policy_size)
				return false;

			generation = entry.generation;
			const Result& result = entry.result;
			value = result.value;
			// set_policyはロック中に呼ばれるため、出力先への単純コピーのみを
			// 行い、このキャッシュへ再入してはならない。
			for (size_t i = 0; i < policy_size; ++i)
				set_policy(i, result.policy[i]);
			// 既に先頭なら、LRU更新用のロック取得は不要。
			if (index == shard.newest)
				return true;
		}

		// ヒット処理を待たせないため、LRU順の更新は排他ロックを
		// 即時取得できた場合だけ行う。
		const std::unique_lock<std::shared_mutex> lock(shard.mutex, std::try_to_lock);
		if (lock.owns_lock())
			shard.TouchIfCurrent(index, key, generation);

		return true;
	}

	template <typename PolicyGetter>
	void Store(const Key key, const float value, const size_t policy_size, PolicyGetter&& get_policy)
	{
		if (!IsEnabled())
			return;

		auto& shard = GetShard(key);
		const std::unique_lock<std::shared_mutex> lock(shard.mutex);
		Index index = shard.Find(key);
		if (index == kNoEntry || shard.entries[index].result.policy.size() != policy_size) {
			// policyを読み終えてから公開し、置き換えた領域を次のStoreで再利用する。
			shard.pending_policy.resize(policy_size);
			for (size_t i = 0; i < policy_size; ++i)
				shard.pending_policy[i] = get_policy(i);
			if (index == kNoEntry)
				index = shard.Insert(key);
			auto& entry = shard.entries[index];
			entry.result.policy.swap(shard.pending_policy);
			entry.result.value = value;
			entry.generation = NextGeneration(shard);
		}
		// 同じキー・policy長では先着の評価を維持する。
		shard.MoveToFront(index);
	}

private:
	friend class PolicyValueCacheTest_StaleLookupAfterCapacityChange_Test;
	using Index = size_t;
	static constexpr Index kNoEntry = static_cast<Index>(-1);
	static constexpr unsigned kShardBits = 8;
	static constexpr size_t kShardCount = size_t{1} << kShardBits;

	struct Entry {
		Result result;
		Key key;
		uint64_t generation;
		Index next_in_bucket;
		Index newer = kNoEntry;
		Index older = kNoEntry;
	};

	// バケットとLRUは添字でつなぐ。entriesが再確保されても添字は変わらない。
	struct alignas(64) Shard {
		size_t capacity = 0;
		std::shared_mutex mutex;
		std::vector<Entry> entries;
		std::vector<Index> buckets;
		std::vector<float> pending_policy;
		Index newest = kNoEntry;
		Index oldest = kNoEntry;
		uint64_t next_generation = 0;

		size_t Bucket(const Key key) const { return (key >> kShardBits) & (buckets.size() - 1); }

		Index Find(const Key key) const
		{
			if (!buckets.empty()) {
				for (Index i = buckets[Bucket(key)]; i != kNoEntry; i = entries[i].next_in_bucket)
					if (entries[i].key == key)
						return i;
			}
			return kNoEntry;
		}

		void LinkToBucket(const Index i)
		{
			Index& head = buckets[Bucket(entries[i].key)];
			entries[i].next_in_bucket = head;
			head = i;
		}

		Index Insert(const Key key)
		{
			Index index = entries.size();
			if (index == capacity) {
				index = oldest;
				Index* link = &buckets[Bucket(entries[index].key)];
				while (*link != index)
					link = &entries[*link].next_in_bucket;
				*link = entries[index].next_in_bucket;
			} else {
				if (index == buckets.size()) {
					buckets.assign(std::max(size_t{8}, buckets.size() * 2), kNoEntry);
					for (Index i = 0; i < index; ++i)
						LinkToBucket(i);
				}
				entries.emplace_back();
			}
			entries[index].key = key;
			LinkToBucket(index);
			return index;
		}

		void MoveToFront(const Index i)
		{
			if (i == newest)
				return;
			auto& entry = entries[i];
			if (entry.newer != kNoEntry)
				entries[entry.newer].older = entry.older;
			if (entry.older != kNoEntry)
				entries[entry.older].newer = entry.newer;
			if (i == oldest)
				oldest = entry.newer;
			entry.newer = kNoEntry;
			entry.older = newest;
			if (newest != kNoEntry)
				entries[newest].newer = i;
			else
				oldest = i;
			newest = i;
		}

		void TouchIfCurrent(const Index i, const Key key, const uint64_t generation)
		{
			// ロックを取り直す間に同じ添字が追い出し・再登録で再利用されることがある。
			if (i < entries.size() && entries[i].generation == generation && entries[i].key == key)
				MoveToFront(i);
		}
	};

	Shard& GetShard(const Key key)
	{
		const size_t hash = std::hash<Key>{}(key);
		const size_t index = shard_count_ == kShardCount
			? hash & (kShardCount - 1)
			: hash % shard_count_;
		return shards_[index];
	}

	static uint64_t NextGeneration(Shard& shard)
	{
		// 0は未初期化Entry用に予約する。実運用での周回は事実上起こらないが、
		// 周回時も0を飛ばして世代比較の意味を維持する。
		if (++shard.next_generation == 0)
			++shard.next_generation;
		return shard.next_generation;
	}

	size_t capacity_ = 0;
	size_t shard_count_ = 0;
	std::array<Shard, kShardCount> shards_;
};
