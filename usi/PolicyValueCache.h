#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.hpp"

class PolicyValueCache {
public:
	struct Result {
		float value = 0.0f;
		std::vector<float> policy;
	};

	void SetCapacity(const size_t capacity)
	{
		capacity_ = capacity;
		shard_count_ = std::min(capacity, kShardCount);

		for (size_t i = 0; i < kShardCount; ++i) {
			auto& shard = shards_[i];
			std::unique_lock<std::shared_mutex> lock(shard.mutex);
			shard.entries.clear();
			shard.lru.clear();
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
		uint64_t generation;
		{
			std::shared_lock<std::shared_mutex> lock(shard.mutex);
			auto it = shard.entries.find(key);
			if (it == shard.entries.end() || it->second.result.policy.size() != policy_size)
				return false;

			generation = it->second.generation;
			const Result& result = it->second.result;
			value = result.value;
			// set_policyはロック中に呼ばれるため、出力先への単純コピーのみを
			// 行い、このキャッシュへ再入してはならない。
			for (size_t i = 0; i < policy_size; ++i)
				set_policy(i, result.policy[i]);
		}

		// ヒット処理を待たせないため、LRU順の更新は排他ロックを
		// 即時取得できた場合だけ行う。
		std::unique_lock<std::shared_mutex> lock(shard.mutex, std::try_to_lock);
		if (lock.owns_lock()) {
			auto it = shard.entries.find(key);
			if (it != shard.entries.end() && it->second.generation == generation)
				shard.lru.splice(shard.lru.begin(), shard.lru, it->second.lru_position);
		}

		return true;
	}

	template <typename PolicyGetter>
	void Store(const Key key, const float value, const size_t policy_size, PolicyGetter&& get_policy)
	{
		if (!IsEnabled())
			return;

		auto& shard = GetShard(key);
		std::unique_lock<std::shared_mutex> lock(shard.mutex);
		auto it = shard.entries.find(key);
		if (it != shard.entries.end()) {
			// 同じ局面を複数GPUが推論した場合は先着結果を維持する。
			// policy長が異なる場合だけ不整合エントリとして置換する。
			if (it->second.result.policy.size() != policy_size) {
				SetResult(it->second.result, value, policy_size, get_policy);
				it->second.generation = NextGeneration(shard);
			}
			shard.lru.splice(shard.lru.begin(), shard.lru, it->second.lru_position);
			return;
		}

		if (shard.entries.size() >= shard.capacity) {
			// mapノードとpolicyの確保済み領域を再利用し、定常状態での
			// Storeごとのヒープ確保・解放を避ける。
			const Key oldest = shard.lru.back();
			auto node = shard.entries.extract(oldest);
			auto& entry = node.mapped();
			*entry.lru_position = key;
			shard.lru.splice(shard.lru.begin(), shard.lru, entry.lru_position);
			node.key() = key;
			SetResult(entry.result, value, policy_size, get_policy);
			entry.generation = NextGeneration(shard);
			shard.entries.insert(std::move(node));
			return;
		}

		shard.lru.push_front(key);
		Entry entry;
		entry.lru_position = shard.lru.begin();
		SetResult(entry.result, value, policy_size, get_policy);
		entry.generation = NextGeneration(shard);
		shard.entries.emplace(key, std::move(entry));
	}

private:
	static constexpr size_t kShardCount = 256;

	struct Entry {
		Result result;
		std::list<Key>::iterator lru_position;
		uint64_t generation = 0;
	};

	struct Shard {
		size_t capacity = 0;
		std::shared_mutex mutex;
		std::list<Key> lru;
		std::unordered_map<Key, Entry> entries;
		uint64_t next_generation = 0;
	};

	Shard& GetShard(const Key key)
	{
		const size_t hash = std::hash<Key>{}(key);
		const size_t index = shard_count_ == kShardCount
			? hash & (kShardCount - 1)
			: hash % shard_count_;
		return shards_[index];
	}

	template <typename PolicyGetter>
	static void SetResult(Result& result, const float value, const size_t policy_size, PolicyGetter& get_policy)
	{
		result.value = value;
		result.policy.resize(policy_size);
		for (size_t i = 0; i < policy_size; ++i)
			result.policy[i] = get_policy(i);
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
