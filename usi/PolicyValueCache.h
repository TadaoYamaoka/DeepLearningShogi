#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.hpp"

class PolicyValueCache {
public:
	struct Result {
		Result(const float value, std::vector<float>&& policy)
			: value(value), policy(std::move(policy)) {}

		float value;
		std::vector<float> policy;
	};

	using ResultPtr = std::shared_ptr<const Result>;

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

	bool Lookup(const Key key, const size_t policy_size, ResultPtr& result)
	{
		if (!IsEnabled())
			return false;

		auto& shard = GetShard(key);
		ResultPtr cached;
		{
			std::shared_lock<std::shared_mutex> lock(shard.mutex);
			auto it = shard.entries.find(key);
			if (it == shard.entries.end() || it->second.result->policy.size() != policy_size)
				return false;
			cached = it->second.result;
		}

		// ヒット処理を待たせないため、LRU順の更新は排他ロックを
		// 即時取得できた場合だけ行う。
		std::unique_lock<std::shared_mutex> lock(shard.mutex, std::try_to_lock);
		if (lock.owns_lock()) {
			auto it = shard.entries.find(key);
			if (it != shard.entries.end() && it->second.result == cached)
				shard.lru.splice(shard.lru.begin(), shard.lru, it->second.lru_position);
		}

		result = std::move(cached);
		return true;
	}

	void Store(const Key key, const float value, std::vector<float>&& policy)
	{
		if (!IsEnabled())
			return;

		auto& shard = GetShard(key);
		std::unique_lock<std::shared_mutex> lock(shard.mutex);
		auto it = shard.entries.find(key);
		if (it != shard.entries.end()) {
			// 同じ局面を複数GPUが推論した場合は先着結果を維持する。
			// policy長が異なる場合だけ不整合エントリとして置換する。
			if (it->second.result->policy.size() != policy.size())
				it->second.result = std::make_shared<const Result>(value, std::move(policy));
			shard.lru.splice(shard.lru.begin(), shard.lru, it->second.lru_position);
			return;
		}

		auto result = std::make_shared<const Result>(value, std::move(policy));
		shard.lru.push_front(key);
		shard.entries.emplace(key, Entry{ std::move(result), shard.lru.begin() });
		if (shard.entries.size() > shard.capacity) {
			const Key oldest = shard.lru.back();
			shard.entries.erase(oldest);
			shard.lru.pop_back();
		}
	}

private:
	static constexpr size_t kShardCount = 256;

	struct Entry {
		ResultPtr result;
		std::list<Key>::iterator lru_position;
	};

	struct Shard {
		size_t capacity = 0;
		std::shared_mutex mutex;
		std::list<Key> lru;
		std::unordered_map<Key, Entry> entries;
	};

	Shard& GetShard(const Key key)
	{
		const size_t hash = std::hash<Key>{}(key);
		const size_t index = shard_count_ == kShardCount
			? hash & (kShardCount - 1)
			: hash % shard_count_;
		return shards_[index];
	}

	size_t capacity_ = 0;
	size_t shard_count_ = 0;
	std::array<Shard, kShardCount> shards_;
};
