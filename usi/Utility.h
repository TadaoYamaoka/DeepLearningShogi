#pragma once

#include <chrono>

typedef std::chrono::high_resolution_clock ray_clock;

// 消費時間の算出
inline double GetSpendTime(const ray_clock::time_point& start_time) {
	return std::chrono::duration_cast<std::chrono::milliseconds>(ray_clock::now() - start_time).count() / 1000.0;
}
