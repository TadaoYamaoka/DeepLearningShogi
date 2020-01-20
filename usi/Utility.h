#pragma once

#include <chrono>

typedef std::chrono::high_resolution_clock game_clock;

// 消費時間の算出
inline double GetSpendTime(const game_clock::time_point& start_time) {
	return std::chrono::duration_cast<std::chrono::milliseconds>(game_clock::now() - start_time).count() / 1000.0;
}
