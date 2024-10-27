﻿#include <numeric>
#include <omp.h>
#include "cppshogi.h"

void init() {
	initTable();
	Position::initZobrist();
	HuffmanCodedPos::init();
}

// make result
inline float make_result(const uint8_t result, const Color color) {
	const GameResult gameResult = (GameResult)(result & 0x3);
	if (gameResult == Draw)
		return 0.5f;

	if ((color == Black && gameResult == BlackWin) ||
		(color == White && gameResult == WhiteWin)) {
		return 1.0f;
	}
	else {
		return 0.0f;
	}
}
template<typename T>
inline T is_sennichite(const uint8_t result) {
	return static_cast<T>(result & GAMERESULT_SENNICHITE ? 1 : 0);
}
template<typename T>
inline T is_nyugyoku(const uint8_t result) {
	return static_cast<T>(result & GAMERESULT_NYUGYOKU ? 1 : 0);
}

void __hcpe_decode_with_value(const size_t len, char* ndhcpe, char* ndfeatures1, char* ndfeatures2, char* ndmove, char* ndresult, char* ndvalue) {
	HuffmanCodedPosAndEval* hcpe = reinterpret_cast<HuffmanCodedPosAndEval*>(ndhcpe);
	features1_t* features1 = reinterpret_cast<features1_t*>(ndfeatures1);
	features2_t* features2 = reinterpret_cast<features2_t*>(ndfeatures2);
	int64_t* move = reinterpret_cast<int64_t*>(ndmove);
	float* result = reinterpret_cast<float*>(ndresult);
	float* value = reinterpret_cast<float*>(ndvalue);

	// set all zero
	std::fill_n((float*)features1, sizeof(features1_t) / sizeof(float) * len, 0.0f);
	std::fill_n((float*)features2, sizeof(features2_t) / sizeof(float) * len, 0.0f);

	Position position;
	for (size_t i = 0; i < len; i++, hcpe++, features1++, features2++, value++, move++, result++) {
		position.set(hcpe->hcp);

		// input features
		make_input_features(position, *features1, *features2);

		// move
		*move = make_move_label(hcpe->bestMove16, position.turn());

		// game result
		*result = make_result(hcpe->gameResult, position.turn());

		// eval
		*value = score_to_value((Score)hcpe->eval);
	}
}

void __hcpe2_decode_with_value(const size_t len, char* ndhcpe2, char* ndfeatures1, char* ndfeatures2, char* ndmove, char* ndresult, char* ndvalue, char* ndaux) {
	HuffmanCodedPosAndEval2* hcpe = reinterpret_cast<HuffmanCodedPosAndEval2*>(ndhcpe2);
	features1_t* features1 = reinterpret_cast<features1_t*>(ndfeatures1);
	features2_t* features2 = reinterpret_cast<features2_t*>(ndfeatures2);
	int64_t* move = reinterpret_cast<int64_t*>(ndmove);
	float* result = reinterpret_cast<float*>(ndresult);
	float* value = reinterpret_cast<float*>(ndvalue);
	auto aux = reinterpret_cast<float(*)[2]>(ndaux);

	// set all zero
	std::fill_n((float*)features1, sizeof(features1_t) / sizeof(float) * len, 0.0f);
	std::fill_n((float*)features2, sizeof(features2_t) / sizeof(float) * len, 0.0f);

	Position position;
	for (size_t i = 0; i < len; i++, hcpe++, features1++, features2++, value++, move++, result++, aux++) {
		position.set(hcpe->hcp);

		// input features
		make_input_features(position, *features1, *features2);

		// move
		*move = make_move_label(hcpe->bestMove16, position.turn());

		// game result
		*result = make_result(hcpe->result, position.turn());

		// eval
		*value = score_to_value((Score)hcpe->eval);

		// sennichite
		(*aux)[0] = is_sennichite<float>(hcpe->result);

		// nyugyoku
		(*aux)[1] = is_nyugyoku<float>(hcpe->result);
	}
}

std::vector<TrainingData> trainingData;
// 重複チェック用 局面に対応するtrainingDataのインデックスを保持
std::unordered_map<HuffmanCodedPos, unsigned int> duplicates;

void __hcpe3_create_cache(const std::string& filepath) {
	std::ofstream ofs(filepath, std::ios::binary);

	// インデックス部
	// 局面数
	const size_t num = trainingData.size();
	ofs.write((const char*)&num, sizeof(num));
	// 各局面の開始位置
	size_t pos = sizeof(num) + sizeof(pos) * num;
	for (const auto& hcpe3 : trainingData) {
		ofs.write((const char*)&pos, sizeof(pos));
		pos += sizeof(Hcpe3CacheBody) + sizeof(Hcpe3CacheCandidate) * hcpe3.candidates.size();
	}

	// ボディ部
	for (const auto& hcpe3 : trainingData) {
		Hcpe3CacheBody body{
			hcpe3.hcp,
			hcpe3.value,
			hcpe3.result,
			hcpe3.count
		};
		ofs.write((const char*)&body, sizeof(body));

		for (const auto kv : hcpe3.candidates) {
			Hcpe3CacheCandidate candidate{
				kv.first,
				kv.second
			};
			ofs.write((const char*)&candidate, sizeof(candidate));
		}
	}

	// メモリ開放
	// trainingDataを開放してしまうため、キャッシュを作成した場合、データはキャッシュから読み込むこと
	trainingData.clear();
	trainingData.shrink_to_fit();
	duplicates.clear();
	std::unordered_map<HuffmanCodedPos, unsigned int>(duplicates).swap(duplicates);
}

// hcpe3キャッシュ
std::ifstream* cache;
std::vector<size_t> cache_pos;
size_t __hcpe3_load_cache(const std::string& filepath) {
	cache = new std::ifstream(filepath, std::ios::binary);
	size_t num;
	cache->read((char*)&num, sizeof(num));
	cache_pos.resize(num + 1);
	cache->read((char*)cache_pos.data(), sizeof(size_t) * num);
	cache->seekg(0, std::ios_base::end);
	cache_pos[num] = cache->tellg();
	return num;
}

size_t __hcpe3_get_cache_num() {
	return cache_pos.size() > 0 ? cache_pos.size() - 1 : 0;
}

TrainingData get_cache(const size_t i) {
	const size_t pos = cache_pos[i];
	const size_t candidateNum = ((cache_pos[i + 1] - pos) - sizeof(Hcpe3CacheBody)) / sizeof(Hcpe3CacheCandidate);
	struct Hcpe3CacheBuf {
		Hcpe3CacheBody body;
		Hcpe3CacheCandidate candidates[MaxLegalMoves];
	} buf;
	cache->seekg(pos, std::ios_base::beg);
	cache->read((char*)&buf, sizeof(Hcpe3CacheBody) + sizeof(Hcpe3CacheCandidate) * candidateNum);
	return TrainingData(buf.body, buf.candidates, candidateNum);
}

TrainingData get_cache_with_lock(const size_t i) {
	const size_t pos = cache_pos[i];
	const size_t candidateNum = ((cache_pos[i + 1] - pos) - sizeof(Hcpe3CacheBody)) / sizeof(Hcpe3CacheCandidate);
	struct Hcpe3CacheBuf {
		Hcpe3CacheBody body;
		Hcpe3CacheCandidate candidates[MaxLegalMoves];
	} buf;
	#pragma omp critical
	{
		cache->seekg(pos, std::ios_base::beg);
		cache->read((char*)&buf, sizeof(Hcpe3CacheBody) + sizeof(Hcpe3CacheCandidate) * candidateNum);
	}
	return TrainingData(buf.body, buf.candidates, candidateNum);
}

// hcpe形式の指し手をone-hotの方策として読み込む
size_t load_hcpe(const std::string& filepath, std::ifstream& ifs, bool use_average, const double eval_scale, size_t& len) {
	for (int p = 0; ifs; ++p) {
		HuffmanCodedPosAndEval hcpe;
		ifs.read((char*)&hcpe, sizeof(HuffmanCodedPosAndEval));
		if (ifs.eof()) {
			break;
		}

		const float value = score_to_value((Score)(hcpe.eval * eval_scale));
		if (use_average) {
			auto ret = duplicates.emplace(hcpe.hcp, trainingData.size());
			if (ret.second) {
				auto& data = trainingData.emplace_back(
					hcpe.hcp,
					value,
					make_result(hcpe.gameResult, hcpe.hcp.color())
				);
				data.candidates[hcpe.bestMove16] = 1;
			}
			else {
				// 重複データの場合、加算する(hcpe3_decode_with_valueで平均にする)
				auto& data = trainingData[ret.first->second];
				data.value += value;
				data.result += make_result(hcpe.gameResult, hcpe.hcp.color());
				data.candidates[hcpe.bestMove16] += 1;
				data.count++;
			}
		}
		else {
			auto& data = trainingData.emplace_back(
				hcpe.hcp,
				value,
				make_result(hcpe.gameResult, hcpe.hcp.color())
			);
			data.candidates[hcpe.bestMove16] = 1;
		}
		++len;
	}

	return trainingData.size();
}

template <bool add>
inline void visits_to_proberbility(TrainingData& data, const std::vector<MoveVisits>& candidates, const double temperature) {
	if (candidates.size() == 1) {
		// one-hot
		const auto& moveVisits = candidates[0];
		if constexpr (add)
			data.candidates[moveVisits.move16] += 1.0f;
		else
			data.candidates[moveVisits.move16] = 1.0f;
	}
	else if (temperature == 0) {
		// greedy
		const auto itr = std::max_element(candidates.begin(), candidates.end(), [](const MoveVisits& a, const MoveVisits& b) { return a.visitNum < b.visitNum; });
		const MoveVisits& moveVisits = *itr;
		if constexpr (add)
			data.candidates[moveVisits.move16] += 1.0f;
		else
			data.candidates[moveVisits.move16] = 1.0f;
	}
	else if (temperature == 1) {
		const float sum_visitNum = (float)std::accumulate(candidates.begin(), candidates.end(), 0, [](int acc, const MoveVisits& move_visits) { return acc + move_visits.visitNum; });
		for (const auto& moveVisits : candidates) {
			const float proberbility = (float)moveVisits.visitNum / sum_visitNum;
			if constexpr (add)
				data.candidates[moveVisits.move16] += proberbility;
			else
				data.candidates[moveVisits.move16] = proberbility;
		}
	}
	else {
		double exponentiated_visits[593];
		double sum = 0;
		for (size_t i = 0; i < candidates.size(); i++) {
			const auto& moveVisits = candidates[i];
			const auto new_visits = std::pow(moveVisits.visitNum, 1.0 / temperature);
			exponentiated_visits[i] = new_visits;
			sum += new_visits;
		}
		for (size_t i = 0; i < candidates.size(); i++) {
			const auto& moveVisits = candidates[i];
			const float proberbility = (float)(exponentiated_visits[i] / sum);
			if constexpr (add)
				data.candidates[moveVisits.move16] += proberbility;
			else
				data.candidates[moveVisits.move16] = proberbility;
		}
	}
}

// フォーマット自動判別
bool is_hcpe(std::ifstream& ifs) {
	if (ifs.tellg() % sizeof(HuffmanCodedPosAndEval) == 0) {
		// 最後のデータがhcpeであるかで判別
		ifs.seekg(-sizeof(HuffmanCodedPosAndEval), std::ios_base::end);
		HuffmanCodedPosAndEval hcpe;
		ifs.read((char*)&hcpe, sizeof(HuffmanCodedPosAndEval));
		if (hcpe.hcp.isOK() && hcpe.bestMove16 >= 1 && hcpe.bestMove16 <= 26703) {
			return true;
		}
	}
	return false;
}

// hcpe3形式のデータを読み込み、ランダムアクセス可能なように加工し、trainingDataに保存する
// 複数回呼ぶことで、複数ファイルの読み込みが可能
size_t __load_hcpe3(const std::string& filepath, bool use_average, double a, double temperature, size_t& len) {
	std::ifstream ifs(filepath, std::ifstream::binary | std::ios::ate);
	if (!ifs) return trainingData.size();

	const double eval_scale = a == 0 ? 1 : 756.0864962951762 / a;

	// フォーマット自動判別
	// hcpeの場合は、指し手をone-hotの方策として読み込む
	if (is_hcpe(ifs)) {
		ifs.seekg(std::ios_base::beg);
		return load_hcpe(filepath, ifs, use_average, eval_scale, len);
	}
	ifs.seekg(std::ios_base::beg);

	std::vector<MoveVisits> candidates;

	for (int p = 0; ifs; ++p) {
		HuffmanCodedPosAndEval3 hcpe3;
		ifs.read((char*)&hcpe3, sizeof(HuffmanCodedPosAndEval3));
		if (ifs.eof()) {
			break;
		}
		assert(hcpe3.moveNum <= 513);

		// 開始局面
		Position pos;
		if (!pos.set(hcpe3.hcp)) {
			std::stringstream ss("INCORRECT_HUFFMAN_CODE at ");
			ss << filepath << "(" << p << ")";
			throw std::runtime_error(ss.str());
		}
		StateListPtr states{ new std::deque<StateInfo>(1) };

		for (int i = 0; i < hcpe3.moveNum; ++i) {
			MoveInfo moveInfo;
			ifs.read((char*)&moveInfo, sizeof(MoveInfo));
			assert(moveInfo.candidateNum <= 593);

			// candidateNum==0の手は読み飛ばす
			if (moveInfo.candidateNum > 0) {
				candidates.resize(moveInfo.candidateNum);
				ifs.read((char*)candidates.data(), sizeof(MoveVisits) * moveInfo.candidateNum);

				const auto hcp = pos.toHuffmanCodedPos();
				const float value = score_to_value((Score)(moveInfo.eval * eval_scale));
				if (use_average) {
					auto ret = duplicates.emplace(hcp, trainingData.size());
					if (ret.second) {
						auto& data = trainingData.emplace_back(
							hcp,
							value,
							make_result(hcpe3.result, pos.turn())
						);
						visits_to_proberbility<false>(data, candidates, temperature);
					}
					else {
						// 重複データの場合、加算する(hcpe3_decode_with_valueで平均にする)
						auto& data = trainingData[ret.first->second];
						data.value += value;
						data.result += make_result(hcpe3.result, pos.turn());
						visits_to_proberbility<true>(data, candidates, temperature);
						data.count++;

					}
				}
				else {
					auto& data = trainingData.emplace_back(
						hcp,
						value,
						make_result(hcpe3.result, pos.turn())
					);
					visits_to_proberbility<false>(data, candidates, temperature);
				}
				++len;
			}

			const Move move = move16toMove((Move)moveInfo.selectedMove16, pos);
			pos.doMove(move, states->emplace_back(StateInfo()));
		}
	}

	return trainingData.size();
}

size_t __hcpe3_patch_with_hcpe(const std::string& filepath, size_t& add_len) {
	std::ifstream ifs(filepath, std::ifstream::binary);
	size_t sum_len = 0;
	while (ifs) {
		HuffmanCodedPosAndEval hcpe;
		ifs.read((char*)&hcpe, sizeof(HuffmanCodedPosAndEval));
		if (ifs.eof()) {
			break;
		}
		bool found = false;
		const float value = score_to_value((Score)hcpe.eval);
		for (auto& data : trainingData) {
			if (data.hcp == hcpe.hcp) {
				found = true;
				data.count = 1;
				data.value = value;
				data.result = make_result(hcpe.gameResult, hcpe.hcp.color());
				data.candidates.clear();
				data.candidates[hcpe.bestMove16] = 1;
			}
		}
		if (!found) {
			auto& data = trainingData.emplace_back(
				hcpe.hcp,
				value,
				make_result(hcpe.gameResult, hcpe.hcp.color())
			);
			data.candidates[hcpe.bestMove16] = 1;
			++add_len;
		}
		++sum_len;
	}
	return sum_len;
}

// load_hcpe3で読み込み済みのtrainingDataから、インデックスを使用してサンプリングする
// 重複データは平均化する
void __hcpe3_decode_with_value(const size_t len, char* ndindex, char* ndfeatures1, char* ndfeatures2, char* ndprobability, char* ndresult, char* ndvalue) {
	unsigned int* index = reinterpret_cast<unsigned int*>(ndindex);
	features1_t* features1 = reinterpret_cast<features1_t*>(ndfeatures1);
	features2_t* features2 = reinterpret_cast<features2_t*>(ndfeatures2);
	auto probability = reinterpret_cast<float(*)[9 * 9 * MAX_MOVE_LABEL_NUM]>(ndprobability);
	float* result = reinterpret_cast<float*>(ndresult);
	float* value = reinterpret_cast<float*>(ndvalue);

	// set all zero
	std::fill_n((float*)features1, sizeof(features1_t) / sizeof(float) * len, 0.0f);
	std::fill_n((float*)features2, sizeof(features2_t) / sizeof(float) * len, 0.0f);
	std::fill_n((float*)probability, 9 * 9 * MAX_MOVE_LABEL_NUM * len, 0.0f);

	#pragma omp parallel for num_threads(2) if (len > 1)
	for (int64_t i = 0; i < len; i++) {
		const auto& hcpe3 = cache ? (len > 1 ? get_cache_with_lock(index[i]) : get_cache(index[i])) : trainingData[index[i]];

		Position position;
		position.set(hcpe3.hcp);

		// input features
		make_input_features(position, features1[i], features2[i]);

		// move probability
		for (const auto kv : hcpe3.candidates) {
			const auto label = make_move_label(kv.first, position.turn());
			assert(label < 9 * 9 * MAX_MOVE_LABEL_NUM);
			probability[i][label] = kv.second / hcpe3.count;
		}

		// game result
		result[i] = hcpe3.result / hcpe3.count;

		// eval
		value[i] = hcpe3.value / hcpe3.count;
	}
}

// load_hcpe3で読み込み済みのtrainingDataから、インデックスを指定してhcpeを取り出す
void __hcpe3_get_hcpe(const size_t index, char* ndhcpe) {
	HuffmanCodedPosAndEval* hcpe = reinterpret_cast<HuffmanCodedPosAndEval*>(ndhcpe);

	const auto& hcpe3 = cache ? get_cache(index) : trainingData[index];

	hcpe->hcp = hcpe3.hcp;
	float max_prob = FLT_MIN;
	for (const auto kv : hcpe3.candidates) {
		const auto& move16 = kv.first;
		const auto& prob = kv.second;
		if (prob > max_prob) {
			hcpe->bestMove16 = move16;
			max_prob = prob;
		}
	}
	hcpe->eval = s16(-logf(1.0f / (hcpe3.value / hcpe3.count) - 1.0f) * 756.0f);
	const auto result = (hcpe3.result / hcpe3.count);
	if (result < 0.5f) {
		hcpe->gameResult = hcpe3.hcp.color() == Black ? WhiteWin : BlackWin;
	}
	else if (result > 0.5f) {
		hcpe->gameResult = hcpe3.hcp.color() == Black ? BlackWin : WhiteWin;
	}
	else {
		hcpe->gameResult = Draw;
	}
}

// evalの補正用データ準備
std::vector<int> eval;
std::vector<float> result;
size_t __load_evalfix(const std::string& filepath) {
	eval.clear();
	result.clear();

	std::ifstream ifs(filepath, std::ifstream::binary | std::ios::ate);
	if (!ifs) return 0;

	// フォーマット自動判別
	bool hcpe3 = !is_hcpe(ifs);
	ifs.seekg(std::ios_base::beg);

	if (hcpe3) {
		for (int p = 0; ifs; ++p) {
			HuffmanCodedPosAndEval3 hcpe3;
			ifs.read((char*)&hcpe3, sizeof(HuffmanCodedPosAndEval3));
			if (ifs.eof()) {
				break;
			}
			assert(hcpe3.moveNum <= 513);

			// 開始局面
			Position pos;
			if (!pos.set(hcpe3.hcp)) {
				std::stringstream ss("INCORRECT_HUFFMAN_CODE at ");
				ss << filepath << "(" << p << ")";
				throw std::runtime_error(ss.str());
			}
			StateListPtr states{ new std::deque<StateInfo>(1) };

			for (int i = 0; i < hcpe3.moveNum; ++i) {
				MoveInfo moveInfo;
				ifs.read((char*)&moveInfo, sizeof(MoveInfo));
				assert(moveInfo.candidateNum <= 593);

				// candidateNum==0の手は読み飛ばす
				if (moveInfo.candidateNum > 0) {
					ifs.seekg(sizeof(MoveVisits) * moveInfo.candidateNum, std::ios_base::cur);
					// 詰みは除く
					if (std::abs(moveInfo.eval) < 30000) {
						eval.emplace_back(moveInfo.eval);
						result.emplace_back(make_result(hcpe3.result, pos.turn()));
					}
				}

				assert(moveInfo.selectedMove16 <= 0x7fff);
				const Move move = move16toMove((Move)moveInfo.selectedMove16, pos);
				pos.doMove(move, states->emplace_back(StateInfo()));
			}
		}
	}
	else {
		// hcpeフォーマット
		for (int p = 0; ifs; ++p) {
			HuffmanCodedPosAndEval hcpe;
			ifs.read((char*)&hcpe, sizeof(HuffmanCodedPosAndEval));
			if (ifs.eof()) {
				break;
			}

			// 詰みは除く
			if (std::abs(hcpe.eval) < 30000) {
				eval.emplace_back(hcpe.eval);
				result.emplace_back(make_result(hcpe.gameResult, hcpe.hcp.color()));
			}
		}
	}

	return result.size();
}

void __hcpe3_prepare_evalfix(char* ndeval, char* ndresult) {
	std::copy(eval.begin(), eval.end(), reinterpret_cast<int*>(ndeval));
	std::copy(result.begin(), result.end(), reinterpret_cast<float*>(ndresult));
}

// 2つのhcpe3キャッシュをマージする
void __hcpe3_merge_cache(const std::string& file1, const std::string& file2, const std::string& out) {
	// file2のhcpをキーとした辞書を作成
	std::ifstream cache2(file2, std::ios::binary);
	size_t num2;
	cache2.read((char*)&num2, sizeof(num2));
	std::vector<size_t> cache2_pos(num2 + 1);
	cache2.read((char*)cache2_pos.data(), sizeof(size_t) * num2);
	cache2.seekg(0, std::ios_base::end);
	cache2_pos[num2] = cache2.tellg();

	std::unordered_map<HuffmanCodedPos, std::pair<size_t, size_t>> cache2_map;
	for (size_t i = 0; i < num2; ++i) {
		auto pos = cache2_pos[i];
		cache2.seekg(pos, std::ios_base::beg);
		HuffmanCodedPos hcp;
		cache2.read((char*)&hcp, sizeof(HuffmanCodedPos));
		cache2_map[hcp] = std::make_pair(pos, cache2_pos[i + 1]);
	}

	// file1のインデックス読み込み
	std::ifstream cache1(file1, std::ios::binary);
	size_t num1;
	cache1.read((char*)&num1, sizeof(num1));
	std::vector<size_t> cache1_pos(num1 + 1);
	cache1.read((char*)cache1_pos.data(), sizeof(size_t) * num1);
	cache1.seekg(0, std::ios_base::end);
	cache1_pos[num1] = cache1.tellg();

	// 重複しない局面数をカウントしてインデックスのサイズを計算
	size_t num_out = num2;
	for (size_t i = 0; i < num1; ++i) {
		auto pos = cache1_pos[i];
		cache1.seekg(pos, std::ios_base::beg);
		HuffmanCodedPos hcp;
		cache1.read((char*)&hcp, sizeof(HuffmanCodedPos));
		// file2に存在するか
		if (cache2_map.find(hcp) == cache2_map.end()) {
			num_out++;
		}
	}

	std::cout << "file1 position num = " << num1 << std::endl;
	std::cout << "file2 position num = " << num2 << std::endl;

	std::ofstream ofs(out, std::ios::binary);

	// インデックスの領域をシーク
	ofs.seekp(sizeof(num_out) + sizeof(size_t) * num_out, std::ios_base::beg);

	std::vector<size_t> out_pos;
	out_pos.reserve(num_out);

	struct Hcpe3CacheBuf {
		Hcpe3CacheBody body;
		Hcpe3CacheCandidate candidates[MaxLegalMoves];
	};

	// file1をシーケンシャルに処理
	for (size_t i = 0; i < num1; ++i) {
		auto pos1 = cache1_pos[i];
		const size_t candidate_num1 = ((cache1_pos[i + 1] - pos1) - sizeof(Hcpe3CacheBody)) / sizeof(Hcpe3CacheCandidate);
		Hcpe3CacheBuf buf1;
		cache1.seekg(pos1, std::ios_base::beg);
		cache1.read((char*)&buf1, sizeof(Hcpe3CacheBody) + sizeof(Hcpe3CacheCandidate) * candidate_num1);

		out_pos.emplace_back(ofs.tellp());

		auto itr = cache2_map.find(buf1.body.hcp);
		if (itr == cache2_map.end()) {
			// file2の辞書に局面が存在しない場合、そのまま出力
			ofs.write((char*)&buf1, sizeof(Hcpe3CacheBody) + sizeof(Hcpe3CacheCandidate) * candidate_num1);
		}
		else {
			// file2の辞書に局面が存在する場合マージ
			auto pos2 = itr->second.first;
			const size_t candidate_num2 = ((itr->second.second - pos2) - sizeof(Hcpe3CacheBody)) / sizeof(Hcpe3CacheCandidate);
			Hcpe3CacheBuf buf2;
			cache2.seekg(pos2, std::ios_base::beg);
			cache2.read((char*)&buf2, sizeof(Hcpe3CacheBody) + sizeof(Hcpe3CacheCandidate) * candidate_num2);

			buf1.body.value += buf2.body.value;
			buf1.body.result += buf2.body.result;
			buf1.body.count += buf2.body.count;

			std::unordered_map<u16, float> candidate_map;
			for (size_t j = 0; j < candidate_num1; ++j) {
				candidate_map[buf1.candidates[j].move16] = buf1.candidates[j].prob;
			}
			for (size_t j = 0; j < candidate_num2; ++j) {
				auto ret = candidate_map.try_emplace(buf2.candidates[j].move16, buf2.candidates[j].prob);
				if (!ret.second) {
					ret.first->second += buf2.candidates[j].prob;
				}
			}
			size_t candidate_i = 0;
			for (const auto& kv : candidate_map) {
				buf1.candidates[candidate_i].move16 = kv.first;
				buf1.candidates[candidate_i].prob = kv.second;
				candidate_i++;
			}

			// 出力
			ofs.write((char*)&buf1, sizeof(Hcpe3CacheBody) + sizeof(Hcpe3CacheCandidate) * candidate_i);

			// 辞書から削除
			cache2_map.erase(itr);
		}
	}

	// 未出力のfile2の局面を出力
	for (const auto& kv : cache2_map) {
		out_pos.emplace_back(ofs.tellp());

		auto pos2 = kv.second.first;
		const size_t candidate_num2 = ((kv.second.second - pos2) - sizeof(Hcpe3CacheBody)) / sizeof(Hcpe3CacheCandidate);
		Hcpe3CacheBuf buf2;
		cache2.seekg(pos2, std::ios_base::beg);
		cache2.read((char*)&buf2, sizeof(Hcpe3CacheBody) + sizeof(Hcpe3CacheCandidate) * candidate_num2);

		ofs.write((char*)&buf2, sizeof(Hcpe3CacheBody) + sizeof(Hcpe3CacheCandidate) * candidate_num2);
	}
	assert(out_pos.size() == num_out);

	// インデックスの出力
	ofs.seekp(0, std::ios_base::beg);
	ofs.write((char*)&num_out, sizeof(num_out));
	ofs.write((char*)out_pos.data(), sizeof(size_t) * num_out);

	assert(ofs.tellp() == out_pos[0]);

    std::cout << "out position num = " << num_out << std::endl;
}
