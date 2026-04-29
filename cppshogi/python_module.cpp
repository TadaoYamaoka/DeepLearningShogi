#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <vector>
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
    ofs.close();

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

            const Move move = move16toMove((Move)moveInfo.selectedMove16, pos);

            // candidateNum==0の手は読み飛ばす
            if (moveInfo.candidateNum > 0) {
                candidates.resize(moveInfo.candidateNum);
                ifs.read((char*)candidates.data(), sizeof(MoveVisits) * moveInfo.candidateNum);

                // 繰り返しと優越/劣等局面になる手は除く
                const auto draw = pos.moveIsDraw(move, 16);
                if (draw != RepetitionDraw && draw != RepetitionSuperior && draw != RepetitionInferior) {
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
            }

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
    size_t* index = reinterpret_cast<size_t*>(ndindex);
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

                assert(moveInfo.selectedMove16 <= 0x7fff);
                const Move move = move16toMove((Move)moveInfo.selectedMove16, pos);

                // candidateNum==0の手は読み飛ばす
                if (moveInfo.candidateNum > 0) {
                    ifs.seekg(sizeof(MoveVisits) * moveInfo.candidateNum, std::ios_base::cur);
                    // 詰みと繰り返しと優越/劣等局面になる手は除く
                    const auto draw = pos.moveIsDraw(move, 16);
                    if (std::abs(moveInfo.eval) < 30000 && draw != RepetitionDraw && draw != RepetitionSuperior && draw != RepetitionInferior) {
                        eval.emplace_back(moveInfo.eval);
                        result.emplace_back(make_result(hcpe3.result, pos.turn()));
                    }
                }

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

// キャッシュの方策と価値をモデルで再評価して加重平均を求める
// モデルの推論結果を受け取る
// 事前にキャッシュがロードされていること
// alpha: 加重平均の係数
// dropoff: モデルの推論結果の方策の確率をトップから何%低下までを採用するか
// temperature: softmax温度パラメータ
void __hcpe3_cache_re_eval(const size_t len, char* ndindex, char* ndlogits, char* ndvalue, const float alpha_p, const float alpha_v, const float alpha_r, const float dropoff, const int limit_candidates, const float temperature) {
    size_t* index = reinterpret_cast<size_t*>(ndindex);
    auto logits = reinterpret_cast<float(*)[9 * 9 * MAX_MOVE_LABEL_NUM]>(ndlogits);
    float* values = reinterpret_cast<float*>(ndvalue);

    const size_t start_index = trainingData.size();
    trainingData.resize(trainingData.size() + len);

    const auto softmax = [](std::vector<float>& probabilities, const float temperature) {
        float max = 0.0f;
        for (int i = 0; i < probabilities.size(); i++) {
            float& x = probabilities[i];
            x /= temperature;
            if (x > max) {
                max = x;
            }
        }
        // オーバーフローを防止するため最大値で引く
        float sum = 0.0f;
        for (int i = 0; i < probabilities.size(); i++) {
            float& x = probabilities[i];
            x = expf(x - max);
            sum += x;
        }
        // normalize
        for (int i = 0; i < probabilities.size(); i++) {
            float& x = probabilities[i];
            x /= sum;
        }
    };

    #pragma omp parallel for num_threads(4)
    for (int64_t i = 0; i < len; i++) {
        auto& hcpe3 = trainingData[i + start_index];
        hcpe3 = get_cache_with_lock(index[i]);

        if (alpha_p > 0) {
            Position pos;
            pos.set(hcpe3.hcp);

            // 合法手でフィルタする
            MoveList<Legal> ml(pos);
            std::vector<float> probabilities;
            probabilities.reserve(ml.size());
            std::vector<u16> legal_moves;
            legal_moves.reserve(ml.size());
            for (; !ml.end(); ++ml) {
                const u16 move16 = (u16)ml.move().proFromAndTo();
                const int move_label = make_move_label(move16, pos.turn());
                probabilities.emplace_back(logits[i][move_label]);
                legal_moves.emplace_back(move16);
            }
            // softmax
            softmax(probabilities, temperature);

            // 確率でフィルタする
            float threshold = 1.0f / MaxLegalMoves;
            if (probabilities.size() > limit_candidates) {
                std::vector<float> sorted_probabilities = probabilities;
                std::nth_element(sorted_probabilities.begin(), sorted_probabilities.begin() + (limit_candidates - 1), sorted_probabilities.end(), std::greater<float>());
                if (sorted_probabilities[limit_candidates - 1] > threshold)
                    threshold = sorted_probabilities[limit_candidates - 1];
                const auto max_it = std::max_element(sorted_probabilities.begin(), sorted_probabilities.begin() + limit_candidates);
                if (*max_it - dropoff > threshold)
                    threshold = *max_it - dropoff;
            }
            else {
                const auto max_it = std::max_element(probabilities.begin(), probabilities.end());
                if (*max_it - dropoff > threshold)
                    threshold = *max_it - dropoff;
            }
            float sum = 0;
            std::unordered_map<u16, float> filtered_probabilities;
            for (size_t j = 0; j < probabilities.size(); ++j) {
                if (probabilities[j] >= threshold) {
                    filtered_probabilities[legal_moves[j]] = probabilities[j];
                    sum += probabilities[j];
                }
            }
            // 正規化
            for (auto& probability : filtered_probabilities) {
                probability.second /= sum;
            }
            assert(filtered_probabilities.size() > 0);

            // マージ
            if (alpha_p == 1) {
                hcpe3.candidates = std::move(filtered_probabilities);
            }
            else {
                for (auto& kv1 : hcpe3.candidates) {
                    auto itr2 = filtered_probabilities.find(kv1.first);
                    if (itr2 == filtered_probabilities.end()) {
                        kv1.second = kv1.second / hcpe3.count * (1 - alpha_p);
                    }
                    else {
                        kv1.second = kv1.second / hcpe3.count * (1 - alpha_p) + itr2->second * alpha_p;
                    }
                }
                for (const auto& kv2 : filtered_probabilities) {
                    auto itr1 = hcpe3.candidates.find(kv2.first);
                    if (itr1 == hcpe3.candidates.end()) {
                        hcpe3.candidates[kv2.first] = kv2.second * alpha_p;
                    }
                }
            }
        }
        else {
            for (auto& kv1 : hcpe3.candidates) {
                kv1.second /= hcpe3.count;
            }
        }
        if (alpha_v > 0) {
            hcpe3.value = hcpe3.value / hcpe3.count * (1 - alpha_v) + values[i] * alpha_v;
        }
        else {
            hcpe3.value /= hcpe3.count;
        }
        if (alpha_r > 0) {
            hcpe3.result = hcpe3.result / hcpe3.count * (1 - alpha_r) + values[i] * alpha_r;
        }
        else {
            hcpe3.result /= hcpe3.count;
        }
        hcpe3.count = 1;
    }
}

void __hcpe3_reserve_train_data(unsigned int size) {
    trainingData.reserve(trainingData.size() + size);
}

template<typename T>
void printStat(std::vector<T>& values) {
    auto sum = std::accumulate(values.begin(), values.end(), 0.0);
    const auto mean = (double)sum / values.size();

    double variance = std::accumulate(values.begin(), values.end(), 0.0, [mean](double sum, double value) { return sum + std::pow(value - mean, 2); }) / values.size();
    const auto std = std::sqrt(variance);

    const auto min = *std::min_element(values.begin(), values.end());
    const auto max = *std::max_element(values.begin(), values.end());

    std::sort(values.begin(), values.end());

    const auto percentile = [](const std::vector<T>& values, const double q) {
        double q_index = (values.size() - 1) * q;
        size_t q_low = static_cast<size_t>(q_index);
        size_t q_high = q_low + 1;
        return values[q_low] + (q_index - q_low) * (values[q_high] - values[q_low]);
    };

    std::cout << "\tcount " << values.size() << "\n";
    std::cout << "\tmean  " << std::fixed << std::setprecision(4) << mean << "\n";
    std::cout << "\tstd   " << std::fixed << std::setprecision(4) << std << "\n";
    std::cout << "\tmin   " << min << "\n";
    std::cout << "\t25%   " << std::fixed << std::setprecision(4) << percentile(values, 0.25) << "\n";
    std::cout << "\t50%   " << std::fixed << std::setprecision(4) << percentile(values, 0.5) << "\n";
    std::cout << "\t75%   " << std::fixed << std::setprecision(4) << percentile(values, 0.75) << "\n";
    std::cout << "\tmax   " << max << std::endl;
}

void __hcpe3_stat_cache() {
    const auto size = cache_pos.size() - 1;

    std::vector<int> counts;
    counts.reserve(size);
    std::vector<short> candidates;
    candidates.reserve(size);
    std::vector<float> values;
    values.reserve(size);
    std::vector<float> results;
    results.reserve(size);

    for (size_t i = 0; i < size; ++i) {
        const auto data = get_cache(i);
        counts.emplace_back(data.count);
        candidates.emplace_back((short)data.candidates.size());
        values.emplace_back(data.value / data.count);
        results.emplace_back(data.result / data.count);
    }

    std::cout << "counts:\n";
    printStat(counts);
    std::cout << "candidates:\n";
    printStat(candidates);
    std::cout << "values:\n";
    printStat(values);
    std::cout << "results:\n";
    printStat(results);
}

std::pair<int, int> __hcpe3_to_hcpe(const std::string& file1, const std::string& file2) {
    std::ifstream ifs(file1, std::ifstream::binary);
    std::ofstream ofs(file2, std::ifstream::binary);

    std::vector<MoveVisits> candidates;

    int positions = 0;
    int p = 0;
    for (; ifs; ++p) {
        HuffmanCodedPosAndEval3 hcpe3;
        ifs.read((char*)&hcpe3, sizeof(HuffmanCodedPosAndEval3));
        if (ifs.eof()) {
            break;
        }
        assert(hcpe3.moveNum <= 513);

        // 開始局面
        Position pos;
        if (!pos.set(hcpe3.hcp)) {
            std::stringstream ss;
            ss << "INCORRECT_HUFFMAN_CODE at " << file1 << "(" << p << ")";
            throw std::runtime_error(ss.str());
        }
        StateListPtr states{ new std::deque<StateInfo>(1) };

        for (int i = 0; i < hcpe3.moveNum; ++i) {
            MoveInfo moveInfo;
            ifs.read((char*)&moveInfo, sizeof(MoveInfo));
            assert(moveInfo.candidateNum <= 593);

            const Move move = move16toMove((Move)moveInfo.selectedMove16, pos);

            // candidateNum==0の手は読み飛ばす
            if (moveInfo.candidateNum > 0) {
                candidates.resize(moveInfo.candidateNum);
                ifs.read((char*)candidates.data(), sizeof(MoveVisits) * moveInfo.candidateNum);

                // 繰り返しと優越/劣等局面になる手は除く
                const auto draw = pos.moveIsDraw(move, 16);
                if (draw != RepetitionDraw && draw != RepetitionSuperior && draw != RepetitionInferior) {
                    HuffmanCodedPosAndEval hcpe = {};
                    hcpe.hcp = pos.toHuffmanCodedPos();
                    hcpe.eval = moveInfo.eval;
                    hcpe.bestMove16 = moveInfo.selectedMove16;
                    hcpe.gameResult = (GameResult)hcpe3.result;

                    ofs.write((char*)&hcpe, sizeof(hcpe));
                    positions++;
                }
            }

            pos.doMove(move, states->emplace_back(StateInfo()));
        }
    }
    return std::make_pair(p, positions);
}

namespace {

constexpr u16 HCPE3_MAX_MOVE_NUM = 513;
constexpr u16 HCPE3_MAX_CANDIDATE_NUM = MaxLegalMoves - 1;

enum class Hcpe3ReadResult {
    Ok,
    Eof,
    Truncated,
};

struct Hcpe3ParsedMove {
    MoveInfo moveInfo;
    std::vector<MoveVisits> visits;
};

struct Hcpe3ParsedGame {
    HuffmanCodedPos start;
    u8 result;
    u8 opponent;
    std::vector<HuffmanCodedPos> positions;
    std::vector<Hcpe3ParsedMove> moves;
};

struct Hcpe3MergedMove {
    u16 selectedMove16;
    int64_t evalSum;
    uint64_t count;
    std::unordered_map<u16, uint64_t> visitSums;
};

struct Hcpe3MergedGame {
    HuffmanCodedPos start;
    u8 result;
    u8 opponent;
    bool active;
    std::vector<HuffmanCodedPos> positions;
    std::vector<Hcpe3MergedMove> moves;
};

struct Hcpe3SuffixKey {
    HuffmanCodedPos hcp;
    u8 result;
    size_t length;
    uint64_t hash;

    bool operator==(const Hcpe3SuffixKey& other) const {
        return result == other.result && length == other.length && hash == other.hash && hcp == other.hcp;
    }
};

struct Hcpe3SuffixKeyHash {
    std::size_t operator()(const Hcpe3SuffixKey& key) const {
        size_t h = std::hash<HuffmanCodedPos>()(key.hcp);
        h ^= std::hash<uint64_t>()(key.hash) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<size_t>()(key.length) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<unsigned int>()(key.result) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

struct Hcpe3SuffixLocation {
    size_t gameIndex;
    size_t offset;
};

struct Hcpe3MergeStats {
    size_t fileNum = 0;
    size_t corruptedFileNum = 0;
    size_t rawGameNum = 0;
    size_t zeroMoveSkippedNum = 0;
};

Hcpe3ReadResult read_hcpe3_exact(std::ifstream& ifs, char* dst, const std::streamsize size) {
    if (size == 0) {
        return Hcpe3ReadResult::Ok;
    }

    ifs.read(dst, size);
    if (ifs) {
        return Hcpe3ReadResult::Ok;
    }
    if (ifs.gcount() == 0 && ifs.eof()) {
        return Hcpe3ReadResult::Eof;
    }
    return Hcpe3ReadResult::Truncated;
}

uint64_t hcpe3_hash_move(const uint64_t hash, const u16 move16) {
    uint64_t h = hash ^ move16;
    h *= 1099511628211ULL;
    h ^= h >> 32;
    return h;
}

std::vector<uint64_t> hcpe3_suffix_hashes(const Hcpe3ParsedGame& game) {
    std::vector<uint64_t> hashes(game.moves.size() + 1);
    hashes[game.moves.size()] = 1469598103934665603ULL ^ game.result;
    for (size_t i = game.moves.size(); i > 0; --i) {
        hashes[i - 1] = hcpe3_hash_move(hashes[i], game.moves[i - 1].moveInfo.selectedMove16);
    }
    return hashes;
}

std::vector<uint64_t> hcpe3_suffix_hashes(const Hcpe3MergedGame& game) {
    std::vector<uint64_t> hashes(game.moves.size() + 1);
    hashes[game.moves.size()] = 1469598103934665603ULL ^ game.result;
    for (size_t i = game.moves.size(); i > 0; --i) {
        hashes[i - 1] = hcpe3_hash_move(hashes[i], game.moves[i - 1].selectedMove16);
    }
    return hashes;
}

Hcpe3SuffixKey hcpe3_make_suffix_key(const Hcpe3ParsedGame& game, const std::vector<uint64_t>& hashes, const size_t offset) {
    return Hcpe3SuffixKey{ game.positions[offset], game.result, game.moves.size() - offset, hashes[offset] };
}

Hcpe3SuffixKey hcpe3_make_suffix_key(const Hcpe3MergedGame& game, const std::vector<uint64_t>& hashes, const size_t offset) {
    return Hcpe3SuffixKey{ game.positions[offset], game.result, game.moves.size() - offset, hashes[offset] };
}

bool hcpe3_suffix_equals(const Hcpe3ParsedGame& parsed, const size_t parsedOffset, const Hcpe3MergedGame& merged, const size_t mergedOffset) {
    const size_t length = parsed.moves.size() - parsedOffset;
    if (!merged.active || parsed.result != merged.result || length != merged.moves.size() - mergedOffset) {
        return false;
    }
    if (!(parsed.positions[parsedOffset] == merged.positions[mergedOffset])) {
        return false;
    }
    for (size_t i = 0; i < length; ++i) {
        if (parsed.moves[parsedOffset + i].moveInfo.selectedMove16 != merged.moves[mergedOffset + i].selectedMove16) {
            return false;
        }
    }
    return true;
}

bool hcpe3_suffix_equals(const Hcpe3MergedGame& lhs, const size_t lhsOffset, const Hcpe3MergedGame& rhs, const size_t rhsOffset) {
    const size_t length = lhs.moves.size() - lhsOffset;
    if (!lhs.active || !rhs.active || lhs.result != rhs.result || length != rhs.moves.size() - rhsOffset) {
        return false;
    }
    if (!(lhs.positions[lhsOffset] == rhs.positions[rhsOffset])) {
        return false;
    }
    for (size_t i = 0; i < length; ++i) {
        if (lhs.moves[lhsOffset + i].selectedMove16 != rhs.moves[rhsOffset + i].selectedMove16) {
            return false;
        }
    }
    return true;
}

void hcpe3_merge_visits(Hcpe3MergedMove& dst, const std::vector<MoveVisits>& visits) {
    for (const auto& visit : visits) {
        dst.visitSums[visit.move16] += visit.visitNum;
    }
}

Hcpe3MergedGame hcpe3_make_merged_game(const Hcpe3ParsedGame& parsed) {
    Hcpe3MergedGame merged;
    merged.start = parsed.start;
    merged.result = parsed.result;
    merged.opponent = parsed.opponent;
    merged.active = true;
    merged.positions = parsed.positions;
    merged.moves.reserve(parsed.moves.size());

    for (const auto& move : parsed.moves) {
        Hcpe3MergedMove mergedMove;
        mergedMove.selectedMove16 = move.moveInfo.selectedMove16;
        mergedMove.evalSum = move.moveInfo.eval;
        mergedMove.count = 1;
        hcpe3_merge_visits(mergedMove, move.visits);
        merged.moves.emplace_back(std::move(mergedMove));
    }
    return merged;
}

void hcpe3_merge_parsed_into_merged(Hcpe3MergedGame& dst, const size_t dstOffset, const Hcpe3ParsedGame& src) {
    for (size_t i = 0; i < src.moves.size(); ++i) {
        auto& dstMove = dst.moves[dstOffset + i];
        const auto& srcMove = src.moves[i];
        dstMove.evalSum += srcMove.moveInfo.eval;
        dstMove.count++;
        hcpe3_merge_visits(dstMove, srcMove.visits);
    }
}

void hcpe3_merge_merged_into_merged(Hcpe3MergedGame& dst, const size_t dstOffset, const Hcpe3MergedGame& src) {
    for (size_t i = 0; i < src.moves.size(); ++i) {
        auto& dstMove = dst.moves[dstOffset + i];
        const auto& srcMove = src.moves[i];
        dstMove.evalSum += srcMove.evalSum;
        dstMove.count += srcMove.count;
        for (const auto& kv : srcMove.visitSums) {
            dstMove.visitSums[kv.first] += kv.second;
        }
    }
}

void hcpe3_add_suffixes(
    std::unordered_multimap<Hcpe3SuffixKey, Hcpe3SuffixLocation, Hcpe3SuffixKeyHash>& suffixIndex,
    const std::vector<Hcpe3MergedGame>& games,
    const size_t gameIndex) {
    const auto& game = games[gameIndex];
    const auto hashes = hcpe3_suffix_hashes(game);
    for (size_t offset = 0; offset < game.moves.size(); ++offset) {
        suffixIndex.emplace(hcpe3_make_suffix_key(game, hashes, offset), Hcpe3SuffixLocation{ gameIndex, offset });
    }
}

bool hcpe3_find_existing_suffix(
    const Hcpe3ParsedGame& parsed,
    const std::vector<uint64_t>& parsedHashes,
    const std::vector<Hcpe3MergedGame>& games,
    const std::unordered_multimap<Hcpe3SuffixKey, Hcpe3SuffixLocation, Hcpe3SuffixKeyHash>& suffixIndex,
    Hcpe3SuffixLocation& location) {
    if (parsed.moves.empty()) {
        return false;
    }

    const auto key = hcpe3_make_suffix_key(parsed, parsedHashes, 0);
    const auto range = suffixIndex.equal_range(key);
    for (auto itr = range.first; itr != range.second; ++itr) {
        const auto& candidate = itr->second;
        if (candidate.gameIndex < games.size() && hcpe3_suffix_equals(parsed, 0, games[candidate.gameIndex], candidate.offset)) {
            location = candidate;
            return true;
        }
    }
    return false;
}

std::vector<std::pair<size_t, size_t>> hcpe3_find_shorter_existing_games(
    const Hcpe3ParsedGame& parsed,
    const std::vector<uint64_t>& parsedHashes,
    const std::vector<Hcpe3MergedGame>& games,
    const std::unordered_multimap<Hcpe3SuffixKey, Hcpe3SuffixLocation, Hcpe3SuffixKeyHash>& suffixIndex) {
    std::vector<std::pair<size_t, size_t>> matches;
    std::unordered_map<size_t, bool> matchedGames;

    for (size_t offset = 1; offset < parsed.moves.size(); ++offset) {
        const auto key = hcpe3_make_suffix_key(parsed, parsedHashes, offset);
        const auto range = suffixIndex.equal_range(key);
        for (auto itr = range.first; itr != range.second; ++itr) {
            const auto& candidate = itr->second;
            if (candidate.offset != 0 || candidate.gameIndex >= games.size() || matchedGames[candidate.gameIndex]) {
                continue;
            }
            if (hcpe3_suffix_equals(parsed, offset, games[candidate.gameIndex], 0)) {
                matches.emplace_back(candidate.gameIndex, offset);
                matchedGames[candidate.gameIndex] = true;
            }
        }
    }
    return matches;
}

void hcpe3_add_game(
    const Hcpe3ParsedGame& parsed,
    std::vector<Hcpe3MergedGame>& games,
    std::unordered_multimap<Hcpe3SuffixKey, Hcpe3SuffixLocation, Hcpe3SuffixKeyHash>& suffixIndex) {
    if (parsed.moves.empty()) {
        return;
    }

    const auto parsedHashes = hcpe3_suffix_hashes(parsed);
    Hcpe3SuffixLocation location;
    if (hcpe3_find_existing_suffix(parsed, parsedHashes, games, suffixIndex, location)) {
        hcpe3_merge_parsed_into_merged(games[location.gameIndex], location.offset, parsed);
        return;
    }

    auto newGame = hcpe3_make_merged_game(parsed);
    const auto shorterMatches = hcpe3_find_shorter_existing_games(parsed, parsedHashes, games, suffixIndex);
    for (const auto& match : shorterMatches) {
        const size_t gameIndex = match.first;
        const size_t offset = match.second;
        if (!games[gameIndex].active || !hcpe3_suffix_equals(parsed, offset, games[gameIndex], 0)) {
            continue;
        }
        hcpe3_merge_merged_into_merged(newGame, offset, games[gameIndex]);
        games[gameIndex].active = false;
    }

    const size_t newIndex = games.size();
    games.emplace_back(std::move(newGame));
    hcpe3_add_suffixes(suffixIndex, games, newIndex);
}

bool hcpe3_read_game(std::ifstream& ifs, const std::string& filepath, const size_t gameIndex, Hcpe3ParsedGame& game, bool& cleanEof) {
    cleanEof = false;

    HuffmanCodedPosAndEval3 hcpe3;
    const auto headerResult = read_hcpe3_exact(ifs, reinterpret_cast<char*>(&hcpe3), sizeof(HuffmanCodedPosAndEval3));
    if (headerResult == Hcpe3ReadResult::Eof) {
        cleanEof = true;
        return false;
    }
    if (headerResult == Hcpe3ReadResult::Truncated || hcpe3.moveNum > HCPE3_MAX_MOVE_NUM) {
        return false;
    }

    Position pos;
    if (!pos.set(hcpe3.hcp)) {
        std::stringstream ss;
        ss << "INCORRECT_HUFFMAN_CODE at " << filepath << "(" << gameIndex << ")";
        throw std::runtime_error(ss.str());
    }
    StateListPtr states{ new std::deque<StateInfo>(1) };

    game.start = hcpe3.hcp;
    game.result = hcpe3.result;
    game.opponent = hcpe3.opponent;
    game.positions.clear();
    game.moves.clear();
    game.positions.reserve(hcpe3.moveNum);
    game.moves.reserve(hcpe3.moveNum);

    for (int i = 0; i < hcpe3.moveNum; ++i) {
        Hcpe3ParsedMove parsedMove;
        const auto moveInfoResult = read_hcpe3_exact(ifs, reinterpret_cast<char*>(&parsedMove.moveInfo), sizeof(MoveInfo));
        if (moveInfoResult != Hcpe3ReadResult::Ok || parsedMove.moveInfo.candidateNum > HCPE3_MAX_CANDIDATE_NUM) {
            return false;
        }

        parsedMove.visits.resize(parsedMove.moveInfo.candidateNum);
        const auto visitsResult = read_hcpe3_exact(
            ifs,
            reinterpret_cast<char*>(parsedMove.visits.data()),
            static_cast<std::streamsize>(sizeof(MoveVisits) * parsedMove.visits.size()));
        if (visitsResult != Hcpe3ReadResult::Ok) {
            return false;
        }

        const Move move = move16toMove((Move)parsedMove.moveInfo.selectedMove16, pos);
        if (!pos.moveIsPseudoLegal<false>(move)) {
            return false;
        }

        game.positions.emplace_back(pos.toHuffmanCodedPos());
        game.moves.emplace_back(std::move(parsedMove));
        pos.doMove(move, states->emplace_back(StateInfo()));
    }

    return true;
}

s16 hcpe3_average_eval(const Hcpe3MergedMove& move) {
    const auto average = static_cast<int64_t>(std::llround(static_cast<double>(move.evalSum) / static_cast<double>(move.count)));
    return static_cast<s16>(std::clamp<int64_t>(average, std::numeric_limits<s16>::min(), std::numeric_limits<s16>::max()));
}

u16 hcpe3_average_visit(const uint64_t sum, const uint64_t count) {
    uint64_t average = (sum + count / 2) / count;
    if (average == 0 && sum > 0) {
        average = 1;
    }
    return static_cast<u16>(std::min<uint64_t>(average, std::numeric_limits<u16>::max()));
}

std::vector<MoveVisits> hcpe3_average_visits(const Hcpe3MergedMove& move) {
    std::vector<MoveVisits> visits;
    visits.reserve(move.visitSums.size());
    for (const auto& kv : move.visitSums) {
        visits.emplace_back(kv.first, hcpe3_average_visit(kv.second, move.count));
    }
    std::sort(visits.begin(), visits.end(), [](const MoveVisits& a, const MoveVisits& b) {
        if (a.visitNum != b.visitNum) {
            return a.visitNum > b.visitNum;
        }
        return a.move16 < b.move16;
    });
    if (visits.size() > HCPE3_MAX_CANDIDATE_NUM) {
        std::stringstream ss;
        ss << "too many candidates after merge: " << visits.size();
        throw std::runtime_error(ss.str());
    }
    return visits;
}

void hcpe3_print_merge_stats(const Hcpe3MergeStats& stats, const std::vector<Hcpe3MergedGame>& games, const size_t mergedGameNum) {
    size_t blackWinNum = 0;
    size_t whiteWinNum = 0;
    size_t drawNum = 0;
    size_t sennichiteNum = 0;
    size_t nyugyokuNum = 0;
    size_t maxMoveNum = 0;
    size_t minMoves = std::numeric_limits<size_t>::max();
    size_t maxMoves = 0;
    uint64_t totalMoves = 0;

    for (const auto& game : games) {
        if (!game.active) {
            continue;
        }
        const size_t moveCount = game.moves.size();
        minMoves = std::min(minMoves, moveCount);
        maxMoves = std::max(maxMoves, moveCount);
        totalMoves += moveCount;

        const auto result = static_cast<GameResult>(game.result & 0x3);
        if (result == BlackWin) {
            blackWinNum++;
        }
        else if (result == WhiteWin) {
            whiteWinNum++;
        }
        else if (result == Draw) {
            drawNum++;
        }

        if (game.result & GAMERESULT_SENNICHITE) {
            sennichiteNum++;
        }
        if (game.result & GAMERESULT_NYUGYOKU) {
            nyugyokuNum++;
        }
        if (game.result & GAMERESULT_MAXMOVE) {
            maxMoveNum++;
        }
    }

    if (mergedGameNum == 0) {
        minMoves = 0;
    }
    const double averageMoves = mergedGameNum == 0 ? 0.0 : static_cast<double>(totalMoves) / static_cast<double>(mergedGameNum);

    std::cout << "Statistics" << std::endl;
    std::cout << "Files: " << stats.fileNum << std::endl;
    std::cout << "Files with truncated tail: " << stats.corruptedFileNum << std::endl;
    std::cout << "Raw games before averaging: " << stats.rawGameNum << std::endl;
    std::cout << "Skipped zero-move games: " << stats.zeroMoveSkippedNum << std::endl;
    std::cout << "Games after averaging: " << mergedGameNum << std::endl;
    std::cout << "Minimum moves: " << minMoves << std::endl;
    std::cout << "Maximum moves: " << maxMoves << std::endl;
    std::cout << "Average moves: " << std::fixed << std::setprecision(2) << averageMoves << std::endl;
    std::cout << "Black wins: " << blackWinNum << std::endl;
    std::cout << "White wins: " << whiteWinNum << std::endl;
    std::cout << "Draws: " << drawNum << std::endl;
    std::cout << "Games ended by sennichite: " << sennichiteNum << std::endl;
    std::cout << "Games ended by nyugyoku declaration: " << nyugyokuNum << std::endl;
    std::cout << "Games ended by max moves: " << maxMoveNum << std::endl;
}

} // namespace

void __hcpe3_merge(const std::vector<std::string>& files, const std::string& out) {
    Hcpe3MergeStats stats;
    stats.fileNum = files.size();

    std::vector<Hcpe3MergedGame> games;
    std::unordered_multimap<Hcpe3SuffixKey, Hcpe3SuffixLocation, Hcpe3SuffixKeyHash> suffixIndex;

    for (const auto& filepath : files) {
        std::cout << filepath << std::endl;

        std::ifstream ifs(filepath, std::ifstream::binary);
        if (!ifs) {
            std::stringstream ss;
            ss << "failed to open " << filepath;
            throw std::runtime_error(ss.str());
        }

        bool corrupted = false;
        for (size_t gameIndex = 0;; ++gameIndex) {
            Hcpe3ParsedGame parsed;
            bool cleanEof = false;
            if (!hcpe3_read_game(ifs, filepath, gameIndex, parsed, cleanEof)) {
                if (!cleanEof) {
                    corrupted = true;
                }
                break;
            }
            stats.rawGameNum++;
            if (parsed.moves.empty()) {
                stats.zeroMoveSkippedNum++;
                continue;
            }
            hcpe3_add_game(parsed, games, suffixIndex);
        }

        if (corrupted) {
            stats.corruptedFileNum++;
        }
    }

    size_t mergedGameNum = 0;
    {
        std::ofstream ofs(out, std::ios::binary);
        if (!ofs) {
            std::stringstream ss;
            ss << "failed to open " << out;
            throw std::runtime_error(ss.str());
        }

        for (const auto& game : games) {
            if (!game.active) {
                continue;
            }

            HuffmanCodedPosAndEval3 hcpe3{};
            hcpe3.hcp = game.start;
            hcpe3.moveNum = static_cast<u16>(game.moves.size());
            hcpe3.result = game.result;
            hcpe3.opponent = game.opponent;
            ofs.write(reinterpret_cast<const char*>(&hcpe3), sizeof(HuffmanCodedPosAndEval3));

            for (const auto& move : game.moves) {
                const auto visits = hcpe3_average_visits(move);
                MoveInfo moveInfo{};
                moveInfo.selectedMove16 = move.selectedMove16;
                moveInfo.eval = hcpe3_average_eval(move);
                moveInfo.candidateNum = static_cast<u16>(visits.size());
                ofs.write(reinterpret_cast<const char*>(&moveInfo), sizeof(MoveInfo));
                if (!visits.empty()) {
                    ofs.write(reinterpret_cast<const char*>(visits.data()), sizeof(MoveVisits) * visits.size());
                }
            }

            mergedGameNum++;
        }
    }

    hcpe3_print_merge_stats(stats, games, mergedGameNum);
}

std::pair<int, int> __hcpe3_clean(const std::string& file1, const std::string& file2) {
    std::ifstream ifs(file1, std::ifstream::binary);
    std::ofstream ofs(file2, std::ifstream::binary);

    int positions = 0;
    int p = 0;
    for (; ifs; ++p) {
        HuffmanCodedPosAndEval3 hcpe3;
        ifs.read((char*)&hcpe3, sizeof(HuffmanCodedPosAndEval3));
        if (ifs.eof()) {
            break;
        }
        assert(hcpe3.moveNum <= 513);

        // 開始局面
        Position pos;
        if (!pos.set(hcpe3.hcp)) {
            std::stringstream ss;
            ss << "INCORRECT_HUFFMAN_CODE at " << file1 << "(" << p << ")";
            throw std::runtime_error(ss.str());
        }
        StateListPtr states{ new std::deque<StateInfo>(1) };

        std::vector<std::pair<MoveInfo, std::vector<MoveVisits>>> moveInfos;
        moveInfos.reserve(hcpe3.moveNum);
        for (int i = 0; i < hcpe3.moveNum; ++i) {
            auto& moveInfo = moveInfos.emplace_back();
            ifs.read((char*)&moveInfo.first, sizeof(MoveInfo));
            if (ifs.eof()) {
                std::cout << "read error" << std::endl;
                return std::make_pair(p, positions);
            }
            assert(moveInfo.first.candidateNum <= 593);

            const Move move = move16toMove((Move)moveInfo.first.selectedMove16, pos);
            if (!pos.moveIsPseudoLegal<false>(move)) {
                std::stringstream ss;
                ss << "illegal move at " << file1 << "(" << p << ")";
                throw std::runtime_error(ss.str());
            }

            if (moveInfo.first.candidateNum > 0) {
                moveInfo.second.resize(moveInfo.first.candidateNum);
                ifs.read((char*)moveInfo.second.data(), sizeof(MoveVisits) * moveInfo.first.candidateNum);
                if (ifs.eof()) {
                    std::cout << "read error" << std::endl;
                    return std::make_pair(p, positions);
                }
                positions++;
            }

            pos.doMove(move, states->emplace_back(StateInfo()));
        }
        ofs.write((char*)&hcpe3, sizeof(HuffmanCodedPosAndEval3));
        for (const auto& moveInfo : moveInfos) {
            ofs.write((char*)&moveInfo.first, sizeof(MoveInfo));
            if (moveInfo.first.candidateNum > 0)
                ofs.write((char*)moveInfo.second.data(), sizeof(MoveVisits) * moveInfo.first.candidateNum);
        }
    }
    return std::make_pair(p, positions);
}

unsigned int __get_max_features2_nyugyoku_num() {
#ifdef NYUGYOKU_FEATURES
    return MAX_FEATURES2_NYUGYOKU_NUM;
#else
    return 0;
#endif
}
