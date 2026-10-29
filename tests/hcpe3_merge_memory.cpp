// Include the implementation to verify internal allocations and suffix entries.
// Build this test without separately compiling python_module.cpp.
#include "../cppshogi/python_module.cpp"

namespace {
using SuffixIndex = std::unordered_multimap<Hcpe3SuffixKey, Hcpe3SuffixLocation, Hcpe3SuffixKeyHash>;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

Hcpe3ParsedGame make_suffix(size_t length) {
    Hcpe3ParsedGame game{};
    game.result = Draw;
    for (size_t i = HCPE3_MAX_MOVE_NUM - length; i < HCPE3_MAX_MOVE_NUM; ++i) {
        HuffmanCodedPos hcp{};
        hcp.data[0] = static_cast<u8>(i);
        hcp.data[1] = static_cast<u8>(i >> 8);
        game.positions.emplace_back(hcp);
        Hcpe3ParsedMove move{};
        move.moveInfo.selectedMove16 = static_cast<u16>(i + 1);
        move.moveInfo.eval = static_cast<s16>(i + 1);
        move.moveInfo.candidateNum = 1;
        move.visits.emplace_back(move.moveInfo.selectedMove16, static_cast<u16>(length));
        game.moves.emplace_back(std::move(move));
    }
    game.start = game.positions.front();
    return game;
}

void check_absorbed_games() {
    std::vector<Hcpe3MergedGame> games;
    SuffixIndex index;
    for (size_t length = 1; length <= HCPE3_MAX_MOVE_NUM; ++length) {
        hcpe3_add_game(make_suffix(length), games, index);
        check(games.size() == length && games.back().active, "merged game order changed");
        check(index.size() == length, "absorbed suffix entries remain");
        for (size_t i = 0; i + 1 < games.size(); ++i) {
            check(!games[i].active && games[i].moves.capacity() == 0
                && games[i].positions.capacity() == 0, "absorbed game still owns allocations");
        }
        for (const auto& entry : index) {
            check(entry.second.gameIndex == length - 1, "suffix points to an inactive game");
        }
    }
    const auto& merged = games.back();
    for (size_t i = 0; i < merged.moves.size(); ++i) {
        const auto& move = merged.moves[i];
        const uint64_t count = i + 1;
        const uint64_t visits = count * (2 * HCPE3_MAX_MOVE_NUM - i) / 2;
        check(move.count == count && move.evalSum == static_cast<int64_t>(count * count)
            && move.visitSums.size() == 1 && move.visitSums.at(static_cast<u16>(i + 1)) == visits,
            "absorption changed merged evaluations or visits");
    }
    // A later short game must still find and update the surviving long game.
    hcpe3_add_game(make_suffix(1), games, index);
    check(games.size() == HCPE3_MAX_MOVE_NUM && index.size() == HCPE3_MAX_MOVE_NUM,
        "later suffix was appended instead of merged");
    check(games.back().moves.back().count == HCPE3_MAX_MOVE_NUM + 1,
        "later suffix did not update the surviving game");
}

void check_shared_suffix_keys() {
    auto first = make_suffix(3);
    auto second = first;
    second.positions[0].data[2] = 1;
    second.moves[0].moveInfo.selectedMove16 = 1000;
    second.start = second.positions[0];
    std::vector<Hcpe3MergedGame> games;
    SuffixIndex index;
    hcpe3_add_game(first, games, index);
    hcpe3_add_game(second, games, index);
    check(games.size() == 2 && index.size() == 6, "branch fixture merged unexpectedly");
    hcpe3_remove_suffixes(index, games, 0);
    check(index.size() == 3, "removal erased another game's shared suffix keys");
    const auto hashes = hcpe3_suffix_hashes(games[1]);
    for (size_t offset = 0; offset < 3; ++offset) {
        const auto range = index.equal_range(hcpe3_make_suffix_key(games[1], hashes, offset));
        check(range.first != range.second && range.first->second.gameIndex == 1,
            "surviving branch lost a suffix entry");
    }
}
}

int main() {
    try {
        check_absorbed_games();
        check_shared_suffix_keys();
        std::cout << "hcpe3 merge memory tests passed\n";
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
