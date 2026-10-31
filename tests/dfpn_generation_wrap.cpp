#include "position.hpp"
#include "dfpn.h"

#include <iostream>
#include <stdexcept>

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}

int main() {
    ns_dfpn::TranspositionTable tt;
    tt.Resize(1);
    tt.NewSearch();
    check(tt.generation == 1, "first generation must be 1");

    const Key key = 0x1234567800000000ULL;
    auto& old_entry = tt.LookUp(key, Hand(0), 10);
    old_entry.pn = 100000000;
    old_entry.dn = 0;
    old_entry.num_searched = UINT_MAX - 1;

    // A different old generation at the far end must also be invalidated.
    auto& last_entry = tt.tt[tt.num_clusters - 1].entries[255];
    last_entry.generation = 2;
    last_entry.pn = 0;
    last_entry.dn = 100000000;

    for (int i = 0; i < 65534; ++i) tt.NewSearch();
    check(tt.generation == UINT16_MAX, "unexpected generation before wrap");
    check(old_entry.generation == 1 && old_entry.dn == 0,
        "ordinary NewSearch must not clear the table");

    tt.NewSearch();
    check(tt.generation == 1, "wrapped generation must be 1");
    for (int64_t i = 0; i < tt.num_clusters; ++i)
        for (const auto& entry : tt.tt[i].entries)
            check(entry.generation == 0, "old generation survived wrap");

    const auto& fresh_entry = tt.LookUp(key, Hand(0), 10);
    check(fresh_entry.pn == 1 && fresh_entry.dn == 1 && fresh_entry.num_searched == 0,
        "lookup reused the old disproof after wrap");

    std::cout << "dfpn generation wrap: OK\n";
}
