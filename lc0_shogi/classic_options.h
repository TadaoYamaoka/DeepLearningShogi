// USI exposure of a bounded subset of upstream classic options.
// GPL-3.0-or-later.
#pragma once
#include "usi.hpp"
#include "utils/optionsparser.h"
#include <memory>
#include <utility>
#include <vector>

namespace lczero::shogi {
class ClassicOptions {
public:
  ClassicOptions();
  void Register(OptionsMap& options);
  // Main thread only, after search/response/PV workers have joined at isready.
  void Snapshot(const OptionsMap& options);
  void ApplyTo(OptionsParser* parser) const;
  float PolicyTemperature() const { return policy_temperature_; }
private:
  std::shared_ptr<OptionsParser> validator_;
  std::vector<std::pair<std::string, std::string>> snapshot_;
  float policy_temperature_ = 1.74f;
};
} // namespace lczero::shogi
