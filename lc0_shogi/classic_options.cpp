#include "classic_options.h"
#include "neural/shared_params.h"
#include "search/classic/params.h"
#include <array>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace lczero::shogi {
namespace {
using P = classic::BaseSearchParams;
struct ExposedOption {
  const OptionId* id;
  const char* initial;
  // Integer USI presentation. Limits mirror the pinned upstream FloatOption
  // bounds because its public API does not expose numeric limits. Upstream
  // still validates every decoded value; defaults/choices/names come from it.
  int scale, minimum, maximum;
};
const std::array<ExposedOption, 12> exposed{{
    {&P::kCpuctId, "2.88", 1000, 0, 100000},
    {&P::kCpuctBaseId, "28288", 1, 1, 1000000000},
    {&P::kCpuctFactorId, "2", 1000, 0, 1000000},
    {&P::kCpuctAtRootId, "2.32", 1000, 0, 100000},
    {&P::kCpuctBaseAtRootId, "25617", 1, 1, 1000000000},
    {&P::kCpuctFactorAtRootId, "2", 1000, 0, 1000000},
    {&P::kRootHasOwnCpuctParamsId, "true", 0, 0, 0},
    {&P::kFpuStrategyId, "reduction", 0, 0, 0},
    {&P::kFpuValueId, "0.54", 1000, -100000, 100000},
    {&P::kFpuStrategyAtRootId, "reduction", 0, 0, 0},
    {&P::kFpuValueAtRootId, "0", 1000, -100000, 100000},
    {&SharedBackendParams::kPolicySoftmaxTemp, "1.74", 1000, 100, 10000},
}};
std::string Decode(const std::string& value, int scale) {
  size_t used = 0;
  const auto integer = std::stoll(value, &used);
  if (used != value.size()) throw std::invalid_argument("Expected integer USI value");
  return std::to_string(static_cast<double>(integer) / scale);
}
} // namespace

ClassicOptions::ClassicOptions() : validator_(std::make_shared<OptionsParser>()) {
  SharedBackendParams::Populate(validator_.get());
  classic::SearchParams::Populate(validator_.get());
  // Root options are pro-only upstream; expose only the explicit allowlist.
  validator_->ProcessFlags({"--show-hidden"});
  for (const auto& option : exposed)
    validator_->SetUciOption(option.id->uci_option(), option.initial);
}

void ClassicOptions::Register(OptionsMap& options) {
  const auto declarations = validator_->ListOptionsUci();
  for (const auto& option : exposed) {
    const std::string name(option.id->uci_option());
    const std::string prefix = "option name " + name + ' ';
    bool found = false;
    for (const auto& line : declarations) {
      if (!line.starts_with(prefix)) continue;
      auto declaration = line.substr(prefix.size());
      const auto offset = declaration.find(" default ");
      if (offset == std::string::npos)
        throw std::logic_error("Missing classic default: " + name);
      std::string initial;
      std::istringstream(declaration.substr(offset + 9)) >> initial;
      if (option.scale) {
        initial = std::to_string(std::llround(std::stod(initial) * option.scale));
        declaration = "type spin default " + initial + " min " +
                      std::to_string(option.minimum) + " max " + std::to_string(option.maximum);
      }
      auto validator = validator_;
      options[name] = USIOption(declaration, initial,
          [validator, name, spec = option](const std::string& value) {
            const auto reject = [&](const std::string& reason) {
              std::cout << "info string Rejected option " << name << ": " << reason;
              if (spec.scale == 1000)
                std::cout << "; specify an integer in 1/1000 units";
              else if (spec.scale)
                std::cout << "; specify an integer";
              std::cout << std::endl;
              return false;
            };
            try {
              if (spec.scale) {
                const auto decoded = Decode(value, spec.scale);
                const auto integer = std::stoll(value);
                if (integer < spec.minimum || integer > spec.maximum)
                  return reject("outside [" + std::to_string(spec.minimum) + ", " +
                                std::to_string(spec.maximum) + "]");
                validator->SetUciOption(name, decoded);
              } else {
                validator->SetUciOption(name, value);
              }
              return true;
            } catch (const std::exception&) {
              return reject(spec.scale ? "invalid integer value" : "invalid value or choice");
            }
          });
      found = true;
      break;
    }
    if (!found) throw std::logic_error("Missing classic declaration: " + name);
  }
  Snapshot(options);
}

void ClassicOptions::Snapshot(const OptionsMap& options) {
  std::vector<std::pair<std::string, std::string>> next;
  for (const auto& option : exposed) {
    const std::string name(option.id->uci_option());
    const std::string value(options.at(name));
    next.emplace_back(name, option.scale ? Decode(value, option.scale) : value);
  }
  snapshot_ = std::move(next);
  policy_temperature_ = validator_->GetOptionsDict().Get<float>(
      SharedBackendParams::kPolicySoftmaxTemp);
}

void ClassicOptions::ApplyTo(OptionsParser* parser) const {
  for (const auto& [name, value] : snapshot_)
    parser->SetUciOption(name, value);
  // The exposed PolicyTemperature is applied once to dlshogi logits by its
  // backend. classic must not apply it again to the returned probabilities.
  parser->GetMutableOptions()->Set<float>(SharedBackendParams::kPolicySoftmaxTemp, 1);
}
} // namespace lczero::shogi
