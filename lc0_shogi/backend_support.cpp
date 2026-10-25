// Backend interface helpers without lc0 model loaders or backend registries.
// GPL-3.0-or-later.
#include "neural/backend.h"
#include "neural/register.h"
namespace lczero {
// shared_params.cc uses only these registry queries while declaring options.
// No lc0 inference factory, model loader or cache wrapper is registered/linked.
BackendManager* BackendManager::Get() {
  static BackendManager manager;
  return &manager;
}
std::vector<std::string> BackendManager::GetBackendNames() const {
  return {"dlshogi"};
}
uint64_t Backend::ConfigurationHash(const OptionsDict &) const { return 0; }
std::vector<EvalResult>
Backend::EvaluateBatch(std::span<const EvalPosition> positions) {
  std::vector<EvalResult> results(positions.size());
  auto computation = CreateComputation();
  for (size_t i = 0; i < positions.size(); ++i) {
    results[i].p.resize(positions[i].legal_moves.size());
    computation->AddInput(positions[i], results[i].AsPtr());
  }
  computation->ComputeBlocking();
  return results;
}
} // namespace lczero
