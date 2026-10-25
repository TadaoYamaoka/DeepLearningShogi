#include "backend.h"
#include "request_pool.h"
#include "cppshogi.h"
#ifdef ONNXRUNTIME
#include "nn_onnxruntime.h"
#else
#include "nn_tensorrt.h"
#endif
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <stdexcept>
#if defined(MAKE_BOOK) || defined(BOOK_POLICY)
#include "book.hpp"
extern std::map<Key, std::vector<BookEntry>> bookMap;
extern bool use_book_policy;
#endif
namespace lczero::shogi {
#ifdef ONNXRUNTIME
using HostFeatures1 = std::array<DType, sizeof(features1_t) / sizeof(DType)>;
using HostFeatures2 = std::array<DType, sizeof(features2_t) / sizeof(DType)>;
#else
using HostFeatures1 = std::array<char, sizeof(packed_features1_t)>;
using HostFeatures2 = std::array<char, sizeof(packed_features2_t)>;
#endif
// Slot storage outlives each computation. TensorRT copies asynchronously, so
// retain page-locked storage until forward() has synchronized its CUDA stream.
template <typename T> class HostBuffer {
public:
  explicit HostBuffer(size_t count) {
#ifdef ONNXRUNTIME
    data_ = new T[count];
#else
    if (cudaHostAlloc(reinterpret_cast<void **>(&data_), count * sizeof(T),
                      cudaHostAllocPortable) != cudaSuccess)
      throw std::runtime_error("Cannot allocate pinned inference buffer");
#endif
  }
  ~HostBuffer() {
#ifdef ONNXRUNTIME
    delete[] data_;
#else
    cudaFreeHost(data_);
#endif
  }
  HostBuffer(const HostBuffer &) = delete;
  HostBuffer &operator=(const HostBuffer &) = delete;
  T *data() { return data_; }
  T &operator[](size_t index) { return data_[index]; }

private:
  T *data_ = nullptr;
};
struct InferenceBuffers {
  explicit InferenceBuffers(size_t batch)
      : x1(batch), x2(batch),
        policy(batch * MAX_MOVE_LABEL_NUM * static_cast<size_t>(SquareNum)),
        value(batch) {}
  HostBuffer<HostFeatures1> x1;
  HostBuffer<HostFeatures2> x2;
  HostBuffer<DType> policy, value;
};
struct DlshogiBackend::Device {
  DeviceConfig config;
  int gpu = 0;
  std::unique_ptr<NN> nn;
  std::vector<std::unique_ptr<InferenceBuffers>> buffers;
  std::mutex onnx_lock;
  ~Device() {
#ifndef ONNXRUNTIME
    cudaSetDevice(gpu);
#endif
    buffers.clear();
    nn.reset();
  }
};
struct DlshogiBackend::Request {
  HostFeatures1 f1{};
  HostFeatures2 f2{};
  std::vector<Move> canonical;
  Color color;
  Key key;
  Key book_key = 0;
  EvalResultPtr destination;
};
class DlshogiBackend::Computation : public BackendComputation {
public:
  explicit Computation(DlshogiBackend *owner) : owner_(owner) {}
  ~Computation() override {
    if (requests_)
      owner_->request_pool_->Return(std::move(requests_));
  }
  // Lease only for inference, never while classic is collecting a batch.
  struct SlotLease {
    explicit SlotLease(DlshogiBackend *backend) : owner(backend) {
      std::unique_lock<std::mutex> lock(owner->slots_mutex_);
      owner->slots_available_.wait(
          lock, [&] { return !owner->available_slots_.empty(); });
      index = owner->available_slots_.front();
      owner->available_slots_.pop_front();
    }
    ~SlotLease() {
      {
        std::lock_guard<std::mutex> lock(owner->slots_mutex_);
        owner->available_slots_.push_back(index);
      }
      owner->slots_available_.notify_one();
    }
    DlshogiBackend *owner;
    size_t index;
  };
  size_t UsedBatchSize() const override {
    const auto *batch = published_requests_.load(std::memory_order_acquire);
    return batch ? batch->size() : 0;
  }
  AddInputResult AddInput(const EvalPosition &input,
                          EvalResultPtr destination) override {
    const auto &position = input.pos.back().GetBoard().Shogi();
    // Contract: the complete legal list, in GenerateLegalMoves() order.
    // classic calls AddInput before FetchSingleNodeResult sorts the new edges;
    // prefetch and DumpNeuralEvaluation also pass freshly generated lists.
    // cppshogi's generator order depends on the board, not the path to it, so
    // cache entries use that same order. Do not pass sorted/subset move lists.
#ifndef NDEBUG
    const auto legal = input.pos.back().GetBoard().GenerateLegalMoves();
    assert(legal.size() == input.legal_moves.size());
    assert(std::equal(legal.begin(), legal.end(), input.legal_moves.begin()));
#endif
    assert(destination.p.empty() ||
           destination.p.size() == input.legal_moves.size());
    const auto key = position.getKey();
    float cached_value;
    if (owner_->cache_.Lookup(key, input.legal_moves.size(), cached_value,
                              [&](size_t i, float p) {
                                if (i < destination.p.size())
                                  destination.p[i] = p;
                              })) {
      if (destination.q)
        *destination.q = 2 * cached_value - 1;
      if (destination.d)
        *destination.d = 0;
      if (destination.m)
        *destination.m = 0;
      return FETCHED_IMMEDIATELY;
    }
    // A hit needs neither a policy temporary nor a feature-bearing Request.
    Request request;
    request.key = key;
    request.color = position.turn();
#if defined(MAKE_BOOK) || defined(BOOK_POLICY)
    if (use_book_policy)
      request.book_key = Book::bookKey(position);
#endif
    request.destination = destination;
    request.canonical.assign(input.legal_moves.begin(),
                             input.legal_moves.end());
#ifdef ONNXRUNTIME
    ::make_input_features(position,
                          *reinterpret_cast<features1_t *>(request.f1.data()),
                          *reinterpret_cast<features2_t *>(request.f2.data()));
#else
    ::make_input_features(position, request.f1.data(), request.f2.data());
#endif
    // classic task helpers may enter AddInput concurrently. Lazy borrowing
    // keeps cache-only computations allocation-free; call_once publishes one
    // reusable buffer to all helpers of this computation.
    std::call_once(requests_once_, [&] {
      requests_ = owner_->request_pool_->Borrow();
      published_requests_.store(requests_.get(), std::memory_order_release);
    });
    requests_->Append(std::move(request));
    return ENQUEUED_FOR_EVAL;
  }
  void ComputeBlocking() override {
    const size_t request_count = UsedBatchSize();
    if (request_count == 0)
      return;
    SlotLease lease(owner_);
    const auto [device_index, slot] = owner_->slots_[lease.index];
    auto &device = *owner_->devices_[device_index];
    auto &buffers = *device.buffers[slot];
    auto &x1 = buffers.x1;
    auto &x2 = buffers.x2;
    auto &policy = buffers.policy;
    auto &value = buffers.value;
#ifndef ONNXRUNTIME
    if (cudaSetDevice(device.gpu) != cudaSuccess)
      throw std::runtime_error("Cannot select inference GPU");
#endif
    // classic WaitForTasks() joins all AddInput writers before this call.
    for (size_t begin = 0; begin < request_count;
         begin += device.config.batch) {
      const int count = static_cast<int>(std::min(
          request_count - begin, static_cast<size_t>(device.config.batch)));
      for (int i = 0; i < count; ++i) {
        x1[i] = (*requests_)[begin + i].f1;
        x2[i] = (*requests_)[begin + i].f2;
      }
#ifdef ONNXRUNTIME
      {
        std::lock_guard<std::mutex> lock(device.onnx_lock);
        device.nn->forward(count, reinterpret_cast<features1_t *>(x1.data()),
                           reinterpret_cast<features2_t *>(x2.data()),
                           policy.data(), value.data());
      }
#else
      device.nn->forward(slot, count,
                         reinterpret_cast<packed_features1_t *>(x1.data()),
                         reinterpret_cast<packed_features2_t *>(x2.data()),
                         policy.data(), value.data());
#endif
      // Count actual completed inference, including prefetch but excluding
      // cache hits and terminal-node visits. Two relaxed updates per backend
      // batch.
      owner_->evaluations_.fetch_add(count, std::memory_order_relaxed);
      owner_->batches_.fetch_add(1, std::memory_order_relaxed);
      for (int i = 0; i < count; ++i) {
        auto &request = (*requests_)[begin + i];
        std::vector<float> probabilities(request.canonical.size());
        float maximum = -INFINITY;
        for (size_t j = 0; j < probabilities.size(); ++j) {
          const int label =
              ::make_move_label(request.canonical[j].raw_data(), request.color);
          probabilities[j] =
              static_cast<float>(
                  policy[static_cast<size_t>(i) * MAX_MOVE_LABEL_NUM *
                             static_cast<size_t>(SquareNum) +
                         label]) /
              owner_->temperature_;
          maximum = std::max(maximum, probabilities[j]);
        }
        float total = 0;
        for (auto &p : probabilities) {
          p = std::exp(p - maximum);
          total += p;
        }
        for (auto &p : probabilities)
          p /= total;
        float win = static_cast<float>(value[i]);
        if (!std::isfinite(win) || win < 0 || win > 1)
          throw std::runtime_error("Invalid dlshogi value output");
#if defined(MAKE_BOOK) || defined(BOOK_POLICY)
        if (use_book_policy) {
          const auto found = bookMap.find(request.book_key);
          if (found != bookMap.end() && !found->second.empty()) {
            std::map<uint16_t, uint16_t> counts;
            int total_count = 0;
            for (const auto &entry : found->second) {
              counts.emplace(entry.fromToPro, entry.count);
              total_count += entry.count;
            }
            if (total_count > 0)
              for (size_t j = 0; j < probabilities.size(); ++j) {
                const auto count = counts.find(request.canonical[j].raw_data());
                const float prior =
                    count == counts.end()
                        ? 0
                        : static_cast<float>(count->second) / total_count;
                probabilities[j] = .5f * (probabilities[j] + prior);
              }
            win = .5f * (win + ::score_to_value(found->second.front().score));
          }
        }
#endif
        owner_->cache_.Store(request.key, win, probabilities.size(),
                             [&](size_t j) { return probabilities[j]; });
        Write(request, win, probabilities);
      }
    }
  }

private:
  static void Write(const Request &request, float win,
                    const std::vector<float> &probabilities) {
    auto destination = request.destination;
    if (destination.q)
      *destination.q = 2 * win - 1;
    if (destination.d)
      *destination.d = 0;
    if (destination.m)
      *destination.m = 0;
    for (size_t i = 0; i < destination.p.size(); ++i)
      destination.p[i] = probabilities[i];
  }
  DlshogiBackend *owner_;
  using Batch = RequestPool<Request>::Batch;
  std::unique_ptr<Batch> requests_;
  std::once_flag requests_once_;
  std::atomic<Batch *> published_requests_{nullptr};
};
DlshogiBackend::DlshogiBackend(const std::array<DeviceConfig, 8> &configs,
                               size_t cache_size, float temperature,
                               int collection_batch)
    : temperature_(temperature) {
  if (temperature <= 0)
    throw std::invalid_argument("Invalid policy temperature");
  cache_.SetCapacity(cache_size);
  for (size_t gpu = 0; gpu < configs.size(); ++gpu) {
    const auto &config = configs[gpu];
    if (config.slots == 0)
      continue;
    if (!std::ifstream(config.model))
      throw std::runtime_error("Cannot open model for GPU " + std::to_string(gpu) +
                               ": " + config.model);
    auto device = std::make_unique<Device>();
    device->config = config;
    device->gpu = static_cast<int>(gpu);
#ifdef ONNXRUNTIME
    device->nn = std::make_unique<NNOnnxRuntime>(
        config.model.c_str(), static_cast<int>(gpu), config.batch);
#else
    if (cudaSetDevice(static_cast<int>(gpu)) != cudaSuccess)
      throw std::runtime_error("Cannot select inference GPU");
    device->nn = std::make_unique<NNTensorRT>(config.model.c_str(),
                                              static_cast<int>(gpu),
                                              config.batch, config.slots);
    device->nn->prepare_slots(config.slots);
#endif
    for (int slot = 0; slot < config.slots; ++slot) {
      device->buffers.push_back(
          std::make_unique<InferenceBuffers>(config.batch));
      available_slots_.push_back(slots_.size());
      slots_.emplace_back(static_cast<int>(devices_.size()), slot);
    }
    max_batch_ = std::max(max_batch_, config.batch);
    recommended_batch_ = recommended_batch_ == 0
                             ? config.batch
                             : std::min(recommended_batch_, config.batch);
    devices_.push_back(std::move(device));
  }
  if (slots_.empty())
    throw std::invalid_argument("At least one inference slot is required");
  SetCollectionBatchSize(collection_batch);
}
DlshogiBackend::~DlshogiBackend() = default;
void DlshogiBackend::SetCollectionBatchSize(int collection_batch) {
  // Configure() leaves classic MaxPrefetch at 32. Guarded at Append even in
  // Release; update this capacity together with any future prefetch setting.
  const int capacity = std::max(max_batch_, collection_batch) + 32;
  if (capacity == request_capacity_)
    return;
  auto pool = std::make_unique<RequestPool<Request>>(capacity);
  request_pool_ = std::move(pool);
  request_capacity_ = capacity;
}
void DlshogiBackend::ResetCache(size_t capacity) {
  cache_.SetCapacity(capacity);
}
DlshogiBackend::InferenceStats DlshogiBackend::GetInferenceStats() const {
  return {evaluations_.load(std::memory_order_relaxed),
          batches_.load(std::memory_order_relaxed)};
}
BackendAttributes DlshogiBackend::GetAttributes() const {
  return {false,      false,     false, static_cast<int>(slots_.size()),
          recommended_batch_, max_batch_};
}
std::unique_ptr<BackendComputation> DlshogiBackend::CreateComputation() {
  return std::make_unique<Computation>(this);
}
} // namespace lczero::shogi
