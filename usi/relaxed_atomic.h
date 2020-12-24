#pragma once

#include <atomic>

template <typename T>
class RelaxedAtomic
{
public:
    RelaxedAtomic() noexcept = default;

    constexpr RelaxedAtomic(T desired) : value(desired) {}

    T load() const noexcept {
        return value.load(std::memory_order_relaxed);
    }

    void store(T desired) noexcept {
        value.store(desired, std::memory_order_relaxed);
    }

    T exchange(T desired) noexcept {
        return value.exchange(desired, std::memory_order_relaxed);
    }

    T fetch_add(T operand) noexcept {
        return value.fetch_add(operand, std::memory_order_relaxed);
    }

    T fetch_sub(T operand) noexcept {
        return value.fetch_sub(operand, std::memory_order_relaxed);
    }

    operator T() const noexcept {
        return load();
    }

    T operator=(T desired) noexcept {
        store(desired);
        return desired;
    }

    T operator+=(T operand) noexcept {
        return fetch_add(operand) + operand;
    }

    T operator-=(T operand) noexcept {
        return fetch_sub(operand) - operand;
    }

private:
    std::atomic<T> value;
};

template <>
class RelaxedAtomic<float>
{
public:
    RelaxedAtomic() noexcept = default;

    constexpr RelaxedAtomic(float desired) : value(desired) {}

    float load() const noexcept {
        return value.load(std::memory_order_relaxed);
    }

    void store(float desired) noexcept {
        value.store(desired, std::memory_order_relaxed);
    }

    float exchange(float desired) noexcept {
        return value.exchange(desired, std::memory_order_relaxed);
    }

    float fetch_add(float operand) noexcept {
        float expected = value.load(std::memory_order_relaxed);
        while (!value.compare_exchange_weak(expected, expected + operand))
            ;
        return expected;
    }

    float fetch_sub(float operand) noexcept {
        float expected = value.load(std::memory_order_relaxed);
        while (!value.compare_exchange_weak(expected, expected - operand))
            ;
        return expected;
    }

    operator float() const noexcept {
        return load();
    }

    float operator=(float desired) noexcept {
        store(desired);
        return desired;
    }

    float operator+=(float operand) noexcept {
        return fetch_add(operand) + operand;
    }

    float operator-=(float operand) noexcept {
        return fetch_sub(operand) - operand;
    }

private:
    std::atomic<float> value;
};
