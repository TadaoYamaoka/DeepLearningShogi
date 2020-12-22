#pragma once

#include <cstdint>

/*  Written in 2015 by Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

/* This is a fixed-increment version of Java 8's SplittableRandom generator
   See http://dx.doi.org/10.1145/2714064.2660195 and
   http://docs.oracle.com/javase/8/docs/api/java/util/SplittableRandom.html

   It is a very fast generator passing BigCrush, and it can be useful if
   for some reason you absolutely want 64 bits of state. */

class SplitMix64
{
private:
	uint64_t x; /* The state can be seeded with any value. */

public:
	SplitMix64(uint64_t x) noexcept : x(x) {}

	uint64_t next() noexcept {
		uint64_t z = (x += 0x9e3779b97f4a7c15);
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
		z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
		return z ^ (z >> 31);
	}
};

/* This is xoshiro128++ 1.0, one of our 32-bit all-purpose, rock-solid
   generators. It has excellent speed, a state size (128 bits) that is
   large enough for mild parallelism, and it passes all tests we are aware
   of.

   For generating just single-precision (i.e., 32-bit) floating-point
   numbers, xoshiro128+ is even faster.

   The state must be seeded so that it is not everywhere zero. */

class Xoshiro128
{
private:
	static inline uint32_t rotl(const uint32_t x, int k) noexcept {
		return (x << k) | (x >> (32 - k));
	}

	uint32_t s[4];

public:
	Xoshiro128(uint64_t seed = 1234567890ULL) noexcept {
		SplitMix64 splitmix{ seed };
		s[0] = static_cast<std::uint32_t>(splitmix.next());
		s[1] = static_cast<std::uint32_t>(splitmix.next());
		s[2] = static_cast<std::uint32_t>(splitmix.next());
		s[3] = static_cast<std::uint32_t>(splitmix.next());
	}

	uint32_t next(void) noexcept {
		const uint32_t result = rotl(s[0] + s[3], 7) + s[0];

		const uint32_t t = s[1] << 9;

		s[2] ^= s[0];
		s[3] ^= s[1];
		s[1] ^= s[2];
		s[0] ^= s[3];

		s[2] ^= t;

		s[3] = rotl(s[3], 11);

		return result;
	}
};