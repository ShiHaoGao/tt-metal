// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-only producer sampling against scripted register reads. No SDK context
// or device is used; the production counter helper is exercised directly.
#include "tools/profiler/profiler_counter.hpp"

#include <cstdio>
#include <exception>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

enum class Register { LiveHigh, Low };

struct Read {
    Register reg;
    uint32_t value;
};

class Counter {
public:
    explicit Counter(std::initializer_list<Read> reads) : reads_(reads) {}

    uint32_t read(Register reg) {
        check(position_ < reads_.size(), "read past the counter script");
        const auto& next = reads_[position_++];
        check(next.reg == reg, "counter register reads are out of order");
        // Model another RISC changing the shared high latch after every low
        // read. The sampler's two callbacks expose only live high and low.
        if (reg == Register::Low) {
            latched_high = 0xbad;
        }
        return next.value;
    }

    kernel_profiler::WallClockSample sample() {
        return kernel_profiler::read_coherent_wall_clock(
            [&] { return read(Register::LiveHigh); }, [&] { return read(Register::Low); });
    }

    void check_complete() const {
        check(position_ == reads_.size(), "sample returned before validating the final live high");
    }

    uint32_t latched_high = 0;

private:
    std::vector<Read> reads_;
    std::size_t position_ = 0;
};

constexpr auto high = Register::LiveHigh;
constexpr auto low = Register::Low;
unsigned tests = 0;
unsigned failures = 0;

template <class F>
void test(const char* name, F body) {
    ++tests;
    try {
        body();
    } catch (const std::exception& error) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s: %s\n", name, error.what());
    }
}

}  // namespace

int main() {
    test("stable sample preserves all 64 counter bits", [] {
        Counter counter{{high, 0xfedcba98}, {low, 0x76543210}, {high, 0xfedcba98}};
        auto sample = counter.sample();
        check(sample.high == 0xfedcba98 && sample.low == 0x76543210,
              "sampling must retain both complete words before wire encoding");
        check(sample.value() == 0xfedcba9876543210ULL, "full-width composition must not truncate high bits");
    });
    test("stable sample validates the final live high", [] {
        Counter counter{{high, 0x123}, {low, 7}, {high, 0x123}};
        check(counter.sample().value() == 0x12300000007ULL, "stable counter value");
        counter.check_complete();
    });
    test("low-32 rollover rejects old high and new low", [] {
        Counter counter{{high, 0x123}, {low, 5}, {high, 0x124},
                        {high, 0x124}, {low, 9}, {high, 0x124}};
        check(counter.sample().value() == 0x12400000009ULL,
              "rollover must retry instead of publishing a timestamp smaller by 2^32");
        counter.check_complete();
    });
    test("rollover after low also retries the whole sample", [] {
        Counter counter{{high, 0x123}, {low, 0xfffffffe}, {high, 0x124},
                        {high, 0x124}, {low, 11}, {high, 0x124}};
        check(counter.sample().value() == 0x1240000000bULL,
              "a changed final high must discard both words from the previous attempt");
        counter.check_complete();
    });
    test("repeated rollover retains a real retry path", [] {
        Counter counter{{high, 0x123}, {low, 5}, {high, 0x124},
                        {high, 0x124}, {low, 8}, {high, 0x125},
                        {high, 0x125}, {low, 12}, {high, 0x125}};
        check(counter.sample().value() == 0x1250000000cULL,
              "the returned sample must come from the final coherent attempt");
        counter.check_complete();
    });
    test("another reader changing the latch does not affect live sampling", [] {
        Counter counter{{high, 0x456}, {low, 42}, {high, 0x456}};
        check(counter.sample().value() == 0x4560000002aULL, "shared latched high must not enter the sample");
        check(counter.latched_high == 0xbad, "fixture must simulate the competing latch write");
        counter.check_complete();
    });
    test("44-bit wire rollover does not truncate the sampled epoch", [] {
        Counter counter{{high, 0xfff}, {low, 3}, {high, 0x1000},
                        {high, 0x1000}, {low, 7}, {high, 0x1000}};
        const auto sample = counter.sample();
        check(sample.value() == 0x100000000007ULL,
              "sampler must keep the full counter even when the later wire encoding wraps");
        check((sample.value() & ((1ULL << 44) - 1)) == 7, "wire masking happens only after coherent acquisition");
        counter.check_complete();
    });
    test("compare full high halves before masking to the wire width", [] {
        Counter counter{{high, 0x123}, {low, 5}, {high, 0x1123},
                        {high, 0x1123}, {low, 17}, {high, 0x1123}};
        check(counter.sample().value() == 0x112300000011ULL,
              "equal low 12 high bits cannot hide a changed full high half");
        counter.check_complete();
    });
    test("failed validation read cannot publish an unchecked sample", [] {
        unsigned high_reads = 0;
        bool failed = false;
        try {
            (void)kernel_profiler::read_coherent_wall_clock(
                [&]() -> uint32_t {
                    if (++high_reads == 2) {
                        throw std::runtime_error("scripted validation read failure");
                    }
                    return 0x123;
                },
                [] { return 7u; });
        } catch (const std::runtime_error&) {
            failed = true;
        }
        check(failed && high_reads == 2, "sampler must not return before its final validation read");
    });

    std::printf("profiler counter: %u passed, %u failed\n", tests - failures, failures);
    return failures == 0 ? 0 : 1;
}
