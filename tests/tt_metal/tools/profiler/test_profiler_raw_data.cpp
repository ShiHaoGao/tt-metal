// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-only classic profiler wire decoding. No SDK context or device is used.
#include <tt-metalium/profiler_raw_data.hpp>
#include "hostdev/profiler_common.h"

#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace tt::tt_metal;

namespace {

using Words = std::vector<uint32_t>;
constexpr auto complete = ProfilerRawDecodeStatus::Complete;
constexpr auto malformed = ProfilerRawDecodeStatus::Malformed;
constexpr auto truncated = ProfilerRawDecodeStatus::Truncated;
constexpr auto classic = ProfilerRawProtocol::ClassicDram;

// Hand-authored classic wire: RISC 2, flat core 7, trace 3, runtime 99;
// firmware [100, 500], kernel [200, 400]. Guaranteed storage is not time ordered.
Words run() {
    return {0, 0, 0x60e2, 99, 0x80011000, 100, 0x90011000, 500,
            0x80022000, 200, 0x90022000, 400};
}

ProfilerRawDecodeResult decode(const Words& words) {
    return decodeProfilerRawStream(classic, words, words.size(), 2, 7);
}

void check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

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
    test("control vector contains every processor's reset address", [] {
        std::vector<uint32_t> control(kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE);
        check(kernel_profiler::DRAM_PROFILER_ADDRESS_T2_0 < control.size(),
              "TRISC2 reset address must not write beyond the control vector");
        control.at(kernel_profiler::DRAM_PROFILER_ADDRESS_T2_0) = 0x12345678;
        check(kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE % 64 == 0,
              "control transfer and following per-RISC buffers must retain 64-byte alignment");
    });
    test("empty end index ignores allocated contents", [] {
        auto result = decodeProfilerRawStream(classic, run(), 0, 2, 7);
        check(result.status == complete && result.runs.empty(), "empty capture must decode without a fake run");
    });
    test("empty span decodes without reads", [] {
        auto result = decode({});
        check(result.status == complete && result.runs.empty(), "zero-length region must be accepted");
    });
    test("header and guaranteed slots retain actual identities", [] {
        auto result = decode(run());
        check(result.status == complete && result.runs.size() == 1, "one run expected");
        const auto& r = result.runs.front();
        check(r.word_offset == 0 && r.risc_id == 2 && r.flat_id == 7 && r.trace_counter == 3 && r.runtime_id == 99,
              "wire identity must not be rewritten");
        check(r.packets.size() == 4, "four guaranteed packets expected");
        check(r.packets[0].slot == ProfilerRawSlot::FirmwareStart && r.packets[0].word_offset == 4 &&
                  r.packets[0].timer_id == 0x11 && r.packets[0].timestamp == 100,
              "firmware start slot");
        check(r.packets[1].slot == ProfilerRawSlot::FirmwareEnd && r.packets[1].word_offset == 6 &&
                  r.packets[1].timer_id == 0x10011 && r.packets[1].timestamp == 500,
              "firmware end must retain packet type and storage order");
        check(r.packets[2].slot == ProfilerRawSlot::KernelStart && r.packets[2].word_offset == 8 &&
                  r.packets[2].timer_id == 0x22 && r.packets[2].timestamp == 200,
              "kernel start must come from slot three");
        check(r.packets[3].slot == ProfilerRawSlot::KernelEnd && r.packets[3].word_offset == 10 &&
                  r.packets[3].timer_id == 0x10022 && r.packets[3].timestamp == 400,
              "kernel end must come from slot four");
    });
    test("same timestamp duplicates and custom order survive", [] {
        auto words = run();
        words.insert(words.end(), {0x80033000, 300, 0x80033000, 300, 0x90033000, 300, 0x80044000, 299});
        auto result = decode(words);
        check(result.status == complete && result.runs[0].packets.size() == 8, "duplicates must not collapse");
        const auto& packets = result.runs[0].packets;
        check(packets[4].word_offset == 12 && packets[5].word_offset == 14 && packets[6].word_offset == 16 &&
                  packets[7].word_offset == 18 && packets[4].timestamp == 300 && packets[7].timestamp == 299 &&
                  packets[5].timer_id == 0x33 && packets[6].timer_id == 0x10033 &&
                  packets[7].slot == ProfilerRawSlot::Custom,
              "raw order and complete timer IDs must survive");
    });
    test("timestamp high bits survive without epoch inference", [] {
        auto words = run();
        words.insert(words.end(), {0xcfffffff, 0xfffffffe});
        auto result = decode(words);
        check(result.status == complete && result.runs[0].packets.back().timestamp == 0xffffffffffeULL &&
                  result.runs[0].packets.back().timer_id == 0x4ffff,
              "all 44 timestamp bits and all 19 timer bits must survive");
    });
    test("multiple runs retain duplicate runtime IDs", [] {
        auto words = run();
        auto next = run();
        words.insert(words.end(), next.begin(), next.end());
        auto result = decode(words);
        check(result.status == complete && result.runs.size() == 2 && result.runs[1].word_offset == 12 &&
                  result.runs[1].runtime_id == 99 && result.runs[1].packets[2].word_offset == 20,
              "separate runs must retain their original offsets and repeated IDs");
    });
    test("unused guaranteed slots do not become packets or headers", [] {
        auto words = run();
        words[4] = words[5] = words[6] = words[7] = 0x80000000;
        auto result = decode(words);
        check(result.status == complete && result.runs.size() == 1 && result.runs[0].packets.size() == 2 &&
                  result.runs[0].packets[0].slot == ProfilerRawSlot::KernelStart,
              "unused firmware slots must not move the kernel slots");
    });
    test("all unused guaranteed slots preserve the run", [] {
        auto words = run();
        for (unsigned i = 4; i < words.size(); ++i) {
            words[i] = 0x80000000;
        }
        auto result = decode(words);
        check(result.status == complete && result.runs.size() == 1 && result.runs[0].packets.empty(),
              "structural decoding cannot fabricate missing kernel boundaries");
    });
    test("custom padding does not swallow next run", [] {
        auto words = run();
        words.insert(words.end(), {0x80000000, 0, 0x80000000, 0});
        auto next = run();
        words.insert(words.end(), next.begin(), next.end());
        auto result = decode(words);
        check(result.status == complete && result.runs.size() == 2 && result.runs[0].packets.size() == 4 &&
                  result.runs[1].word_offset == 16,
              "padding is neither a packet nor a header");
    });
    test("zero static ID with a real timestamp is a packet", [] {
        auto words = run();
        words.insert(words.end(), {0x80000000, 17, 0x90000000, 18});
        auto result = decode(words);
        check(result.status == complete && result.runs[0].packets.size() == 6 &&
                  result.runs[0].packets[4].timer_id == 0 && result.runs[0].packets[4].timestamp == 17,
              "zero static IDs must not be dropped by timer-ID truth tests");
    });
    test("TS_DATA payload is not scanned for headers", [] {
        auto words = run();
        words.insert(words.end(), {0xb0055000, 301, 0, 0, 0xc0066000, 302});
        auto result = decode(words);
        check(result.status == complete && result.runs.size() == 1 && result.runs[0].packets.size() == 6,
              "zero payload words are not a new-run sentinel");
        const auto& packet = result.runs[0].packets[4];
        check(packet.word_offset == 12 && packet.timer_id == 0x30055 && packet.timestamp == 301 &&
                  packet.data_count == 1 && packet.data[0] == 0 && result.runs[0].packets[5].word_offset == 16,
              "TS_DATA consumes exactly one 64-bit payload");
    });
    test("TS_DATA_16B preserves payload bits and packet boundaries", [] {
        auto words = run();
        words.insert(words.end(), {0xd0055000, 303, 0x12345678, 0x9abcdef0, 0, 0, 0xc0066000, 304});
        auto result = decode(words);
        check(result.status == complete && result.runs[0].packets.size() == 6, "both data and event expected");
        const auto& packet = result.runs[0].packets[4];
        check(packet.data_count == 2 && packet.data[0] == 0x123456789abcdef0ULL && packet.data[1] == 0 &&
                  result.runs[0].packets[5].word_offset == 18,
              "TS_DATA_16B consumes exactly two 64-bit payloads");
    });
    test("ZONE_TOTAL retains raw sum without inventing a timestamp", [] {
        auto words = run();
        words.insert(words.end(), {0xa0077000, 12345});
        auto result = decode(words);
        check(result.status == complete && result.runs[0].packets.back().timer_id == 0x20077 &&
                  result.runs[0].packets.back().timestamp == 12345 && result.runs[0].packets.back().data_count == 0,
              "sum is the raw low word, not a reconstructed zone-start time");
    });
    test("control end index prevents reading trailing allocation contents", [] {
        auto words = run();
        words.insert(words.end(), {0xffffffff, 0xffffffff, 0});
        auto result = decodeProfilerRawStream(classic, words, 12, 2, 7);
        check(result.status == complete && result.runs[0].packets.size() == 4, "unreported tail must not be parsed");
    });
    test("control end beyond supplied storage rejects", [] {
        check(decodeProfilerRawStream(classic, run(), 14, 2, 7).status == truncated, "oversized end index");
    });
    test("overflowing control end rejects before indexing", [] {
        check(decodeProfilerRawStream(classic, run(), std::numeric_limits<uint32_t>::max(), 2, 7).status == truncated,
              "maximum end index cannot wrap arithmetic into a valid region");
    });
    test("odd marker extent rejects", [] {
        auto words = run();
        words.push_back(0xc0033000);
        check(decode(words).status == truncated, "one marker word cannot form a packet");
    });
    test("incomplete header rejects", [] {
        check(decode({0, 0}).status == truncated, "sentinel alone cannot form a run");
    });
    test("incomplete guaranteed prefix rejects", [] {
        auto words = run();
        words.resize(10);
        check(decode(words).status == truncated, "missing fourth slot must be truncated");
    });
    test("trailing incomplete run rejects", [] {
        auto words = run();
        words.insert(words.end(), {0, 0});
        check(decode(words).status == truncated, "last sentinel cannot be ignored");
    });
    test("TS_DATA missing payload rejects", [] {
        auto words = run();
        words.insert(words.end(), {0xb0055000, 300});
        check(decode(words).status == truncated, "data marker without payload");
    });
    test("TS_DATA_16B missing second payload rejects", [] {
        auto words = run();
        words.insert(words.end(), {0xd0055000, 300, 0, 0});
        check(decode(words).status == truncated, "16B data must not accept one payload");
    });
    test("foreign RISC rejects at header", [] {
        check(decodeProfilerRawStream(classic, run(), 12, 3, 7).status == malformed, "wrong RISC cannot be attributed");
    });
    test("foreign flat core rejects at header", [] {
        check(decodeProfilerRawStream(classic, run(), 12, 2, 8).status == malformed, "wrong core cannot be attributed");
    });
    test("foreign identity on a later run rejects", [] {
        auto words = run();
        auto next = run();
        next[2] = 0x60e3;
        words.insert(words.end(), next.begin(), next.end());
        check(decode(words).status == malformed, "every run header needs identity verification");
    });
    test("unrepresentable expected flat core rejects", [] {
        check(decodeProfilerRawStream(classic, run(), 12, 2, 263).status == malformed,
              "expected identity must not be masked to match");
    });
    test("nonexistent expected RISC rejects", [] {
        auto words = run();
        words[2] = 0x60f8;
        check(decodeProfilerRawStream(classic, words, 12, 24, 7).status == malformed,
              "unused RISC encodings are not supported processors");
    });
    test("reserved header bits reject", [] {
        auto words = run();
        words[2] |= 0x20000000;
        check(decode(words).status == malformed, "reserved identity bits cannot be ignored");
    });
    test("invalid custom marker validity rejects", [] {
        auto words = run();
        words.insert(words.end(), {0x40033000, 300});
        check(decode(words).status == malformed, "missing validity cannot be accepted from timer bits");
    });
    test("zero guaranteed slots are not a nested sentinel", [] {
        auto words = run();
        words[4] = words[5] = 0;
        check(decode(words).status == malformed, "guaranteed-slot validity is not run framing");
    });
    test("wrong packet kind in guaranteed slot rejects", [] {
        auto words = run();
        words[8] = 0xb0022000;
        check(decode(words).status == malformed, "kernel-start slot cannot contain a data packet");
    });
    test("reversed guaranteed boundary packet kind rejects", [] {
        auto words = run();
        words[10] = 0x80022000;
        check(decode(words).status == malformed, "end slot cannot silently relabel a start");
    });
    test("unknown packet kind rejects", [] {
        auto words = run();
        words.insert(words.end(), {0xe0033000, 300});
        check(decode(words).status == malformed, "reserved type six must not be skipped");
    });
    test("pre-sentinel packets reject", [] {
        auto words = run();
        words.insert(words.begin(), {0xb0033000, 99, 0, 1});
        check(decode(words).status == malformed, "no guessing which later run owns pre-sentinel data");
    });
    test("nonzero sentinel rejects", [] {
        auto words = run();
        words[1] = 1;
        check(decode(words).status == malformed, "sentinel must be exactly two zero words");
    });
    test("unsupported protocol rejects even empty data", [] {
        auto result = decodeProfilerRawStream(static_cast<ProfilerRawProtocol>(2), {}, 0, 2, 7);
        check(result.status == ProfilerRawDecodeStatus::UnsupportedProtocol && result.runs.empty(),
              "a different wire protocol must not be decoded as classic");
    });

    std::printf("%u/%u tests passed\n", tests - failures, tests);
    return failures ? 1 : 0;
}
