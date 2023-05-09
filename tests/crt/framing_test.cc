/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <gtest/gtest.h>
#include <tvm/runtime/crt/page_allocator.h>
#include <tvm/runtime/crt/rpc_common/frame_buffer.h>
#include <tvm/runtime/crt/rpc_common/framing.h>

#include <string>
#include <vector>

#include "buffer_write_stream.h"
#include "crt_config.h"

using ::tvm::runtime::micro_rpc::Escape;
using ::tvm::runtime::micro_rpc::FrameBuffer;
using ::tvm::runtime::micro_rpc::Framer;
using ::tvm::runtime::micro_rpc::Unframer;

class FramerTest : public ::testing::Test {
 protected:
  BufferWriteStream<300> write_stream_;
  Framer framer_{&write_stream_};
};

class TestPacket {
 public:
  static std::vector<const TestPacket*> instances;

  // NOTE: take payload and wire as arrays to avoid clipping at \0
  template <int N, int M>
  TestPacket(const std::string name, const char (&payload)[N], const char (&wire)[M])
      : name{name}, payload{payload, N - 1}, wire{wire, M - 1} {  // omit trailing \0
    instances.emplace_back(this);
  }

  inline const uint8_t* payload_data() const {
    return reinterpret_cast<const uint8_t*>(payload.data());
  }

  inline const uint8_t* wire_data() const { return reinterpret_cast<const uint8_t*>(wire.data()); }

  std::string name;
  std::string payload;
  std::string wire;
};

std::vector<const TestPacket*> TestPacket::instances;

#define TEST_PACKET(name, payload, wire) \
  static const TestPacket k##name {      \
#name, payload, wire                 \
  }

// NOTE: golden packet CRCs are generated with this python:
// import binascii
// import struct
// struct.pack('<H', binascii.crc_hqx('\xff\xfd\x05\0\0\0three', 0xffff))

TEST_PACKET(Packet1, "one", "\xff\xfd\3\0\0\0one\x58\xf4");
TEST_PACKET(Packet2, "two2", "\xff\xfd\4\0\0\0two2\x13\x11");
TEST_PACKET(Packet3, "three", "\xff\xfd\5\0\0\0threec\x9f");
TEST_PACKET(EscapeCodeInSizePacket,
            "this payload is exactly 255 characters long. chunk is 64 bytes. "
            "this payload is exactly 255 characters long. chunk is 64 bytes. "
            "this payload is exactly 255 characters long. chunk is 64 bytes. "
            "this payload is exactly 255 characters long. chunk is 64 bytes.",
            "\xff\xfd\xff\xff\0\0\0"
            "this payload is exactly 255 characters long. chunk is 64 bytes. "
            "this payload is exactly 255 characters long. chunk is 64 bytes. "
            "this payload is exactly 255 characters long. chunk is 64 bytes. "
            "this payload is exactly 255 characters long. chunk is 64 bytes."
            "6~");
TEST_PACKET(ZeroLengthPacket, "", "\xff\xfd\0\0\0\0\203D");

// Generated with:
// import binascii
// import random
// import string
// import struct
// escaped_prefix = b'es_\xff\xff_cape'
// crc = b''
// while b'\xff' not in crc:
//   suffix = bytes(''.join(random.choices(string.printable, k=10)), 'utf-8')
//   packet = b'\xff\xfd' + struct.pack('<I', len(escaped_prefix + suffix)) + escaped_prefix +
//   suffix crc = struct.pack('<H', binascii.crc_hqx(packet, 0xffff))
// print(suffix)
// print(packet + crc.replace(b'\xff', b'\xff\xff'))
TEST_PACKET(EscapePacket, "es_\xff_capeir/^>t@\"hr",
            "\xff\xfd\x13\0\0\0es_\xff\xff_capeir/^>t@\"hr\xb4\xff\xff");

TEST_F(FramerTest, ValidPacketTrain) {
  EXPECT_EQ(kTvmErrorNoError, framer_.Write(kPacket1.payload_data(), kPacket1.payload.size()));
  EXPECT_EQ(kTvmErrorNoError, framer_.Write(kPacket2.payload_data(), kPacket2.payload.size()));
  framer_.Reset();
  EXPECT_EQ(kTvmErrorNoError, framer_.Write(kPacket3.payload_data(), kPacket3.payload.size()));

  EXPECT_EQ("\xfe" + kPacket1.wire +     // packet1 plus nop prefix.
                kPacket2.wire +          // packet2, no prefix.
                "\xfe" + kPacket3.wire,  // packet3 plus nop prefix.
            write_stream_.BufferContents());
}

TEST_F(FramerTest, ZeroLengthPacket) {
  EXPECT_EQ(kTvmErrorNoError,
            framer_.Write(kZeroLengthPacket.payload_data(), kZeroLengthPacket.payload.size()));
  EXPECT_EQ("\xfe" + kZeroLengthPacket.wire, write_stream_.BufferContents());
}

TEST_F(FramerTest, Escapes) {
  EXPECT_EQ(kTvmErrorNoError,
            framer_.Write(kEscapePacket.payload_data(), kEscapePacket.payload.size()));
  EXPECT_EQ("\xfe" + kEscapePacket.wire, write_stream_.BufferContents());
}

class UnframerTest : public ::testing::Test {
 protected:
  BufferWriteStream<300> write_stream_;
  Unframer unframer_{&write_stream_};
};

TEST_F(UnframerTest, PacketTooLong) {
  const uint8_t escape[2] = {uint8_t(Escape::kEscapeStart), uint8_t(Escape::kPacketStart)};
  uint16_t crc = tvm::runtime::micro_rpc::crc16_compute(escape, sizeof(escape), nullptr);
  size_t bytes_consumed;
  EXPECT_EQ(kTvmErrorNoError, unframer_.Write(escape, sizeof(escape), &bytes_consumed));
  EXPECT_EQ(sizeof(escape), bytes_consumed);

  uint32_t packet_length = write_stream_.capacity() + 1;
  uint8_t* packet_length_bytes = reinterpret_cast<uint8_t*>(&packet_length);
  for (size_t i = 0; i < sizeof(packet_length); i++) {
    ASSERT_NE('\xff', packet_length_bytes[i]);
  }
  crc = tvm::runtime::micro_rpc::crc16_compute(packet_length_bytes, sizeof(packet_length), &crc);
  EXPECT_EQ(kTvmErrorNoError,
            unframer_.Write(packet_length_bytes, sizeof(packet_length), &bytes_consumed));
  EXPECT_EQ(sizeof(packet_length), bytes_consumed);

  unsigned int long_payload_len = decltype(write_stream_)::capacity() + 1;
  auto long_payload = std::make_unique<uint8_t[]>(long_payload_len);
  for (size_t i = 0; i < long_payload_len; i++) {
    long_payload[i] = i & 0xff;
    if (long_payload[i] == uint8_t(Escape::kEscapeStart)) {
      long_payload[i] = 0;
    }
  }
  crc = tvm::runtime::micro_rpc::crc16_compute(long_payload.get(), long_payload_len, &crc);
  EXPECT_EQ(kTvmErrorWriteStreamShortWrite,
            unframer_.Write(long_payload.get(), long_payload_len, &bytes_consumed));
  EXPECT_EQ(write_stream_.capacity(), bytes_consumed);

  EXPECT_EQ(kTvmErrorNoError,
            unframer_.Write(reinterpret_cast<uint8_t*>(&crc), sizeof(crc), &bytes_consumed));
  EXPECT_EQ(2UL, bytes_consumed);  // 2, because framer is now in kFindPacketStart.
  EXPECT_FALSE(write_stream_.packet_done());
  EXPECT_FALSE(write_stream_.is_valid());
  EXPECT_EQ(std::string(reinterpret_cast<char*>(long_payload.get()), write_stream_.capacity()),
            write_stream_.BufferContents());

  // Writing a smaller packet directly afterward should work.
  write_stream_.Reset();
  EXPECT_EQ(kTvmErrorNoError,
            unframer_.Write(kPacket1.wire_data(), kPacket1.wire.size(), &bytes_consumed));
  EXPECT_EQ(kPacket1.wire.size(), bytes_consumed);
  EXPECT_TRUE(write_stream_.packet_done());
  EXPECT_TRUE(write_stream_.is_valid());
  EXPECT_EQ(kPacket1.payload, write_stream_.BufferContents());
}

class UnframerTestParameterized : public UnframerTest,
                                  public ::testing::WithParamInterface<const TestPacket*> {};

TEST_P(UnframerTestParameterized, TestFullPacket) {
  size_t bytes_consumed;
  EXPECT_EQ(kTvmErrorNoError,
            unframer_.Write(GetParam()->wire_data(), GetParam()->wire.size(), &bytes_consumed));
  EXPECT_EQ(GetParam()->wire.size(), bytes_consumed);
  EXPECT_TRUE(write_stream_.packet_done());
  EXPECT_TRUE(write_stream_.is_valid());
  EXPECT_EQ(GetParam()->payload, write_stream_.BufferContents());
}

TEST_P(UnframerTestParameterized, TestByteAtATime) {
  size_t bytes_consumed;
  size_t wire_size = GetParam()->wire.size();
  for (size_t i = 0; i < wire_size; i++) {
    EXPECT_EQ(kTvmErrorNoError,
              unframer_.Write(reinterpret_cast<const uint8_t*>(&GetParam()->wire[i]), 1,
                              &bytes_consumed));
    EXPECT_EQ(1UL, bytes_consumed);
    EXPECT_EQ(i == wire_size - 1, write_stream_.packet_done());
  }
  EXPECT_TRUE(write_stream_.is_valid());
  EXPECT_EQ(GetParam()->payload, write_stream_.BufferContents());
}

TEST_P(UnframerTestParameterized, TestArbitraryBoundary) {
  size_t bytes_consumed;
  size_t wire_size = GetParam()->wire.size();
  for (size_t i = 1; i < wire_size; i++) {
    unframer_.Reset();
    write_stream_.Reset();
    EXPECT_EQ(kTvmErrorNoError, unframer_.Write(GetParam()->wire_data(), i, &bytes_consumed));
    EXPECT_EQ(i, bytes_consumed);
    EXPECT_FALSE(write_stream_.packet_done());
    EXPECT_EQ(kTvmErrorNoError,
              unframer_.Write(&GetParam()->wire_data()[i], wire_size - i, &bytes_consumed));
    EXPECT_EQ(wire_size - i, bytes_consumed);
    EXPECT_TRUE(write_stream_.packet_done());
    EXPECT_TRUE(write_stream_.is_valid());
    EXPECT_EQ(GetParam()->payload, write_stream_.BufferContents());
  }
}

TEST_P(UnframerTestParameterized, TestArbitraryPacketReset) {
  size_t bytes_consumed;
  size_t wire_size = GetParam()->wire.size();

  // This test interrupts packet transmission at an arbitrary point in the packet and restarts from
  // the beginning. It simulates handling a device reset in the protocol. The behavior of the framer
  // depends on how much of the packet had been transmitted, so the test is split into parts:

  // Part 1. Restarting during the initial escape sequence.
  unframer_.Reset();
  write_stream_.Reset();
  EXPECT_EQ(kTvmErrorNoError, unframer_.Write(GetParam()->wire_data(), 1, &bytes_consumed));
  EXPECT_EQ(1UL, bytes_consumed);
  EXPECT_EQ(kTvmErrorNoError, unframer_.Write(GetParam()->wire_data(), wire_size, &bytes_consumed));
  EXPECT_EQ(wire_size, bytes_consumed);
  EXPECT_TRUE(write_stream_.packet_done());
  EXPECT_TRUE(write_stream_.is_valid());
  EXPECT_EQ(GetParam()->payload, write_stream_.BufferContents());

  // Part 2. Restarting after the initial escape sequence.
  for (size_t i = 2; i < wire_size; i++) {
    unframer_.Reset();
    write_stream_.Reset();
    EXPECT_EQ(kTvmErrorNoError, unframer_.Write(GetParam()->wire_data(), i, &bytes_consumed));
    EXPECT_EQ(i, bytes_consumed);

    // First test byte-by-byte interruption.
    // Interrupt the packet transmission. The first byte will return no error as it is the escape
    // byte.
    EXPECT_EQ(kTvmErrorNoError, unframer_.Write(GetParam()->wire_data(), 1, &bytes_consumed));
    EXPECT_EQ(1UL, bytes_consumed);
    EXPECT_FALSE(write_stream_.packet_done());

    // Secondt byte will return a short packet error.
    EXPECT_EQ(kTvmErrorFramingShortPacket,
              unframer_.Write(&GetParam()->wire_data()[1], 1, &bytes_consumed));
    EXPECT_EQ(0UL, bytes_consumed);
    EXPECT_FALSE(write_stream_.packet_done());

    EXPECT_EQ(kTvmErrorNoError,
              unframer_.Write(&GetParam()->wire_data()[1], wire_size - 1, &bytes_consumed));
    EXPECT_EQ(wire_size - 1, bytes_consumed);
    EXPECT_TRUE(write_stream_.packet_done());
    EXPECT_TRUE(write_stream_.is_valid());
    EXPECT_EQ(GetParam()->payload, write_stream_.BufferContents());

    // Next, test interruption just by sending the whole payload at once.
    unframer_.Reset();
    write_stream_.Reset();
    EXPECT_EQ(kTvmErrorNoError, unframer_.Write(GetParam()->wire_data(), i, &bytes_consumed));
    EXPECT_EQ(i, bytes_consumed);

    // Interrupt the packet transmission. The first Write() call will just consume 1 byte to reset
    // the internal state.
    EXPECT_EQ(kTvmErrorFramingShortPacket,
              unframer_.Write(GetParam()->wire_data(), wire_size, &bytes_consumed));
    EXPECT_EQ(1UL, bytes_consumed);
    EXPECT_FALSE(write_stream_.packet_done());
    EXPECT_EQ(kTvmErrorNoError,
              unframer_.Write(&GetParam()->wire_data()[1], wire_size - 1, &bytes_consumed));
    EXPECT_EQ(wire_size - 1, bytes_consumed);
    EXPECT_TRUE(write_stream_.packet_done());
    EXPECT_TRUE(write_stream_.is_valid());
    EXPECT_EQ(GetParam()->payload, write_stream_.BufferContents());

    break;
  }
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
INSTANTIATE_TEST_CASE_P(UnframerTests, UnframerTestParameterized,
                        ::testing::ValuesIn(TestPacket::instances));
#pragma GCC diagnostic pop
