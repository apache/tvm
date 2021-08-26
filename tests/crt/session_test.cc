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
#include <tvm/runtime/crt/rpc_common/session.h>

#include <string>
#include <vector>

#include "buffer_write_stream.h"
#include "crt_config.h"

using ::tvm::runtime::micro_rpc::Framer;
using ::tvm::runtime::micro_rpc::MessageType;
using ::tvm::runtime::micro_rpc::Session;
using ::tvm::runtime::micro_rpc::Unframer;

extern "C" {
void TestSessionMessageReceivedThunk(void* context, MessageType message_type, FrameBuffer* buf);
}

class ReceivedMessage {
 public:
  ReceivedMessage(MessageType type, std::string message) : type{type}, message{message} {}

  bool operator==(const ReceivedMessage& other) const {
    return other.type == type && other.message == message;
  }

  MessageType type;
  std::string message;
};

class TestSession {
 public:
  explicit TestSession(uint8_t initial_nonce)
      : framer{&framer_write_stream},
        receive_buffer{receive_buffer_array, sizeof(receive_buffer_array)},
        sess{&framer, &receive_buffer, TestSessionMessageReceivedThunk, this},
        unframer{sess.Receiver()},
        initial_nonce{initial_nonce} {}

  void WriteTo(TestSession* other) {
    auto framer_buffer = framer_write_stream.BufferContents();
    size_t bytes_to_write = framer_buffer.size();
    const uint8_t* write_cursor = reinterpret_cast<const uint8_t*>(framer_buffer.data());
    while (bytes_to_write > 0) {
      size_t bytes_consumed;
      auto to_return = other->unframer.Write(write_cursor, bytes_to_write, &bytes_consumed);
      EXPECT_EQ(to_return, kTvmErrorNoError);
      bytes_to_write -= bytes_consumed;
      write_cursor += bytes_consumed;
    }
  }

  void ClearBuffers() {
    framer_write_stream.Reset();
    messages_received.clear();
    sess.ClearReceiveBuffer();
  }

  std::vector<ReceivedMessage> messages_received;
  BufferWriteStream<300> framer_write_stream;
  Framer framer;
  uint8_t receive_buffer_array[300];
  FrameBuffer receive_buffer;
  Session sess;
  Unframer unframer;
  uint8_t initial_nonce;
};

#define EXPECT_FRAMED_PACKET(session, expected)          \
  EXPECT_EQ(std::string(expected, sizeof(expected) - 1), \
            (session).framer_write_stream.BufferContents());

extern "C" {
void TestSessionMessageReceivedThunk(void* context, MessageType message_type, FrameBuffer* buf) {
  std::string message;
  if (message_type != MessageType::kStartSessionReply) {
    uint8_t message_buf[300];
    EXPECT_LE(buf->ReadAvailable(), sizeof(message_buf));
    size_t message_size_bytes = buf->Read(message_buf, sizeof(message_buf));
    message = std::string(reinterpret_cast<char*>(message_buf), message_size_bytes);
  }

  static_cast<TestSession*>(context)->messages_received.emplace_back(
      ReceivedMessage(message_type, message));
}
}

class SessionTest : public ::testing::Test {
 public:
  static constexpr const uint8_t kAliceNonce = 0x3c;
  static constexpr const uint8_t kBobNonce = 0xab;

  TestSession alice_{kAliceNonce};
  TestSession bob_{kBobNonce};
};

TEST_F(SessionTest, NormalExchange) {
  tvm_crt_error_t err;
  err = alice_.sess.Initialize(alice_.initial_nonce);
  EXPECT_EQ(kTvmErrorNoError, err);
  EXPECT_FRAMED_PACKET(alice_,
                       "\xfe\xff\xfd\x03\0\0\0\0\0\x02"
                       "fw");
  alice_.WriteTo(&bob_);

  err = bob_.sess.Initialize(bob_.initial_nonce);
  EXPECT_EQ(kTvmErrorNoError, err);
  EXPECT_FRAMED_PACKET(bob_,
                       "\xfe\xff\xfd\x03\0\0\0\0\0\x02"
                       "fw");
  alice_.WriteTo(&alice_);

  bob_.ClearBuffers();
  alice_.ClearBuffers();

  err = alice_.sess.StartSession();
  EXPECT_EQ(err, kTvmErrorNoError);
  EXPECT_FRAMED_PACKET(alice_, "\xff\xfd\x04\0\0\0\x82\0\0\x01{\xE9");

  bob_.ClearBuffers();
  alice_.WriteTo(&bob_);
  EXPECT_FRAMED_PACKET(bob_,
                       "\xff\xfd\x4\0\0\0\x82"
                       "f\x01\x01\x81\xf3");
  EXPECT_TRUE(bob_.sess.IsEstablished());

  bob_.WriteTo(&alice_);
  EXPECT_TRUE(alice_.sess.IsEstablished());
  ASSERT_EQ(alice_.messages_received.size(), 1UL);
  EXPECT_EQ(alice_.messages_received[0], ReceivedMessage(MessageType::kStartSessionReply, ""));

  alice_.ClearBuffers();
  alice_.sess.SendMessage(MessageType::kNormal, reinterpret_cast<const uint8_t*>("hello"), 5);
  EXPECT_FRAMED_PACKET(alice_,
                       "\xFF\xFD\b\0\0\0\x82"
                       "f\x10hello\x90(");
  alice_.WriteTo(&bob_);
  ASSERT_EQ(bob_.messages_received.size(), 2UL);
  EXPECT_EQ(bob_.messages_received[0], ReceivedMessage(MessageType::kStartSessionReply, ""));
  EXPECT_EQ(bob_.messages_received[1], ReceivedMessage(MessageType::kNormal, "hello"));

  bob_.ClearBuffers();
  bob_.sess.SendMessage(MessageType::kNormal, reinterpret_cast<const uint8_t*>("olleh"), 5);
  EXPECT_FRAMED_PACKET(bob_,
                       "\xff\xfd\b\0\0\0\x82"
                       "f\x10ollehLv");
  bob_.WriteTo(&alice_);
  ASSERT_EQ(alice_.messages_received.size(), 1UL);
  EXPECT_EQ(alice_.messages_received[0], ReceivedMessage(MessageType::kNormal, "olleh"));

  alice_.ClearBuffers();
  bob_.ClearBuffers();

  alice_.sess.SendMessage(MessageType::kLog, reinterpret_cast<const uint8_t*>("log1"), 4);
  EXPECT_FRAMED_PACKET(alice_, "\xff\xfd\a\0\0\0\0\0\x03log1\xf0\xd4");
  alice_.WriteTo(&bob_);
  ASSERT_EQ(bob_.messages_received.size(), 1UL);
  EXPECT_EQ(bob_.messages_received[0], ReceivedMessage(MessageType::kLog, "log1"));

  bob_.sess.SendMessage(MessageType::kLog, reinterpret_cast<const uint8_t*>("zero"), 4);
  EXPECT_FRAMED_PACKET(bob_, "\xff\xfd\a\0\0\0\0\0\x03zero\xb2h");
  bob_.WriteTo(&alice_);
  ASSERT_EQ(alice_.messages_received.size(), 1UL);
  EXPECT_EQ(alice_.messages_received[0], ReceivedMessage(MessageType::kLog, "zero"));
}

TEST_F(SessionTest, LogBeforeSessionStart) {
  alice_.sess.SendMessage(MessageType::kLog, reinterpret_cast<const uint8_t*>("log1"), 4);
  EXPECT_FRAMED_PACKET(alice_, "\xfe\xff\xfd\a\0\0\0\0\0\x03log1\xf0\xd4");
  alice_.WriteTo(&bob_);
  ASSERT_EQ(bob_.messages_received.size(), 1UL);
  EXPECT_EQ(bob_.messages_received[0], ReceivedMessage(MessageType::kLog, "log1"));

  bob_.sess.SendMessage(MessageType::kLog, reinterpret_cast<const uint8_t*>("zero"), 4);
  EXPECT_FRAMED_PACKET(bob_, "\xfe\xff\xfd\a\0\0\0\0\0\x03zero\xb2h");
  bob_.WriteTo(&alice_);
  ASSERT_EQ(alice_.messages_received.size(), 1UL);
  EXPECT_EQ(alice_.messages_received[0], ReceivedMessage(MessageType::kLog, "zero"));
}

static constexpr const char kBobStartPacket[] = "\xff\xfd\x04\0\0\0f\0\0\x01`\xa7";

TEST_F(SessionTest, DoubleStart) {
  tvm_crt_error_t err;
  err = alice_.sess.Initialize(alice_.initial_nonce);
  EXPECT_EQ(kTvmErrorNoError, err);
  EXPECT_FRAMED_PACKET(alice_,
                       "\xfe\xff\xfd\x03\0\0\0\0\0\x02"
                       "fw");
  alice_.WriteTo(&bob_);

  err = bob_.sess.Initialize(bob_.initial_nonce);
  EXPECT_EQ(kTvmErrorNoError, err);
  EXPECT_FRAMED_PACKET(bob_,
                       "\xfe\xff\xfd\x03\0\0\0\0\0\x02"
                       "fw");
  alice_.WriteTo(&alice_);

  bob_.ClearBuffers();
  alice_.ClearBuffers();

  EXPECT_EQ(kTvmErrorNoError, alice_.sess.StartSession());
  EXPECT_FRAMED_PACKET(alice_, "\xff\xfd\x04\0\0\0\x82\0\0\x01{\xe9");
  EXPECT_FALSE(alice_.sess.IsEstablished());

  EXPECT_EQ(kTvmErrorNoError, bob_.sess.StartSession());
  EXPECT_FRAMED_PACKET(bob_, kBobStartPacket);
  EXPECT_FALSE(bob_.sess.IsEstablished());

  // Sending Alice -> Bob should have no effect (regenerated Bob nonce > regenerated Alice nonce).
  bob_.framer_write_stream.Reset();
  alice_.WriteTo(&bob_);
  EXPECT_FRAMED_PACKET(bob_, "");
  EXPECT_FALSE(bob_.sess.IsEstablished());

  // Sending Bob -> Alice should start the session.
  alice_.ClearBuffers();
  size_t bytes_consumed;
  EXPECT_EQ(kTvmErrorNoError,
            alice_.unframer.Write(reinterpret_cast<const uint8_t*>(kBobStartPacket),
                                  sizeof(kBobStartPacket), &bytes_consumed));
  EXPECT_EQ(bytes_consumed, sizeof(kBobStartPacket));
  EXPECT_FRAMED_PACKET(alice_, "\xFF\xFD\x4\0\0\0fE\x01\x01\fb");
  EXPECT_TRUE(alice_.sess.IsEstablished());

  bob_.ClearBuffers();
  alice_.WriteTo(&bob_);
  EXPECT_TRUE(bob_.sess.IsEstablished());
}
