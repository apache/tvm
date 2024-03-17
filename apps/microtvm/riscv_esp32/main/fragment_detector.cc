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

#include "fragment_detector.h"

FragmentDetector::FragmentDetector(Detection& det) : det_(det) {
  // Number of samples in a fragment
  samples_per_fragment_det_ = PARTS(A_SAMPLE_RATE * det_.word_det_fragment_time_msec, 1000);
  // Number of samples in the indentation before the word
  samples_per_before_det_ = PARTS(A_SAMPLE_RATE * det_.before_word_time_msec, 1000);
  // Characteristic number of fragments to determine the beginning of a word
  fragments_per_word_det_ = PARTS(det_.word_det_time_msec, det_.word_det_fragment_time_msec);
  // Characteristic number of fragments for interword detection
  fragments_per_interword_det_ =
      PARTS(det_.interword_det_time_msec, det_.word_det_fragment_time_msec);
  // Number of fragments for syllable duration
  fragments_per_syllable_det_ =
      PARTS(det_.syllable_det_time_msec, det_.word_det_fragment_time_msec);
}

FragmentDetector::~FragmentDetector() {}

// Prepare for a new detection process
void FragmentDetector::Init(const size_t samples, const size_t bytes_width) {
  det_.fragment_byte_size = 0;
  if (det_.fragments) det_.fragments->clear();
  bytes_width_ = bytes_width;
  samples_ = samples;
  sample_num_ = 0;
  fragment_trait_ = d0;
  fragment_samples_ = 0;
  det_trait_ = 0;
  det_sample_num_ = 0;
  det_sample_cnt_ = 0;
  word_fragments_ = 0;
  interword_fragments_ = 0;
  word_trait_ = d0;
  word_sample_num_ = 0;
}

// Pass a sample to the detection logic:
// divide the stream into short fragments (e.g. 10 msec), calculate the characteristic parameters
// for these fragments and analyze them
void FragmentDetector::Detect(int32_t value) {
  if (sample_num_ >= samples_) return;

  // Calculate the fragment characteristic parameter (fragment_trait_)
  if (value < 0) {
    if (-value > det_.environment) fragment_trait_ -= static_cast<proc_t>(value);
  } else {
    if (value > det_.environment) fragment_trait_ += static_cast<proc_t>(value);
  }

  if (++fragment_samples_ >= samples_per_fragment_det_) {
    // Here we have a fragment (sequential stream interval) and its characteristic parameter
    fragment_samples_ = 0;
    if (fragment_trait_) {
      // A potentially useful fragment; calculate the word characteristic parameter (word_trait)
      if (!word_fragments_) {
        word_sample_num_ = sample_num_ + 1 - samples_per_fragment_det_;
        word_trait_ = fragment_trait_;
      } else
        word_trait_ += fragment_trait_;

      word_fragments_++;
      interword_fragments_ = 0;
    } else {
      // A potentially unuseful fragment
      if (++interword_fragments_ >= fragments_per_interword_det_) {
        if (word_fragments_ >= fragments_per_word_det_) {
          // Here the word ended on the current fragment
          const size_t interword_sample_num =
              sample_num_ + 1 - interword_fragments_ * samples_per_fragment_det_;
          const size_t word_end_sample_num =
              ((interword_sample_num - word_sample_num_) >=
               (fragments_per_syllable_det_ * samples_per_fragment_det_))
                  ? interword_sample_num - 1
                  : word_sample_num_ + fragments_per_syllable_det_ * samples_per_fragment_det_ - 1;
          const float trait = static_cast<float>(word_trait_) / word_fragments_;
          if (trait > det_trait_) {
            // Choose the 'strongest' word
            det_trait_ = trait;
            det_sample_num_ = (word_sample_num_ >= samples_per_before_det_)
                                  ? word_sample_num_ - samples_per_before_det_
                                  : 0;
            det_sample_cnt_ = word_end_sample_num - det_sample_num_ + 1;
          }
          if (det_.fragments) {
            // Collect detected words
            DetectItem di;
            const size_t sn = (word_sample_num_ >= samples_per_before_det_)
                                  ? word_sample_num_ - samples_per_before_det_
                                  : 0;
            di.fragment_byte_offset = sn * bytes_width_;
            di.fragment_byte_size = (word_end_sample_num - sn + 1) * bytes_width_;
            det_.fragments->push_back(di);
          }
        }
        word_fragments_ = 0;
      }
    }
    fragment_trait_ = d0;
  }

  if (++sample_num_ >= samples_) {
    if (word_fragments_ >= fragments_per_word_det_) {
      // The word is detected at the end of the stream interval
      const float trait = static_cast<float>(word_trait_) / word_fragments_;
      if (trait > det_trait_) {
        det_sample_num_ = (word_sample_num_ >= samples_per_before_det_)
                              ? word_sample_num_ - samples_per_before_det_
                              : 0;
        det_sample_cnt_ = samples_ - det_sample_num_;
      }
      if (det_.fragments) {
        DetectItem di;
        const size_t sn = (word_sample_num_ >= samples_per_before_det_)
                              ? word_sample_num_ - samples_per_before_det_
                              : 0;
        di.fragment_byte_offset = sn * bytes_width_;
        di.fragment_byte_size = (samples_ - sn) * bytes_width_;
        det_.fragments->push_back(di);
      }
    }
    det_.fragment_byte_size = det_sample_cnt_ * bytes_width_;
    det_.fragment_byte_offset = det_sample_num_ * bytes_width_;
  }
}
