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

#ifndef _FRAG_H_
#define _FRAG_H_

#include "spec.h"

/*
 * Two ways of preparing MFCC coefficients before shifts are proposed:
 * 1. coefficients are recalculated for each phrase (any A_MFCC_xx_PREPARED_TEMPLATE flags are not
 *    defined; the microphone stream is read in chunks of 1000 msec)
 * 2. coefficients are prepared only 1 time before the start of the main procedure cycle
 *    A_MFCC_ZERO_PREPARED_TEMPLATE - filled with zeros
 *    A_MFCC_NOISE_PREPARED_TEMPLATE - filled with random data
 *    A_MFCC_ENVIRONMENT_PREPARED_TEMPLATE - calculated based on the real noise of the environment
 *      (microphone stream is taken in portions of 500 msec).
 */

/*
 * A_STRIDE_MS: determines the size of the stride of the audio buffer data
 *  (not to be confused with the stride of the MFCC coefficients calculation procedure)
 * A_INIT_STRIDE_MS: determines the size of the first stride of the audio buffer data
 * A_STRIDED_DURATION_MS: determines the boundary for the last stride of the audio buffer data
 */

#define A_STRIDE_MS KWS_WIN_MS
#define A_INIT_STRIDE_MS (5 * A_STRIDE_MS)
#define A_STRIDED_DURATION_MS (A_INIT_STRIDE_MS + 2 * A_STRIDE_MS)

/*
 * A_DET_START_TIME_MS: the time to start and turn on the audio interface
 * A_DET_ENVIRONMENT_TIME_MS: the time interval for analyzing the mic environment (silence)
 * A_DET_WORD_FRAGMENT_TIME_MS: the time interval by which the cumulative values
 *  of the flow characteristics are calculated
 * A_DET_WORD_TIME_MS: the time interval by which the definition of the beginning
 *  of a word is performed
 * A_DET_INTERWORD_TIME_MS: the time interval by which the end of the word is determined
 * A_DET_BEFORE_WORD_MS: the interval of 'silence' before the word for the inference procedure
 * A_DET_SYLLABLE_TIME_MS: estimated syllable duration
 */

#define A_DET_START_TIME_MS 1000
#define A_DET_ENVIRONMENT_TIME_MS 5000
#define A_DET_WORD_FRAGMENT_TIME_MS 10
#define A_DET_WORD_TIME_MS 50
#define A_DET_BEFORE_WORD_MS 50
#define A_DET_INTERWORD_TIME_MS 250
#define A_DET_SYLLABLE_TIME_MS 250

#endif  // _FRAG_H_
