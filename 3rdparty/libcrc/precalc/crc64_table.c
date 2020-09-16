/*
 * Library: libcrc
 * File:    precalc/crc64_table.c
 * Author:  Lammert Bies
 *
 * This file is licensed under the MIT License as stated below
 *
 * Copyright (c) 2016 Lammert Bies
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Description
 * -----------
 * The source file precalc/crc64_table.c contains the routines which are needed
 * to generate the lookup table for 64 bit CRC calculations.
 */

#include <stdbool.h>
#include <stdlib.h>
#include "checksum.h"
#include "precalc.h"

/*
 * void init_crc64_tab( void );
 *
 * For optimal speed, the CRC64 calculation uses a table with pre-calculated
 * bit patterns which are used in the XOR operations in the program. This table
 * is generated during compilation of the library and added to the library as a
 * table with constant values.
 */

void init_crc64_tab( void ) {

	uint64_t i;
	uint64_t j;
	uint64_t c;
	uint64_t crc;

	for (i=0; i<256; i++) {

		crc = 0;
		c   = i << 56;

		for (j=0; j<8; j++) {

			if ( ( crc ^ c ) & 0x8000000000000000ull ) crc = ( crc << 1 ) ^ CRC_POLY_64;
			else                                       crc =   crc << 1;

			c = c << 1;
		}

		crc_tab_precalc[i] = crc;
	}

}  /* init_crc64_tab */
