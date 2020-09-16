/*
 * Library: libcrc
 * File:    precalc/crc32_table.c
 * Author:  Lammert Bies
 *
 * This file is licensed under the MIT License as stated below
 *
 * Copyright (c) 1999-2016 Lammert Bies
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
 * The source file precalc/crc32_table.c contains the routines to calculate the
 * values in the lookup table used to calculate 32 bit CRC values.
 */

#include <stdbool.h>
#include <stdlib.h>
#include "checksum.h"
#include "precalc.h"

/*
 * void init_crc32_tab( void );
 *
 * For optimal speed, the CRC32 calculation uses a table with pre-calculated
 * bit patterns which are used in the XOR operations in the program.
 */

void init_crc32_tab( void ) {

	uint32_t i;
	uint32_t j;
	uint32_t crc;

	for (i=0; i<256; i++) {

		crc = i;

		for (j=0; j<8; j++) {

			if ( crc & 0x00000001L ) crc = ( crc >> 1 ) ^ CRC_POLY_32;
			else                     crc =   crc >> 1;
		}

		crc_tab_precalc[i] = crc;
	}

}  /* init_crc32_tab */
