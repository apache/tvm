/*
 * Library: libcrc
 * File:    src/crcsick.c
 * Author:  Lammert Bies
 *
 * This file is licensed under the MIT License as stated below
 *
 * Copyright (c) 2007-2016 Lammert Bies
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
 * The source file src/crcsick.c contains routines that help in calculating the
 * CRC value that is used as dataprotection measure in communications with Sick
 * electronic devices.
 */

#include <stdlib.h>
#include "checksum.h"

/*
 * uint16_t crc_sick( const unsigned char *input_str, size_t num_bytes );
 *
 * The function crc_sick() calculates the SICK CRC value of an input string in
 * one pass.
 */

uint16_t crc_sick( const unsigned char *input_str, size_t num_bytes ) {

	uint16_t crc;
	uint16_t low_byte;
	uint16_t high_byte;
	uint16_t short_c;
	uint16_t short_p;
	const unsigned char *ptr;
	size_t a;

	crc     = CRC_START_SICK;
	ptr     = input_str;
	short_p = 0;

	if ( ptr != NULL ) for (a=0; a<num_bytes; a++) {

		short_c = 0x00FF & (uint16_t) *ptr;

		if ( crc & 0x8000 ) crc = ( crc << 1 ) ^ CRC_POLY_SICK;
		else                crc =   crc << 1;

		crc    ^= ( short_c | short_p );
		short_p = short_c << 8;

		ptr++;
	}

	low_byte  = (crc & 0xFF00) >> 8;
	high_byte = (crc & 0x00FF) << 8;
	crc       = low_byte | high_byte;

	return crc;

}  /* crc_sick */

/*
 * uint16_t update_crc_sick( uint16_t crc, unsigned char c, unsigned char prev_byte );
 *
 * The function update_crc_sick() calculates a new CRC-SICK value based on the
 * previous value of the CRC and the next byte of the data to be checked.
 */

uint16_t update_crc_sick( uint16_t crc, unsigned char c, unsigned char prev_byte ) {

	uint16_t short_c;
	uint16_t short_p;

	short_c  =   0x00FF & (uint16_t) c;
	short_p  = ( 0x00FF & (uint16_t) prev_byte ) << 8;

	if ( crc & 0x8000 ) crc = ( crc << 1 ) ^ CRC_POLY_SICK;
	else                crc =   crc << 1;

	crc ^= ( short_c | short_p );

	return crc;

}  /* update_crc_sick */
