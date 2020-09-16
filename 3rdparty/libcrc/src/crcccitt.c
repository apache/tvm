/*
 * Library: libcrc
 * File:    src/crcccitt.c
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
 * The module src/crcccitt.c contains routines which are used to calculate the
 * CCITT CRC values of a string of bytes.
 */

#include <stdbool.h>
#include <stdlib.h>
#include "checksum.h"

static uint16_t		crc_ccitt_generic( const unsigned char *input_str, size_t num_bytes, uint16_t start_value );
static void             init_crcccitt_tab( void );

static bool             crc_tabccitt_init       = false;
static uint16_t         crc_tabccitt[256];

/*
 * uint16_t crc_xmodem( const unsigned char *input_str, size_t num_bytes );
 *
 * The function crc_xmodem() performs a one-pass calculation of an X-Modem CRC
 * for a byte string that has been passed as a parameter.
 */

uint16_t crc_xmodem( const unsigned char *input_str, size_t num_bytes ) {

	return crc_ccitt_generic( input_str, num_bytes, CRC_START_XMODEM );

}  /* crc_xmodem */

/*
 * uint16_t crc_ccitt_1d0f( const unsigned char *input_str, size_t num_bytes );
 *
 * The function crc_ccitt_1d0f() performs a one-pass calculation of the CCITT
 * CRC for a byte string that has been passed as a parameter. The initial value
 * 0x1d0f is used for the CRC.
 */

uint16_t crc_ccitt_1d0f( const unsigned char *input_str, size_t num_bytes ) {

	return crc_ccitt_generic( input_str, num_bytes, CRC_START_CCITT_1D0F );

}  /* crc_ccitt_1d0f */

/*
 * uint16_t crc_ccitt_ffff( const unsigned char *input_str, size_t num_bytes );
 *
 * The function crc_ccitt_ffff() performs a one-pass calculation of the CCITT
 * CRC for a byte string that has been passed as a parameter. The initial value
 * 0xffff is used for the CRC.
 */

uint16_t crc_ccitt_ffff( const unsigned char *input_str, size_t num_bytes ) {

	return crc_ccitt_generic( input_str, num_bytes, CRC_START_CCITT_FFFF );

}  /* crc_ccitt_ffff */

/*
 * static uint16_t crc_ccitt_generic( const unsigned char *input_str, size_t num_bytes, uint16_t start_value );
 *
 * The function crc_ccitt_generic() is a generic implementation of the CCITT
 * algorithm for a one-pass calculation of the CRC for a byte string. The
 * function accepts an initial start value for the crc.
 */

static uint16_t crc_ccitt_generic( const unsigned char *input_str, size_t num_bytes, uint16_t start_value ) {

	uint16_t crc;
	const unsigned char *ptr;
	size_t a;

	if ( ! crc_tabccitt_init ) init_crcccitt_tab();

	crc = start_value;
	ptr = input_str;

	if ( ptr != NULL ) for (a=0; a<num_bytes; a++) {

		crc = (crc << 8) ^ crc_tabccitt[ ((crc >> 8) ^ (uint16_t) *ptr++) & 0x00FF ];
	}

	return crc;

}  /* crc_ccitt_generic */

/*
 * uint16_t update_crc_ccitt( uint16_t crc, unsigned char c );
 *
 * The function update_crc_ccitt() calculates a new CRC-CCITT value based on
 * the previous value of the CRC and the next byte of the data to be checked.
 */

uint16_t update_crc_ccitt( uint16_t crc, unsigned char c ) {

	if ( ! crc_tabccitt_init ) init_crcccitt_tab();

	return (crc << 8) ^ crc_tabccitt[ ((crc >> 8) ^ (uint16_t) c) & 0x00FF ];

}  /* update_crc_ccitt */

/*
 * static void init_crcccitt_tab( void );
 *
 * For optimal performance, the routine to calculate the CRC-CCITT uses a
 * lookup table with pre-compiled values that can be directly applied in the
 * XOR action. This table is created at the first call of the function by the
 * init_crcccitt_tab() routine.
 */

static void init_crcccitt_tab( void ) {

	uint16_t i;
	uint16_t j;
	uint16_t crc;
	uint16_t c;

	for (i=0; i<256; i++) {

		crc = 0;
		c   = i << 8;

		for (j=0; j<8; j++) {

			if ( (crc ^ c) & 0x8000 ) crc = ( crc << 1 ) ^ CRC_POLY_CCITT;
			else                      crc =   crc << 1;

			c = c << 1;
		}

		crc_tabccitt[i] = crc;
	}

	crc_tabccitt_init = true;

}  /* init_crcccitt_tab */
