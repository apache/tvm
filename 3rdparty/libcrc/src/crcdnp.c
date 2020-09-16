/*
 * Library: libcrc
 * File:    src/crcdnp.c
 * Author:  Lammert Bies
 *
 * This file is licensed under the MIT License as stated below
 *
 * Copyright (c) 2005-2016 Lammert Bies
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
 * The source file src/crcdnp.c contains routines which are used to calculate
 * the CRC value in DNP messages.
 */



#include <stdbool.h>
#include <stdlib.h>
#include "checksum.h"

static void             init_crcdnp_tab( void );

static bool             crc_tabdnp_init         = false;
static uint16_t         crc_tabdnp[256];

/*
 * uint16_t crc_dnp( const unsigned char* input_str, size_t num_bytes );
 *
 * The function crc_dnp() calculates the DNP CRC checksum of a provided byte
 * string in one pass.
 */

uint16_t crc_dnp( const unsigned char *input_str, size_t num_bytes ) {

	uint16_t crc;
	uint16_t low_byte;
	uint16_t high_byte;
	const unsigned char *ptr;
	size_t a;

	if ( ! crc_tabdnp_init ) init_crcdnp_tab();

	crc = CRC_START_DNP;
	ptr = input_str;

	if ( ptr != NULL ) for (a=0; a<num_bytes; a++) {

		crc = (crc >> 8) ^ crc_tabdnp[ (crc ^ (uint16_t) *ptr++) & 0x00FF ];
	}

	crc       = ~crc;
	low_byte  = (crc & 0xff00) >> 8;
	high_byte = (crc & 0x00ff) << 8;
	crc       = low_byte | high_byte;

	return crc;

}  /* crc_dnp */

/*
 * uint16_t update_crc_dnp( uint16_t crc, unsigned char c );
 *
 * The function update_crc_dnp() is called for every new byte in a row that
 * must be feeded tot the CRC-DNP routine to calculate the DNP CRC.
 */

uint16_t update_crc_dnp( uint16_t crc, unsigned char c ) {

	if ( ! crc_tabdnp_init ) init_crcdnp_tab();

	return (crc >> 8) ^ crc_tabdnp[ (crc ^ (uint16_t) c) & 0x00FF ];

}  /* update_crc_dnp */

/*
 * static void init_crcdnp_tab( void );
 *
 * For better performance, the DNP CRC calculation uses a precompiled list with
 * bit patterns that are used in the XOR operation in the main routine. This
 * table is calculated once at the start of the program by the
 * init_crcdnp_tab() routine.
 */

static void init_crcdnp_tab( void ) {

	int i;
	int j;
	uint16_t crc;
	uint16_t c;

	for (i=0; i<256; i++) {

		crc = 0;
		c   = (uint16_t) i;

		for (j=0; j<8; j++) {

			if ( (crc ^ c) & 0x0001 ) crc = ( crc >> 1 ) ^ CRC_POLY_DNP;
			else                      crc =   crc >> 1;

			c = c >> 1;
		}

		crc_tabdnp[i] = crc;
	}

	crc_tabdnp_init = true;

}  /* init_crcdnp_tab */
