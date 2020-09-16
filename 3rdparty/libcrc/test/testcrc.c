/*
 * Library: libcrc
 * File:    test/testcrc.c
 * Author:  Lammert Bies
 *
 * This file is licensed under the MIT License as stated below
 *
 * License
 * -------
 * Copyright (c) 2008-2016 Lammert Bies
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
 * The source file test/testcrc.c contains routines which test if the
 * implementation of the CRC routines from the libcrc library went without
 * problems.
 */

#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include "testall.h"

#include "../include/checksum.h"

						/************************************************/
struct chk_tp {					/*						*/
	const char *	input;			/* The input string to be checked		*/
	uint8_t		crc8;			/* The  8 bit wide CRC8 of the input string	*/
	uint16_t	crc16;			/* The 16 bit wide CRC16 of the input string	*/
	uint32_t	crc32;			/* The 32 bit wide CRC32 of the input string	*/
	uint64_t	crc64_ecma;		/* The 64 bit wide CRC64-ECMA of the input	*/
	uint64_t	crc64_we;		/* The 64 bit wide CRC64-WE of the input string	*/
	uint16_t	crcdnp;			/* The 16 bit wide DNP CRC of the string	*/
	uint16_t	crcmodbus;		/* The 16 bit wide Modbus CRC of the string	*/
	uint16_t	crcsick;		/* The 16 bit wide Sick CRC of the string	*/
	uint16_t	crcxmodem;		/* The 16 bit wide XModem CRC of the string	*/
	uint16_t	crc1d0f;		/* The 16 bit wide CCITT CRC with 1D0F start	*/
	uint16_t	crcffff;		/* The 16 bit wide CCITT CRC with FFFF start	*/
	uint16_t	crckermit;		/* The 16 bit wide CRC Kermit of the string	*/
};						/*						*/
						/************************************************/

static struct chk_tp checks[] = {
	{ "123456789",    0xA2, 0xBB3D, 0xCBF43926ul, 0x6C40DF5F0B497347ull, 0x62EC59E3F1A4F00Aull, 0x82EA, 0x4B37, 0x56A6, 0x31C3, 0xE5CC, 0x29B1, 0x8921 },
	{ "Lammert Bies", 0xA5, 0xB638, 0x43C04CA6ul, 0xF806F4F5C0F3257Cull, 0xFE25A9F50630F789ull, 0x4583, 0xB45C, 0x1108, 0xCEC8, 0x67A2, 0x4A31, 0xF80D },
	{ "",             0x00, 0x0000, 0x00000000ul, 0x0000000000000000ull, 0x0000000000000000ull, 0xFFFF, 0xFFFF, 0x0000, 0x0000, 0x1D0F, 0xFFFF, 0x0000 },
	{ " ",            0x86, 0xD801, 0xE96CCF45ul, 0xCC7AF1FF21C30BDEull, 0x568617D9EF46BE26ull, 0x50D6, 0x98BE, 0x2000, 0x2462, 0xE8FE, 0xC592, 0x0221 },
	{ NULL,           0,    0,      0,            0,                     0,                     0,      0,      0,      0,      0,      0,      0      }
};

/*
 * int test_crc( bool verbose );
 *
 * The function test_crc_32() tests the functionality of the implementation of
 * the CRC library functions on a specific platform.
 */

int test_crc( bool verbose ) {

	int a;
	int errors;
	int len;
	const unsigned char *ptr;
	uint8_t crc8;
	uint16_t crc16;
	uint16_t crcdnp;
	uint16_t crcmodbus;
	uint16_t crcsick;
	uint16_t crcxmodem;
	uint16_t crc1d0f;
	uint16_t crcffff;
	uint16_t crckermit;
	uint32_t crc32;
	uint64_t crc64_ecma;
	uint64_t crc64_we;

	errors = 0;

	printf( "Testing CRC routines: " );

	a = 0;
	while ( checks[a].input != NULL ) {

		ptr = (const unsigned char *) checks[a].input;
		len = strlen( checks[a].input );

		crc8       = crc_8(          ptr, len );
		crc16      = crc_16(         ptr, len );
		crc32      = crc_32(         ptr, len );
		crc64_ecma = crc_64_ecma(    ptr, len );
		crc64_we   = crc_64_we(      ptr, len );
		crcdnp     = crc_dnp(        ptr, len );
		crcmodbus  = crc_modbus(     ptr, len );
		crcsick    = crc_sick(       ptr, len );
		crcxmodem  = crc_xmodem(     ptr, len );
		crc1d0f    = crc_ccitt_1d0f( ptr, len );
		crcffff    = crc_ccitt_ffff( ptr, len );
		crckermit  = crc_kermit(     ptr, len );

		if ( crc8 != checks[a].crc8 ) {

			if ( verbose ) printf( "\n    FAIL: CRC8 \"%s\" return 0x%02" PRIX8 ", not 0x%02" PRIX8
							, checks[a].input, crc8, checks[a].crc8 );
			errors++;
		}

		if ( crc16 != checks[a].crc16 ) {

			if ( verbose ) printf( "\n    FAIL: CRC16 \"%s\" return 0x%04" PRIX16 ", not 0x%04" PRIX16
							, checks[a].input, crc16, checks[a].crc16 );
			errors++;
		}

		if ( crc32 != checks[a].crc32 ) {

			if ( verbose ) printf( "\n    FAIL: CRC32 \"%s\" returns 0x%08" PRIX32 ", not 0x%08" PRIX32
							, checks[a].input, crc32, checks[a].crc32 );
			errors++;
		}

		if ( crc64_ecma != checks[a].crc64_ecma ) {

			if ( verbose ) printf( "\n    FAIL: CRC64 ECMA \"%s\" returns 0x%016" PRIX64 ", not 0x%016" PRIX64
							, checks[a].input, crc64_ecma, checks[a].crc64_ecma );
			errors++;
		}

		if ( crc64_we != checks[a].crc64_we ) {

			if ( verbose ) printf( "\n    FAIL: CRC64 WE \"%s\" returns 0x%016" PRIX64 ", not 0x%016" PRIX64
							, checks[a].input, crc64_we, checks[a].crc64_we );
			errors++;
		}

		if ( crcdnp != checks[a].crcdnp ) {

			if ( verbose ) printf( "\n    FAIL CRC DNP \"%s\" returns 0x%04" PRIX16 ", not 0x%04" PRIX16
							, checks[a].input, crcdnp, checks[a].crcdnp );
			errors++;
		}

		if ( crcmodbus != checks[a].crcmodbus ) {

			if ( verbose ) printf( "\n    FAIL CRC Modbus \"%s\" returns 0x%04" PRIX16 ", not 0x%04X" PRIX16
							, checks[a].input, crcmodbus, checks[a].crcmodbus );
			errors++;
		}

		if ( crcsick != checks[a].crcsick ) {

			if ( verbose ) printf( "\n    FAIL CRC Sick \"%s\" returns 0x%04" PRIX16 ", not 0x%04" PRIX16
							, checks[a].input, crcsick, checks[a].crcsick );
			errors++;
		}

		if ( crcxmodem != checks[a].crcxmodem ) {

			if ( verbose ) printf( "\n    FAIL CRC Xmodem \"%s\" returns 0x%04" PRIX16 ", not 0x%04" PRIX16
							, checks[a].input, crcxmodem, checks[a].crcxmodem );
			errors++;
		}

		if ( crc1d0f != checks[a].crc1d0f ) {

			if ( verbose ) printf( "\n    FAIL CRC CCITT 1d0f \"%s\" returns 0x%04" PRIX16 ", not 0x%04" PRIX16
							, checks[a].input, crc1d0f, checks[a].crc1d0f );
			errors++;
		}

		if ( crcffff != checks[a].crcffff ) {

			if ( verbose ) printf( "\n    FAIL CRC CCITT ffff \"%s\" returns 0x%04" PRIX16 ", not 0x%04" PRIX16
							, checks[a].input, crcffff, checks[a].crcffff );
			errors++;
		}

		if ( crckermit != checks[a].crckermit ) {

			if ( verbose ) printf( "\n    FAIL CRC Kermit \"%s\" returns 0x%04" PRIX16 ", not 0x%04" PRIX16
							, checks[a].input, crckermit, checks[a].crckermit );
			errors++;
		}

		a++;
	}

	if ( errors == 0 ) printf( "OK\n" );
	else {

		if ( verbose ) printf( "\n    " );
		printf( "FAILED %d checks\n", errors );
	}

	return errors;

}  /* test_crc */
