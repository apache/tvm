/*
 * Library: libcrc
 * File:    src/nmea-chk.c
 * Author:  Lammert Bies
 *
 * This file is licensed under the MIT License as stated below
 *
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
 * The source file src/nmea-chk.c contains routines to calculate the checksum
 * in NMEA messages.
 */

#include <stdio.h>
#include <stdlib.h>
#include "checksum.h"

/*
 * unsigned char *checksum_NMEA( const unsigned char *input_str, unsigned char *result );
 *
 * The function checksum_NMEA() calculates the checksum of a valid NMEA string.
 * The routine does not try to validate the string itself. A leading '$' will
 * be ignored, as this character is part of the NMEA sentence, but not part of
 * the checksum calculation. The calculation stops, whenever a linefeed,
 * carriage return, '*' or end of string is scanned.
 *
 * Because there is no NMEA syntax checking involved, the function always
 * returns with succes, unless a NULL pointer is provided as parameter. The
 * return value is a pointer to the result buffer provided by the calling
 * application, or NULL in case of error.
 *
 * The result buffer must be at least three characters long. Two for the
 * checksum value and the third to store the EOS. The result buffer is not
 * filled when an error occurs.
 */

unsigned char * checksum_NMEA( const unsigned char *input_str, unsigned char *result ) {

	const unsigned char *ptr;
	unsigned char checksum;

	if ( input_str == NULL ) return NULL;
	if ( result    == NULL ) return NULL;

	checksum = 0;
	ptr      = (const unsigned char *) input_str;

	if ( *ptr == '$' ) ptr++;

	while ( *ptr  &&  *ptr != '\r'  &&  *ptr != '\n'  &&  *ptr != '*' ) checksum ^= *ptr++;

	snprintf( (char *) result, 3, "%02hhX", checksum );

	return result; 

}  /* checksum_NMEA */
