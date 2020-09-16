/*
 * Library: libcrc
 * File:    test/testall.c
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
 * The source file test/testall.c contains routines to check the proper
 * functionality of routines in the libcrc library on the current platform.
 */

#include <stdio.h>
#include <stdlib.h>
#include "testall.h"

/*
 * int main( void );
 *
 * Testall is a commandline utility that tests the functionality of the libcrc
 * routines on a the current platform. The result is printed to stdout. The
 * program returns an integer value which can be catched by a shell script. The
 * value is equal to the number of errors encountered.
 */

int main( void ) {

	int problems;

	printf( "\n" );

	problems  = 0;
	problems += test_crc( true );
	problems += test_checksum_NMEA( true );

	printf( "\n" );

	if ( problems == 0 ) printf( "**** All tests succeeded\n\n" );
	else                 printf( "**** A TOTAL OF %d TESTS FAILED, PLEASE CORRECT THE DETECTED PROBLEMS ****\n\n", problems );

	return problems;

}  /* main (libcrc test) */
