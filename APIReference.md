#Libcrc API Reference

Libcrc is a library to calculate various checksums of data blobs. The library is written in C and can be compiled with any modern C compiler. The API to the library is described in this document.

## Constants

* [CRC start values](doc/crc_start.md)
* [CRC polynomials](doc/crc_poly.md)

## Functions

* [`checksum_NMEA( input_str, result );`](doc/checksum_nmea.md)
* [`crc_8( input_str, num_bytes );`](doc/crc_8.md)
* [`crc_16( input_str, num_bytes );`](doc/crc_16.md)
* [`crc_32( input_str, num_bytes );`](doc/crc_32.md)
* [`crc_64_ecma( input_str, num_bytes );`](doc/crc_64_ecma.md)
* [`crc_64_we( input_str, num_bytes );`](doc/crc_64_we.md)
* [`crc_ccitt_1d0f( input_str, num_bytes );`](doc/crc_ccitt_1d0f.md)
* [`crc_ccitt_ffff( input_str, num_bytes );`](doc/crc_ccitt_ffff.md)
* [`crc_dnp( input_str, num_bytes );`](doc/crc_dnp.md)
* [`crc_kermit( input_str, num_bytes );`](doc/crc_kermit.md)
* [`crc_modbus( input_str, num_bytes );`](doc/crc_modbus.md)
* [`crc_sick( input_str, num_bytes );`](doc/crc_sick.md)
* [`crc_xmodem( input_str, num_bytes );`](doc/crc_xmodem.md)
* [`update_crc_8( crc, c );`](doc/update_crc_8.md)
* [`update_crc_16( crc, c );`](doc/update_crc_16.md)
* [`update_crc_32( crc, c );`](doc/update_crc_32.md)
* [`update_crc_64( crc, c );`](doc/update_crc_64.md)
* [`update_crc_ccitt( crc, c );`](doc/update_crc_ccitt.md)
* [`update_crc_dnp( crc, c );`](doc/update_crc_dnp.md)
* [`update_crc_kermit( crc, c );`](doc/update_crc_kermit.md)
* [`update_crc_sick( crc, c, prev_byte );`](doc/update_crc_sick.md)
