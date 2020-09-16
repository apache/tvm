# Libcrc API Reference

### `checksum_NMEA( input_str, result );`

### Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
|**`input_str`**|`const unsigned char *`|The NUL terminated input string for which the NMEA checksum must be calculated|
|**`result`**|`unsigned char *`|Storage buffer to the calculated NMEA checksum|

### Return Value

| Type | Description |
| :--- | :--- |
|`unsigned char *`|Pointer to the storage buffer with the NMEA checksum result|

### Description

The function `checksum_NMEA()` calculates the checksum in NMEA messages. The NMEA protocol is mainly used in marine equipment. The string may optionally be starting with a **`$`** character. This character is automatically ignored by the checksum calculation algorithm. The end of the NMEA is reached when the algorithm detects the end of the NUL terminated string, a *newline character* or a **`*`** character.

The function returns the checksum as a NUL terminated string with two hexadecimal characters stored in a caller provided buffer. For this reason the calling function must provide a buffer which can at least contain 3 bytes. If NULL is provided as either the pointer to the input string or the pointer to the result buffer the function returns NULL. Otherwise the return value is a pointer to the beginning of the result string.

### See Also

* [`crc_8();`](crc_8.md)
