# Libcrc API Reference

### `crc_8( input_str, num_bytes );`

### Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
|**`input_str`**|`const unsigned char *`|The input byte buffer for which the CRC must be calculated|
|**`num_bytes`**|`size_t`|The number of characters in the input buffer|

### Return Value

| Type | Description |
| :--- | :--- |
|`uint8_t`|The resulting CRC value|

### Description

The function `crc_8()` calculates a 8 bit CRC value of an input byte buffer based on the CRC algorithm as it is used in Sensirion SHTxx temperature and humidity sensors.  The buffer length is provided as a parameter and the resulting CRC is returned as a return value by the function. The size of the buffer is limited to `SIZE_MAX`.

### See Also

* [`update_crc_8();`](update_crc_8.md)
