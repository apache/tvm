# Libcrc API Reference

### `crc_sick( input_str, num_bytes );`

### Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
|**`input_str`**|`const unsigned char *`|The input byte buffer for which the CRC must be calculated|
|**`num_bytes`**|`size_t`|The number of characters in the input buffer|

### Return Value

| Type | Description |
| :--- | :--- |
|`uint16_t`|The resulting CRC value|

### Description

The function `crc_sick()` calculates a 16 bit CRC value of an input byte buffer based on the CRC calculation algorithm used by Sensor manufacturer Sick in some of their sensors.  The buffer length is provided as a parameter and the resulting CRC is returned as a return value by the function. The size of the buffer is limited to `SIZE_MAX`.

The CRC calculation used by Sick differs from generic CRC calculation algorithms in that every input byte is passed twice through the algorithm.

### See Also

* [`update_crc_sick();`](update_crc_sick.md)
