# Libcrc - Multi platform MIT licensed CRC library in C

Libcrc is a multi platform CRC library which has been under development since
1999. The original version of the source code has been available on [www.lammertbies.nl](https://www.lammertbies.nl/)
since that time. Recently that code was merged with previously unpublished
developments and that library is now available at [Github](https://github.com/lammertb/libcrc/).
An online version of the CRC calculation routines is also available at [www.lammertbies.nl](https://www.lammertbies.nl/comm/info/crc-calculation.html).

The CRC library with API reference also has a new home at [www.libcrc.org](http://libcrc.org). An
online version of the API documentation is available on this website.

## License

The original version of the source code available on www.lammertbies.nl had no license attached. The
repackaged version on Github is licensed with the MIT license to make the library
useful in open and closed source products independent of their licensing scheme.

## Differences between versions

It is safe to say that in general you should use the latest stable version of the library available
from the [releases page](https://github.com/lammertb/libcrc/releases). In some circumstances it may be
necessary to use an older version or to use the development version instead. The differences between
these versions can be found in the [Changelog](CHANGELOG.md) document which is part of the distribution.

Please note that the development version is a snapshot of the ongoing development of the library and that
that version may not be stable and reliable enough for production environments.

## Platforms

Since the first version, the source code has been compiled and used on many platforms ranging from 8 bit micro controllers
to 64 bit multi core servers. Most platform dependent issues should therefore have been ironed out. Currently
the code is developed and maintained mainly in 32 bit and 64 bit environments. New versions of the code are
regularly compiled and checked on the systems mentioned in the following lists.

### 32 bit development environments
|Operating System|Compiler|
| :--- | :--- |
|Centos 6.8|gcc 4.4.7|
|Debian 8.6|gcc 4.9.2|
|FreeBSD 10.3|clang 3.4.1|
|Raspbian|gcc 4.8|
|Windows 7|Visual Studio 2015|

### 64 bit development environments
|Operating system|Compiler|
| :--- | :--- |
|Centos 6.8|gcc 4.4.7|
|Centos 7.2.1511|gcc 4.8.5|
|Debian 8.6|gcc 4.9.2|
|FreeBSD 10.3|clang 3.4.1|
|OS X El Capitan 10.11.6|Apple LLVM 8.0.0|
|Windows 7|Visual Studio 2015|
