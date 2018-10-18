cdef extern from "stdio.h":
    extern int printf(const char * format, ...)
def SayHello():
    printf("Hello world!\n")
