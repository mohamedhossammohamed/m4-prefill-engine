CC ?= clang
CFLAGS ?= -O3 -shared -fPIC -march=armv8.4-a+dotprod

all: libdequant.dylib

libdequant.dylib: dequant.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f libdequant.dylib

.PHONY: all clean
