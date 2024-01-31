SHELL=bash
COMP=g++
CCOMP=gcc
PREFIX ?=/usr/local
FLAGS=-std=c++11 --std=gnu++11 -fPIC
IFLAGS=-I$(PREFIX)/include
LFLAGS=-L$(PREFIX)/lib

all: lib/liboptimml.so lib/liboptimml.a

lib/liboptimml.so: build/eig.o build/nnls.o build/lstsq.o build/functions.o build/brent.o build/multivar.o build/mixcomp.o
	$(CCOMP) $(IFLAGS) $(LFLAGS) -shared -o lib/liboptimml.so build/eig.o build/nnls.o build/lstsq.o build/functions.o build/brent.o build/multivar.o build/mixcomp.o -lstdc++ -llapack

lib/liboptimml.a: build/eig.o build/nnls.o build/lstsq.o build/functions.o build/brent.o build/multivar.o build/mixcomp.o
	ar rcs lib/liboptimml.a build/eig.o build/nnls.o build/lstsq.o build/functions.o build/brent.o build/multivar.o build/mixcomp.o

build/brent.o: src/brent.cpp src/brent.h src/functions.h
	$(COMP) $(FLAGS) $(IFLAGS) -c src/brent.cpp -o build/brent.o

build/multivar.o: src/multivar.cpp src/multivar.h src/functions.h src/lstsq.h src/eig.h
	$(COMP) $(FLAGS) $(IFLAGS) -c src/multivar.cpp -o build/multivar.o 

build/mixcomp.o: src/mixcomp.cpp src/mixcomp.h src/multivar.h
	$(COMP) $(FLAGS) $(IFLAGS) -c src/mixcomp.cpp -o build/mixcomp.o

build/functions.o: src/functions.cpp src/functions.h
	$(COMP) $(FLAGS) $(IFLAGS) -c src/functions.cpp -o build/functions.o

build/nnls.o: src/nnls/nnls.c src/nnls/nnls.h
	$(CCOMP) $(CFLAGS) -c src/nnls/nnls.c -o build/nnls.o

build/lstsq.o: src/lstsq.cpp src/lstsq.h
	$(COMP) $(FLAGS) $(IFLAGS) -c src/lstsq.cpp -o build/lstsq.o

build/eig.o: src/eig.cpp src/eig.h
	$(COMP) $(FLAGS) $(IFLAGS) -c src/eig.cpp -o build/eig.o

clean:
	rm build/*.o
	rm lib/*.so
	rm lib/*.a

install: | $(PREFIX)/lib $(PREFIX)/include/optimML
	cp lib/*.so $(PREFIX)/lib
	cp lib/*.a $(PREFIX)/lib
	cp src/*.h $(PREFIX)/include/optimML

$(PREFIX)/lib:
	mkdir -p $(PREFIX)/lib

$(PREFIX)/include/optimML:
	mkdir -p $(PREFIX)/include/optimML
