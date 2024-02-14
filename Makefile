SHELL=bash
COMP=g++
CCOMP=gcc
PREFIX ?=/usr/local
FLAGS=-std=c++11 --std=gnu++11 -fPIC
IFLAGS=-I$(PREFIX)/include
LFLAGS=-L$(PREFIX)/lib
STLBFGS_O=build/linesearch.o build/stlbfgs.o
all: lib/liboptimml.so lib/liboptimml.a

lib/liboptimml.so: build/functions.o build/solver.o build/univar.o build/multivar.o build/golden.o build/brent.o build/multivar_ml.o build/mixcomp.o $(STLBFGS_O)
	$(CCOMP) $(IFLAGS) $(LFLAGS) -shared -o lib/liboptimml.so build/functions.o build/solver.o build/univar.o build/multivar.o build/golden.o build/brent.o build/multivar_ml.o build/mixcomp.o $(STLBFGS_O) -lstdc++

lib/liboptimml.a: build/functions.o build/solver.o build/univar.o build/multivar.o build/golden.o build/brent.o build/multivar_ml.o build/mixcomp.o $(STLBFGS_O)
	ar rcs lib/liboptimml.a build/functions.o build/solver.o build/univar.o build/multivar.o build/golden.o build/brent.o build/multivar_ml.o build/mixcomp.o $(STLBFGS_O)

build/solver.o: src/solver.cpp src/solver.h src/functions.h
	$(COMP) $(FLAGS) $(IFLAGS) -c src/solver.cpp -o build/solver.o

build/univar.o: src/univar.cpp src/univar.h src/solver.h src/functions.h
	$(COMP) $(FLAGS) $(IFLAGS) -c src/univar.cpp -o build/univar.o

build/multivar.o: src/multivar.cpp src/multivar.h src/solver.h src/functions.h
	$(COMP) $(FLAGS) $(IFLAGS) -c src/multivar.cpp -o build/multivar.o

build/golden.o: src/golden.cpp src/golden.h src/univar.h
	$(COMP) $(FLAGS) $(IFLAGS) -c src/golden.cpp -o build/golden.o

build/brent.o: src/brent.cpp src/brent.h src/univar.h
	$(COMP) $(FLAGS) $(IFLAGS) -c src/brent.cpp -o build/brent.o

build/multivar_ml.o: src/multivar_ml.cpp src/multivar.h
	$(COMP) $(FLAGS) $(IFLAGS) -c src/multivar_ml.cpp -o build/multivar_ml.o 

build/mixcomp.o: src/mixcomp.cpp src/mixcomp.h src/multivar_ml.h src/multivar.h
	$(COMP) $(FLAGS) $(IFLAGS) -c src/mixcomp.cpp -o build/mixcomp.o

build/functions.o: src/functions.cpp src/functions.h
	$(COMP) $(FLAGS) $(IFLAGS) -c src/functions.cpp -o build/functions.o

build/linesearch.o: src/stlbfgs/linesearch.h src/stlbfgs/linesearch.cpp
	$(COMP) $(FLAGS) $(IFLAGS) -c src/stlbfgs/linesearch.cpp -o build/linesearch.o

build/stlbfgs.o: src/stlbfgs/stlbfgs.h src/stlbfgs/stlbfgs.cpp
	$(COMP) $(FLAGS) $(IFLAGS) -c src/stlbfgs/stlbfgs.cpp -o build/stlbfgs.o

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
