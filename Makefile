SHELL=bash
COMP=g++
CCOMP=gcc
PREFIX ?=/usr/local
FLAGS=-std=c++11 --std=gnu++11 -fPIC -g
IFLAGS=-I$(PREFIX)/include
LFLAGS=-L$(PREFIX)/lib
ifeq ($(findstring cellbouncer, ${CONDA_PREFIX}), cellbouncer)
	IFLAGS += -I${CONDA_PREFIX}/include
	LFLAGS += -L${CONDA_PREFIX}/lib
endif
STLBFGS_O=build/linesearch.o build/stlbfgs.o
# Try to add OpenMP support if available
OMPFLAG =
OMP_TEST_CODE = src/omp_test.c
all: check_openmp lib/liboptimml.so lib/liboptimml.a

check_openmp:
	@echo "#include <omp.h>" > $(OMP_TEST_CODE)
	@echo "int main() { return 0; }" >> $(OMP_TEST_CODE)
	@if $(CC) $(OMP_TEST_CODE) -fopenmp -o /dev/null >/dev/null 2>&1; then \
		echo "OpenMP supported, enabling -fopenmp"; \
		OMPFLAG=-fopenmp; \
	else \
		echo "OpenMP not supported, disabling OpenMP"; \
	fi
	@rm -f $(OMP_TEST_CODE)

lib/liboptimml.so: build/functions.o build/solver.o build/univar.o build/multivar.o build/brent.o build/multivar_ml.o build/mixcomp.o build/multivar_sys.o $(STLBFGS_O)
	$(CCOMP) $(IFLAGS) $(LFLAGS) -shared -o lib/liboptimml.so build/functions.o build/solver.o build/univar.o build/multivar.o build/brent.o build/multivar_ml.o build/mixcomp.o build/multivar_sys.o $(STLBFGS_O) -lstdc++

lib/liboptimml.a: build/functions.o build/solver.o build/univar.o build/multivar.o build/brent.o build/multivar_ml.o build/mixcomp.o build/multivar_sys.o $(STLBFGS_O)
	ar rcs lib/liboptimml.a build/functions.o build/solver.o build/univar.o build/multivar.o build/brent.o build/multivar_ml.o build/mixcomp.o build/multivar_sys.o $(STLBFGS_O)

build/solver.o: src/solver.cpp src/solver.h src/functions.h
	$(COMP) $(FLAGS) $(IFLAGS) -c src/solver.cpp -o build/solver.o

build/univar.o: src/univar.cpp src/univar.h src/solver.h src/functions.h
	$(COMP) $(FLAGS) $(IFLAGS) -c src/univar.cpp -o build/univar.o

build/multivar.o: src/multivar.cpp src/multivar.h src/solver.h src/functions.h
	$(COMP) $(FLAGS) $(IFLAGS) -c src/multivar.cpp -o build/multivar.o

build/brent.o: src/brent.cpp src/brent.h src/univar.h
	$(COMP) $(FLAGS) $(IFLAGS) -c src/brent.cpp -o build/brent.o

build/multivar_ml.o: src/multivar_ml.cpp src/multivar.h
	$(COMP) $(FLAGS) $(IFLAGS) -c src/multivar_ml.cpp -o build/multivar_ml.o 

build/mixcomp.o: src/mixcomp.cpp src/mixcomp.h src/multivar_ml.h src/multivar.h
	$(COMP) $(FLAGS) $(IFLAGS) -c src/mixcomp.cpp -o build/mixcomp.o

build/multivar_sys.o: src/multivar_sys.cpp src/multivar_ml.h
	$(COMP) $(FLAGS) $(IFLAGS) -c src/multivar_sys.cpp -o build/multivar_sys.o

build/functions.o: src/functions.cpp src/functions.h
	$(COMP) $(FLAGS) $(IFLAGS) -c src/functions.cpp -o build/functions.o

build/linesearch.o: src/stlbfgs/linesearch.h src/stlbfgs/linesearch.cpp
	$(COMP) $(FLAGS) $(IFLAGS) -c src/stlbfgs/linesearch.cpp -o build/linesearch.o

build/stlbfgs.o: check_openmp src/stlbfgs/stlbfgs.h src/stlbfgs/stlbfgs.cpp
	$(COMP) $(FLAGS) $(IFLAGS) $(OMPFLAG) -c src/stlbfgs/stlbfgs.cpp -o build/stlbfgs.o

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
