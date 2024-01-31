# optimML
A fast and flexible C++ library for numeric optimization of complex log likelihood functions, including mixture components that must sum to 1

## Requirements
LAPACK needs to be available, with the lapack.h C header file somewhere the compiler can find. [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) works for this. To save headaches, though it's available as a [conda package](https://anaconda.org/anaconda/openblas). 
