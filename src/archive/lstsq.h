#ifndef _LSTSQ_H
#define _LSTSQ_H

#include <string>
#include <algorithm>
#include <vector>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <functional>
#include <cstdlib>
#include <utility>
#include <math.h>
#include <lapack.h>

bool lstsq(std::vector<std::vector<double> >& A,
    std::vector<double>& b,
    std::vector<double>& x);

bool lstsq(std::vector<std::vector<double> >& A,
    std::vector<double>& b,
    std::vector<double>& w,
    std::vector<double>& x);

bool nn_lstsq(std::vector<std::vector<double> >& A,
    std::vector<double>& b,
    std::vector<double>& x);

bool nn_lstsq(std::vector<std::vector<double> >& A,
    std::vector<double>& b,
    std::vector<double>& w,
    std::vector<double>& x);
 
#endif
