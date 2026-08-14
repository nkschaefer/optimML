#ifndef _OPTIMML_FUNCTIONS_H
#define _OPTIMML_FUNCTIONS_H
#include <string>
#include <algorithm>
#include <vector>
#include <iterator>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <utility>
#include <math.h>
#include <functional>

// ===== useful functions =====

double logit(double x);
double expit(double x);
double poisll(double x, double l);
void bfgs(std::vector<double>& params,
    std::function<void(const std::vector<double>&, double&, std::vector<double>&)> func);

#endif
