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

// Nelder-mead max-finding for 1D and 2D problems
double nelder_mead_1d(const std::function<double(double)>& f,
    double guess, 
    double& best,
    double step = 0.05,
    double tol = 1e-8,
    int maxit = 500);
double nelder_mead_2d(const std::function<double(double,double)>& f,
    double gx, 
    double gy,
    double& outx, 
    double& outy,
    double stepx = 0.05, 
    double stepy = 0.05,
    double tol = 1e-8, 
    int maxit = 500);

#endif
