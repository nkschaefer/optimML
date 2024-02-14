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
#include <map>
#include <set>
#include <functional>
#include <cstdlib>
#include <utility>
#include <math.h>
#include "functions.h"
#include "golden.h"

using std::cout;
using std::endl;
using namespace std;

// ----- Golden section search to find a maximum/minimum of a function
// ----- in a given interval without access to derivative information
//
// This algorithm is a very slight modification of the one presented in
// Chapter 10.3 of Numerical Recipes in C++ (Third Edition) by Press, 
// Teukolsky, Vetterling, & Flannery https://numerical.recipes/

optimML::golden_solver::golden_solver(univar_func f){
    init(f);
    max = true;
}

void optimML::golden_solver::set_min(){
    max = false;
}

void optimML::golden_solver::set_max(){
    max = true;
}

double optimML::golden_solver::solve(double lower, double upper){

    double tol = delta_thresh;

    double x0 = lower;
    double x3 = upper;

    double R = 0.61803399;
    double C = 1.0 - R;
    
    // Choose third point as midway between x0 and x3
    double x1 = (x0 + x3)/2.0;
    double x2 = x1 + C*(x3-x1);

    double f1 = eval_ll_x(x1);
    double f2 = eval_ll_x(x2);
    
    double xmax;
    int nit = 0;
    while (nit < maxiter && abs(x3-x0) > tol*(abs(x1) + abs(x2))){
        ++nit;
        if ((max && f2 > f1) || (!max && f2 < f1)){
            x0 = x1;
            x1 = x2;
            x2 = R*x2 + C*x3;   
            
            f1 = f2;
            f2 = eval_ll_x(x2);
        }
        else{
            x3 = x2;
            x2 = x1;
            x1 = R*x1 + C*x0;

            f2 = f1;
            f1 = eval_ll_x(x1);
        }
    }
    if ((max && f1 > f2) || (!max && f1 < f2)){
        this->log_likelihood = f1;
        xmax = x1;
    }
    else{
        this->log_likelihood = f2;
        xmax = x2;
    }
    
    return xmax;
}

