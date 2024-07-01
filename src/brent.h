#ifndef _OPTIMML_BRENT_H
#define _OPTIMML_BRENT_H
#include <string>
#include <algorithm>
#include <vector>
#include <iterator>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <functional>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <cstdlib>
#include <utility>
#include <math.h>
#include "functions.h"
#include "univar.h"

// ===== Brent's method to find roots of univariate functions in defined intervals =====
// 
// Input variables can be constrained to (0, infinity) or (0, 1) using log or logit
// transformation automatically.
//
// Prior distributions on variables can also be incorporated.
//
// If the second derivative is known and supplied, a standard error of the MLE is 
// provided using the Fisher information.
//

// This class will optimize (find the root of the derivative of) a log likelihood
// function within a fixed interval using Brent's root finding method. 

// The function should be univariate (for example, you could differentiate a 
// multivariate log likelihood function wrt to the single variable of interest).

// You can optionally include a prior, and you can also constrain the variable
// to be either positive or within (0,1).

namespace optimML{
    class brent_solver: public univar{
        private:
            // minimum end of range
            double a;
            // middle point of range/solution
            double b;
            // end point of range
            double c;
            // log likelihood at point a
            double f_a;
            // log likelihood at point b
            double f_b;
            // log likelihood at point c
            double f_c;
            
            // Keep track of the size of the previous two steps taken
            double step_prev;
            double step_2ago;

            double quadfit(bool& success);
            bool golden();
            bool interpolate();
            bool interpolate_der();
            bool interpolate_root();
            
            bool no_deriv;
            bool root;
        public:
            
            void set_root();
            void set_max();

            brent_solver(univar_func ll);
            brent_solver(univar_func ll, univar_func dll);
            brent_solver(univar_func ll, univar_func dll, univar_func d2ll);
            
            double solve(double min, double max);

            double se;
            bool se_found;
    };
}

#endif
