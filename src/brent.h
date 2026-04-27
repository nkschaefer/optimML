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

// ===== Brent's method to find roots or maxima of univariate functions in defined intervals =====
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

// You can also use this class to find zeros of a univariate function.

// Supplying derivative information is helpful but not required.

// You can optionally include a prior, and you can also constrain the variable
// to be either positive or within (0,1).

namespace optimML{

    typedef std::function< double(double, std::vector<double>& ) > reconcile_fun;

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
           
            // If root finding, keep an optional "rhs" (constant, not
            // computed from data) to subtract
            double rhs;
            
            // If root finding, can also provide an arbitrary number of
            // other functions to evaluate at each data point, and then
            // a function to reconcile all of their output
            std::vector<univar_func> additional_funcs;
            std::vector<double> additional_funcs_out;
            reconcile_fun reconcile_func;
            bool reconcile_func_set;

            // Keep track of the size of the previous two steps taken
            double step_prev;
            double step_2ago;

            double quadfit(bool& success);
            double quadfit2(bool& success);

            bool golden();
            bool interpolate();
            bool interpolate_der();
            bool interpolate_root();
            
            bool no_deriv;
            bool root;
            
            bool bracket_root(int attempt_no);
            bool bracket_max(int attempt_no);
            
            // Override function evaluation to subtract RHS if we are root finding.
            double eval_ll_x(double x);

        public:
            
            void set_root();
            void set_max();
            void set_root_rhs(double r);
            void add_root_function(univar_func f);
            void set_root_reconcile_function(reconcile_fun f);

            brent_solver(univar_func ll);
            brent_solver(univar_func ll, univar_func dll);
            brent_solver(univar_func ll, univar_func dll, univar_func d2ll);
            
            double solve(double min, double max, bool attempt_bracket=false);

            double se;
            bool se_found;
    };
}

#endif
