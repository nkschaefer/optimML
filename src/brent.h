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

// Generalized function for use with Brent's method for root finding. 
// Returns either log likelihood or its first or second derivative.
// 
// Arguments: 
//  value at which to evaluate function
//  map of param name -> value for double params
//  map of param name -> value for int params
typedef std::function< double( double,
    std::map<std::string, double >&,
    std::map<std::string, int >& ) > brent_func;

// This class will optimize (find the root of the derivative of) a log likelihood
// function within a fixed interval using Brent's root finding method. 

// The function should be univariate (for example, you could differentiate a 
// multivariate log likelihood function wrt to the single variable of interest).

// You can optionally include a prior, and you can also constrain the variable
// to be either positive or within (0,1).

class brentSolver{
    private: 
        
        // How do we compute log likelihood?
        brent_func ll_x;
        // How do we compute derivative of log likelihood?
        brent_func dll_dx;
        // How do we compute 2nd derivative of log likelihood? (optional)
        brent_func d2ll_dx2;
        
        // Can we compute standard error (using second derivative)?
        bool has_d2ll;
        
        // Did the user store a prior distribution?
        bool has_prior;

        // How do we compute log likelihood of prior? (optional)
        brent_func ll_x_prior;
        // How do we compute derivate of log likelihood of prior? (optional)
        brent_func dll_dx_prior;
        // How do we compute second derivative of log likelihood of prior? (optional)
        brent_func d2ll_dx2_prior;

        // How many data points are there?
        int n_data;
        
        std::vector<std::string> params_double_names;
        std::vector<double*> params_double_vals;
        
        std::vector<std::string> params_int_names;
        std::vector<int*> params_int_vals;
        
        std::map<std::string, double> param_double_cur;
        std::map<std::string, int> param_int_cur;
        
        // Pointer to location in above maps
        // Position is same as in params_double_names and params_double_vals
        // For the jth variable, ith data point:
        // name = params_double_names[j]
        // val = *params_double_vals[j][i]
        // pointer to update in map: param_double_ptr[j]

        std::vector<double*> param_double_ptr;
        std::vector<int*> param_int_ptr;

        
        // Map parameter name -> vector of doubles
        std::map<std::string, std::vector<double> > params_double;
        // Map parameter name -> vector of ints
        std::map<std::string, std::vector<int> > params_int;
        
        // For prior: map parameter name -> double
        std::map<std::string, double> params_prior_double;
        // For prior: map parameter name -> int
        std::map<std::string, int> params_prior_int;
        
        // Should the independent variable be constrained to positive numbers?
        bool trans_log;
        
        // Should the independent variable be constrained to (0,1)?
        bool trans_logit;
        
        // Maximum iterations
        int maxiter;

        // Delta threshold
        double delta_thresh;
        
        // Set defaults
        void init();
        
        // Evaluate log likelihood at a given parameter value 
        double eval_ll_x(double x);
        // Evaluate derivative of log likelihood function at given parameter value
        double eval_dll_dx(double x);
        // Evaluate second derivative of log likelihood function at given parameter value
        double eval_d2ll_dx2(double x);

    public:

        brentSolver(brent_func ll, brent_func dll);
        brentSolver(brent_func ll, brent_func dll, brent_func dll2);
        void add_prior(brent_func ll, brent_func dll);
        void add_prior(brent_func ll, brent_func dll, brent_func dll2);
        bool add_data(std::string name, std::vector<double>& data);
        bool add_data(std::string name, std::vector<int>& data);
        bool add_data_fixed(std::string name, double data);
        bool add_data_fixed(std::string name, int data);
        bool add_prior_param(std::string name, double data);
        bool add_prior_param(std::string name, int data);
        void constrain_pos();
        void constrain_01();
        void set_delta(double delt);
        void set_maxiter(int i);
        
        // Debugging, etc. - print x, log likelihood, deriv log likelihood,
        // and second deriv log likelihood (if possible) over the range
        // with the given step size
        void print(double, double, double);

        // Find a root in the given interval. 
        double solve(double lower, double upper);

        // Was a root found?
        bool root_found;
        
        // Was a standard error computed/found?
        bool se_found;

        // Result
        double root;

        // Standard error (or 0 if 2nd derivative unavailable or positive at root)
        double se;

};

#endif
