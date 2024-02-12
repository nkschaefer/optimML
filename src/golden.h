#ifndef _OPTIMML_GOLDEN_H
#define _OPTIMML_GOLDEN_H
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

typedef std::function< double( double,
    std::map<std::string, double >&,
    std::map<std::string, int >& ) > golden_func;

class golden_solver{
    
    private:
        golden_func func;
        bool has_prior;
        golden_func prior;

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
        
        // For prior: map parameter name -> double
        std::map<std::string, double> params_prior_double;
        // For prior: map parameter name -> int
        std::map<std::string, int> params_prior_int;
        
        // Optional weights on individual observations
        std::vector<double> weights;
        
        // Maximum iterations
        int maxiter;

        // Delta threshold
        double delta_thresh; 
        
        double eval(double x);
    
    public:

        golden_solver(golden_func);

        void add_prior(golden_func f);

        bool add_data(std::string name, std::vector<double>& data);
        bool add_data(std::string name, std::vector<int>& data);
        bool add_data_fixed(std::string name, double data);
        bool add_data_fixed(std::string name, int data);
        bool add_weights(std::vector<double>& weights);
        bool add_prior_param(std::string name, double data);
        bool add_prior_param(std::string name, int data);
        
        void set_delta(double delt);
        void set_maxiter(int i);
        
        // The function evaluated at its maximum/minimum
        double y;

        // Debugging: print values over given range
        void print(double, double, double);

        // Find a maximum (or minimum) in the given interval
        double solve(double lower, double upper, bool max);
};

#endif
