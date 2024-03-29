#ifndef _OPTIMML_MULTIVAR_SYS_H
#define _OPTIMML_MULTIVAR_SYS_H
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
#include <unordered_map>
#include <set>
#include <cstdlib>
#include <utility>
#include <math.h>
#include "solver.h"
#include "multivar_ml.h"

namespace optimML{
    
    // Function to be used in solving a system of equations. 
    typedef std::function< double ( const std::vector<double>& ) > multivar_sys_func;
    // Derivative of above
    typedef std::function< void ( const std::vector<double>&, std::vector<double>& ) > multivar_sys_func_d;
   
    // Solves a multivariate system of equations.
    
    // Uses BFGS to minimize the sum of squared differences between equation evaluations
    // and true RHS values.
    
    // Works by wrapping multivar_ml_solver and using it to minimize the sum of squared
    // differences instead of to maximize log likelihood.

    class multivar_sys_solver{
        
        private:
            std::vector<double> params; 
            multivar_ml_solver solver;
            
            std::map<int, multivar_sys_func> equations1;
            std::map<int, multivar_sys_func_d> equations1_deriv;
            std::map<int, multivar_func> equations2;
            std::map<int, multivar_func_d> equations2_deriv;
            
            std::map<std::string, std::vector<double> > data_d;
            std::map<std::string, std::vector<int> > data_i;
            
            std::vector<double> weights;

            int n_equations;
            
            std::vector<bool> trans_log;
            std::vector<bool> trans_logit;

            // Store right-hand side of each equation     
            std::vector<double> rhs;

            // Store results of function evaluations 
            std::vector<double> y;
            
            // For one equation at a time - store its derivative wrt
            // all independent variables (determined by external function)
            std::vector<double> dyi_dx;

            double eval_sum_sq(const std::vector<double>& params,
                const std::map<std::string, double>& params_d,
                const std::map<std::string, int>& params_i);

            void eval_dsum_sq(const std::vector<double>& params,
                const std::map<std::string, double>& params_d,
                const std::map<std::string, int>& params_i,
                std::vector<double>& results);

        public:
            
            // Initialize with initial guesses of parameters
            multivar_sys_solver(std::vector<double> params_init);

            // Parameter values at optimum
            std::vector<double> results;

            // Residual sum of squares
            double rss;
                        
            bool add_equation(multivar_sys_func func, 
                multivar_sys_func_d func_deriv, 
                double rhs);
            
            // Include weight
            bool add_equation(multivar_sys_func func,
                multivar_sys_func_d func_deriv,
                double rhs,
                double weight);

            bool add_equation(multivar_func func,
                multivar_func_d func_deriv,
                double rhs);
            
            // Include weight
            bool add_equation(multivar_func func,
                multivar_func_d func_deriv,
                double rhs,
                double weight);

            // Add data key/val to go with most recently-added equation
            bool add_data(std::string name, double dat);
            bool add_data(std::string name, int dat);
            
            // Add data key/val to equation by equation index
            bool set_data(int idx, std::string name, double dat);
            bool set_data(int idx, std::string name, int dat);

            bool constrain_pos(int idx);
            bool constrain_01(int idx);
            
            bool solve();

    };

}

#endif
