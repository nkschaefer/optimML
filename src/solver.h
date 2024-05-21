#ifndef _OPTIMML_SOLVER_H
#define _OPTIMML_SOLVER_H
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

// ===== Base class for solvers =====

namespace optimML{

    // Function for prior distributions over individual x variables
    typedef std::function< double( double,
        const std::map<std::string, double>&,
        const std::map<std::string, int>& ) > prior_func;

    class solver{
        protected:
            
            bool initialized;
            
            // Pre-set prior functions        
            static double phi(double x);
            static double dnorm(double x, double mu, double sigma);
            static double ll_prior_normal(double x, const std::map<std::string, double>& params_d,
                const std::map<std::string, int>& params_i);
            static double dll_prior_normal(double x, const std::map<std::string, double>& params_d,
                const std::map<std::string, int>& params_i);
            static double d2ll_prior_normal(double x, const std::map<std::string, double>& params_d,
                const std::map<std::string, int>& params_i);
            static double ll_prior_beta(double x, const std::map<std::string, double>& params_d,
                const std::map<std::string, int>& params_i);
            static double dll_prior_beta(double x, const std::map<std::string, double>& params_d,
                const std::map<std::string, int>& params_i);
            static double d2ll_prior_beta(double x, const std::map<std::string, double>& params_d,
                const std::map<std::string, int>& params_i);
            static double ll_prior_poisson(double x, const std::map<std::string, double>& params_d,
                const std::map<std::string, int>& params_i);
            static double dll_prior_poisson(double x, const std::map<std::string, double>& params_d,
                const std::map<std::string, int>& params_i);
            static double d2ll_prior_poisson(double x, const std::map<std::string, double>& params_d,
                const std::map<std::string, int>& params_i); 

            static double dummy_prior_func(double x, const std::map<std::string, double>& params_d,
                const std::map<std::string, int>& params_i);   

            // Optional: observation weights for weighted ML calculation
            std::vector<double> weights;    
            
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

            // Maximum iterations
            int maxiter;

            // Delta threshold
            double delta_thresh;   
            
            // To be called before function evaluations: fill data structures with current
            // data points, given index            
            void prepare_data(int i);
            
            // Make things user-friendly after finding a solution and before exiting
            void fill_results(double ll);
            
        public:
            
            solver();

            bool add_data(std::string name, std::vector<double>& data);
            bool add_data(std::string name, std::vector<int>& data);
            bool add_data_fixed(std::string name, double data);
            bool add_data_fixed(std::string name, int data);
            
            bool add_weights(std::vector<double>& weights);
            
            void set_delta(double delt);
            void set_maxiter(int i);   
            
            double log_likelihood;

    };
}

#endif
