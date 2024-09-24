#ifndef _OPTIMML_UNIVAR_H
#define _OPTIMML_UNIVAR_H
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
#include <mutex>
#include <condition_variable>
#include "functions.h"
#include "solver.h"


// ==== Class to represent solvers for univariate functions

namespace optimML{
    
    // Generalized function for use with univariate functions. 
    // Returns either log likelihood or its first or second derivative.
    // 
    // Arguments: 
    //  value at which to evaluate function
    //  map of param name -> value for double params
    //  map of param name -> value for int params
    typedef std::function< double( double,
        const std::map<std::string, double >&,
        const std::map<std::string, int >& ) > univar_func;
    
    class univar: public solver{
        protected:
            
            // How do we compute log likelihood?
            univar_func ll_x;
            // How do we compute derivative of log likelihood?
            univar_func dll_dx;
            // How do we compute 2nd derivative of log likelihood (optional)?
            univar_func d2ll_dx2;
            
            // Tolerance for convergence
            double xval_precision;

            double cur_ll_x;
            double cur_dll_dx;
            double cur_d2ll_dx2;
                        
            // Do we know the first and second derivatives?
            bool has_dll_dx;
            bool has_d2ll_dx2;
            
            // Did the user specify a prior distribution?
            bool has_prior;

            // Store functions to compute prior
            prior_func ll_x_prior;
            prior_func dll_dx_prior;
            prior_func d2ll_dx2_prior;
            
            std::map<std::string, double> params_prior_double;
            std::map<std::string, int> params_prior_int;

            // Should the independent variable be constrained to positive numbers?
            bool trans_log;
        
            // Should the independent variable be constrained to (0,1)?
            bool trans_logit;
            
            // Print warning messages for bad function evaluations
            void dump_cur_params();
            void dump_prior_params();

            // Evaluate all functions at current parameter value
            void eval_funcs(double x, bool eval_ll_x, bool eval_dll_dx, bool eval_d2ll_dx2);
            double eval_ll_x(double x);
            double eval_dll_dx(double x);
            double eval_d2ll_dx2(double x);
            void init(univar_func ll_x);
            void init(univar_func ll_x, univar_func dll_dx);
            void init(univar_func ll_x, univar_func dll_dx, univar_func d2ll_dx2);
            
            double x_t;
            double df_dt_x;
            double d2f_dt2_x;

            // ----- For multithreading
            double ll_threads;
            double dll_threads;
            double d2ll_threads;
            std::mutex* ll_mutex;
            std::mutex* dll_mutex;
            std::mutex* d2ll_mutex;
            
            bool thread_compute_ll;
            bool thread_compute_dll;
            bool thread_compute_d2ll;
            void create_threads();
            void launch_threads();
            void worker(int thread_idx) override;
            bool threads_init;

        public:
            
            univar();
            ~univar();

            void add_prior(prior_func ll);
            void add_prior(prior_func ll, prior_func dll);
            void add_prior(prior_func ll, prior_func dll, prior_func dll2);
            void add_normal_prior(double mu, double sigma);
            void add_normal_prior(double mu, double sigma, double a, double b);
            void add_beta_prior(double alpha, double beta);
            
            bool add_prior_param(std::string name, double data);
            bool add_prior_param(std::string name, int data);
            
            void constrain_pos();
            void constrain_01();
            
            void set_epsilon(double);

            // For debugging
            void print(double start, double end, double step);
            
            // Find MLE on a given interval
            // This method should be overridden by child classes
            double solve(double lower, double upper);

            // Was there a root in the interval?
            bool root_found;

    };
}

#endif
