#ifndef _OPTIMML_MULTIVAR_H
#define _OPTIMML_MULTIVAR_H
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

// ===== Multivariate Newton-Raphson-related functions =====

// Generalized function for use with multi-variate Newton-Raphson
// solver. Returns either log likelihood or its first or second 
// derivative.
//
// Arguments:
//  parameter of independent variables at which to evaluate function
//  map of param name -> value for double params
//  map of param name -> value for int params

typedef std::function< double ( std::vector<double>&,
    std::map<std::string, double >&,
    std::map<std::string, int>& ) > multivar_func;

// Same as above, but also index into parameter vector for variable
// whose first derivative is being evaluated

typedef std::function< void ( std::vector<double>&,
    std::map<std::string, double>&,
    std::map<std::string, int>&,
    std::vector<double>& ) > multivar_func_d;

// Same as above, but also two indices into parameter vector for
// variables involved in second derivative (first wrt i, then wrt j)

typedef std::function< void ( std::vector<double>&,
    std::map<std::string, double>&,
    std::map<std::string, int>&,
    std::vector<std::vector<double> >& ) > multivar_func_d2;

// Function for prior distributions over individual x variables
typedef std::function< double( double,
    std::map<std::string, double>&,
    std::map<std::string, int>& ) > multivar_prior_func;

/**
 * This class uses Newton-Raphson (with optional constraints) to find maximum 
 * likelihood estimates of an array of input variables.
 */
class multivar_ml_solver{
    private: 
        
        // Have we set everything up?
        bool initialized;

        // How many variables in x?
        int n_param;
                
        // How many variables in x that aren't mixture components?
        int n_param_extern;

        // Current values of independent variables
        std::vector<double> x;
        
        // Current values of transformed independent variables
        std::vector<double> x_t;
        
        // Current values of transformed independent variables that aren't mixture
        // components
        std::vector<double> x_t_extern;
        
        // Allow an external function to return derivative wrt all variables under
        // consideration in one function call
        std::vector<double> dy_dt_extern;

        // Allow an external function to return all second partial derivatives 
        // in one function call
        std::vector<std::vector<double> > d2y_dt2_extern;

        // Current values of derivative of transformation function wrt x
        std::vector<double> dt_dx;

        // Current values of second derivative of transformation function wrt x
        std::vector<std::vector<double> > d2t_dx2;
        
        // Second derivative wrt to p (summary value for mixture proportions)
        std::vector<double> d2y_dpdt;
        std::vector<double> d2y_dtdp;
        double dy_dp;
        double d2y_dp2;

        // Current values of first derivative of log likelihood wrt transformed x
        std::vector<double> dy_dt;
        
        // Current values of first derivatives of prior distributions wrt transformed x
        std::vector<double> dy_dt_prior;

        // How do we compute log likelihood?
        multivar_func ll_x;
        // How do we compute derivative of log likelihood?
        multivar_func_d dll_dx;
        // How do we compute 2nd derivative of log likelihood?
        multivar_func_d2 d2ll_dx2;
        
        std::vector<bool> has_prior;
        // Allow users to provide a Dirichlet prior over mixture components (optional)
        bool has_prior_mixcomp;
        std::vector<double> dirichlet_prior_mixcomp;

        // How to we compute log likelihood of prior on each variable? (optional)
        std::vector<multivar_prior_func*> ll_x_prior;
        // How do we compute derivative of log likelihood of prior on each variable? (optional)
        std::vector<multivar_prior_func*> dll_dx_prior;
        // How to we compute 2nd derivative of log likelihood of prior on each variable? (optional)
        std::vector<multivar_prior_func*> d2ll_dx2_prior;
        
        // Fixed functions for prior for mixture components: only allow Dirichlet distribution
        double ll_mixcomp_prior();
        std::vector<double> dy_dt_mixcomp_prior;
        void dll_mixcomp_prior();
        // Don't need to store off-diagonals since there are none
        std::vector<double> d2y_dt2_mixcomp_prior;
        void d2ll_mixcomp_prior();
                
        // What is the maximum allowable value for any (un-transformed) variable to take?
        double xval_max;
        // What is the minimum allowable value for any (un-transformed) variable to take?
        double xval_min;
        // What is the minimum allowable value for any (transformed) log variable?
        double xval_log_min;
        // What is the maximum allowable value for any (transformed) log variable?
        double xval_log_max;
        // What is the minimum allowable value for any (transformed) logit variable?
        double xval_logit_min;
        // What is the maximum allowable value for any (transformed) logit variable?
        double xval_logit_max;
        
        // Should we take any variables out of consideration (slipped below min/max thresholds)?
        std::vector<bool> x_skip;
         
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
        std::vector<std::map<std::string, double> > params_prior_double;
        // For prior: map parameter name -> int
        std::vector<std::map<std::string, int> > params_prior_int;
        
        // Should the independent variables be constrained to positive numbers?
        std::vector<bool> trans_log;
        
        // Should the independent variables be constrained to (0,1)?
        std::vector<bool> trans_logit;
        
        // How many mixture component variables are there
        int nmixcomp;

        // Data for mixture components
        std::vector<std::vector<double> > mixcompfracs;
        // Computed p-values for mixture components
        std::vector<double> mixcomp_p;

        // Maximum iterations
        int maxiter;

        // Delta threshold
        double delta_thresh;
        
        // Sums to help when calculating 1st and 2nd derivatives of mixture proportions
        double mixcompsum;
        double mixcompsum_f;
        // Also store square & cube to cut down on function evaluations
        double mixcompsum_2;
        double mixcompsum_3;

        double eval_funcs();

        // Evaluate log likelihood at a given parameter value 
        double eval_ll_x(int i);       
        double eval_ll_all();

        // Evaluate derivative of log likelihood function at given parameter value
        void eval_dll_dx(int i);
        
        // Evaluate second derivative of log likelihood function at given parameter value
        void eval_d2ll_dx2(int i);
        
        // Maximum log likelihood encountered thus far
        double llmax;
        
        // Values of x at maximum log likelihood encountered thus far
        std::vector<double> xmax;
        
        void fill_results(double ll);
        
        bool check_negative_definite();
        
        bool backtrack(std::vector<double>& deltavec,
            double& loglik, double& loglik_prev, double& delta);
        
        void print_function_error();
        void print_function_error_prior(int idx);

    public:
        
        // Gradient
        std::vector<double> G;

        // Hessian
        std::vector<std::vector<double> > H;
        
        void init(std::vector<double> params_init, multivar_func ll, 
            multivar_func_d dll, multivar_func_d2 d2ll);
                
        multivar_ml_solver(std::vector<double> params_init,
            multivar_func ll, multivar_func_d dll, multivar_func_d2 d2ll);
        
        multivar_ml_solver();

        void add_prior(int idx, multivar_prior_func ll, multivar_prior_func dll, 
            multivar_prior_func dll2);
        
        bool add_data(std::string name, std::vector<double>& data);
        bool add_data(std::string name, std::vector<int>& data);
        bool add_data_fixed(std::string name, double data);
        bool add_data_fixed(std::string name, int data);

        bool add_prior_param(int idx, std::string name, double data);
        bool add_prior_param(int idx, std::string name, int data);
        
        bool add_mixcomp(std::vector<std::vector<double> >& data);
        bool add_mixcomp_fracs(std::vector<double>& fracs);
        bool add_mixcomp_prior(std::vector<double>& alphas);

        void randomize_mixcomps();
        bool set_param(int idx, double val);

        void constrain_pos(int idx);
        void constrain_01(int idx);

        void set_delta(double delt);
        void set_maxiter(int i);
        
        double log_likelihood;

        // Find root. 
        bool solve();

        // Result
        std::vector<double> results;
        
        std::vector<double> results_mixcomp;

        // Standard error (or 0 if 2nd derivative unavailable or positive at root)
        std::vector<double> se;

};

#endif
