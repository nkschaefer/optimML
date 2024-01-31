#ifndef _MD_OPTIM_H
#define _MD_OPTIM_H
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

// ===== OPTIMIZATION-RELATED FUNCTIONS =====

// Generalized function to compute the first or second derivative
// of a log likelihood (or general loss) function, given a value x
// and a vector of parameters (arbitrary number)
typedef std::function< double( double, const std::vector<double>& ) > optim_func;

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

typedef std::function< double ( std::vector<double>&,
    std::map<std::string, double>&,
    std::map<std::string, int>&,
    int ) > multivar_func_d;

// Same as above, but also two indices into parameter vector for
// variables involved in second derivative (first wrt i, then wrt j)

typedef std::function< double ( std::vector<double>&,
    std::map<std::string, double>&,
    std::map<std::string, int>&,
    int, int ) > multivar_func_d2;

// Function for prior distributions over individual x variables
typedef std::function< double( double,
    std::map<std::string, double>&,
    std::map<std::string, int>& ) > multivar_prior_func;

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
        bool add_param(std::string name, std::vector<double>& data);
        bool add_param(std::string name, std::vector<int>& data);
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

/**
 * This class uses Newton-Raphson (with optional constraints) to find maximum 
 * likelihood estimates of an array of input variables.
 */
class multivar_ll_solver{
    private: 
        
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

        // How to we compute log likelihood of prior on each variable? (optional)
        std::vector<multivar_prior_func*> ll_x_prior;
        // How do we compute derivative of log likelihood of prior on each variable? (optional)
        std::vector<multivar_prior_func*> dll_dx_prior;
        // How to we compute 2nd derivative of log likelihood of prior on each variable? (optional)
        std::vector<multivar_prior_func*> d2ll_dx2_prior;
        
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
        
        multivar_ll_solver(std::vector<double> params_init,
            multivar_func ll, multivar_func_d dll, multivar_func_d2 d2ll);
         
        void add_prior(int idx, multivar_prior_func ll, multivar_prior_func dll, 
            multivar_prior_func dll2);
        
        bool add_param(std::string name, std::vector<double>& data);
        bool add_param(std::string name, std::vector<int>& data);
        
        bool add_prior_param(int idx, std::string name, double data);
        bool add_prior_param(int idx, std::string name, int data);
        
        bool add_mixcomp(std::vector<std::vector<double> >& data);
        bool add_mixcomp_fracs(std::vector<double>& fracs);
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

// This class seeks to find the maximum likelihood (or minimum
// sum of squares) solution to a problem where a mixture of an arbitrary
// number of things contributed to some measured variable, and you want
// to know what proportion of each thing went into the mixture.

// The problem/data should be of this form:

// Unknown variables (x): vector of x_1, x_2, x_3 ...
//  each x_i should be a probability between 0 and 1
//  The sum of all x_i together should equal 1

// Coefficients (A): matrix of coefficients for x.
//  Each row contains a coefficient for each x
//  so that A[i][0] * x[0] + A[i][1] * x[1] + A[i][2] * x[2] ... = y_i

// Observations (b): sampled data, or distribution parameters, that
//  relate to each y_i. This could just be a vector of expected y values
//  that you plan to compare to the predicted values (using least squares),
//  or a 2-column matrix of (mu, sigma) values representing a normal
//  distribution from which each y_i was sampled, or even a (n, k) pair of
//  observations sampled from a binomial distribution whose mean is y_i.

// Stores logit-transformed x_i values and optimizes the vector of 
//   expit(x_i) / sum_over_j(expit(x_j)), then back-transforms these
//   to probabilities at the end. This enforces the rules that each
//   must be within (0,1) and sum to one. The logit transform also puts
//   a ceiling on each x_i, so the sum of all doesn't get uncontrollably
//   large (which interferes with optimization).

// Uses Newton-Raphson and requires prior knowledge of how to build the
//  Hessian matrix.
//
class mixPropOpt{
    private:
        bool initialized;
        
        // Values to fit to (i.e. distribution params corresponding to each row)
        std::vector<std::vector<double> >* b;

        // Coefficients
        std::vector<std::vector<double> >* A;
        
        // Parameters being optimized
        std::vector<double> params;
        
        // Number of parameters being optimized
        int n_params;
        
        // Function to compute log likelihood / loss function
        optim_func y;

        // Function to compute derivative of loss function
        optim_func dy_dx;

        // Function to compute second derivative of loss function
        optim_func d2y_dx2;
        
        // Stop after the largest change in an individual probability
        // is smaller than this
        double prop_delta_thresh;

        // Stop after the change in log likelihood (or cost function)
        // is smaller than this
        double delta_thresh;
        
        // Stop after this number of iterations
        int maxits;
        
        int nb(std::vector<std::vector<double> >& b);
        
        void init(std::vector<std::vector<double> >& A,
            std::vector<std::vector<double> >& b,
            optim_func f,
            optim_func deriv,
            optim_func deriv2);
    
    public:
        // Hessian
        std::vector<std::vector<double> > H;

        // Gradient
        std::vector<double> G;
        
        std::vector<double> results;
        
        // Constructor for presets
        mixPropOpt(std::vector<std::vector<double> >& A,
            std::vector<std::vector<double> >& b,
            std::string name);

        // Constructor for custom functions
        mixPropOpt(std::vector<std::vector<double> >& A,
            std::vector<std::vector<double> >& b,
            optim_func,
            optim_func deriv,
            optim_func deriv2);

        void set_delta(double);
        void set_prop_delta(double);
        void set_maxits(int);
        void set_init_props(std::vector<double>&);

        // Note: no effort made here to calculate standard errors from
        // the Hessian, because the true "optimum" of the likelihood
        // of equations is very likely to be a function of illegal 
        // values (i.e. one proportion greater than one, or some less
        // than zero). 
        //
        // If this is the case, the standard error calculations
        // are not valid. Some individuals being approx. 0% of the mixture,
        // for example, means that those functions are pushed toward
        // negative infinity, and the curvature of the likelihood function
        // at negative infinity is not meaningful.
        //
        // It's still useful to know, though, that those individuals are 
        // 0% of the mixture. If variance of mixture proportions is
        // desired, one could estimate these proportions several different
        // ways (i.e. with different types of input data) and fit a
        // Dirichlet distribution to the proportions. 
        
        double fit();
        
        static double y_ls(double x, const std::vector<double>& p);
        static double dy_dx_ls(double x, const std::vector<double>& p);
        static double d2y_dx2_ls(double x, const std::vector<double>& p);
        static double y_norm(double x, const std::vector<double>& p);
        static double dy_dx_norm(double x, const std::vector<double>& p);
        static double d2y_dx2_norm(double x, const std::vector<double>& p);
        static double y_beta(double x, const std::vector<double>& p);
        static double dy_dx_beta(double x, const std::vector<double>& p);
        static double d2y_dx2_beta(double x, const std::vector<double>& p);
};

#endif
