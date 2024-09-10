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
#include "solver.h"

namespace optimML{
    
    // Generalized function representing multivariate log likelihood
    // functions.

    typedef std::function< double ( const std::vector<double>&,
        const std::map<std::string, double >&,
        const std::map<std::string, int>& ) > multivar_func;

    // Derivative of above - modifies last parameter to contain gradient
    // values instead of returning a single value

    typedef std::function< void ( const std::vector<double>&,
        const std::map<std::string, double>&,
        const std::map<std::string, int>&,
        std::vector<double>& ) > multivar_func_d;

    // Second derivative - modifies last parameter to contain Hessian values
    // instead of returning a single value

    typedef std::function< void ( const std::vector<double>&,
        const std::map<std::string, double>&,
        const std::map<std::string, int>&,
        std::vector<std::vector<double> >& ) > multivar_func_d2;   

    class multivar: public solver{
        
        protected:
            
            static void dummy_d2_func(const std::vector<double>& p, 
                const std::map<std::string, double>& params_d,
                const std::map<std::string, int>& params_i, 
                std::vector<std::vector<double> >& results);

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
            
            // How many groups of variables (not mixture components) that must sum to 1?
            int n_param_grp;
            
            // Which parameters are members of which groups that must sum to 1?
            std::map<int, int> param2grp;
                       
            std::vector<std::map<std::string, double> > params_prior_double;
            std::vector<std::map<std::string, int> > params_prior_int;
            
            // How to we compute log likelihood of prior on each variable? (optional)
            std::vector<prior_func> ll_x_prior;
            // How do we compute derivative of log likelihood of prior on each variable? (optional)
            std::vector<prior_func> dll_dx_prior;
            // How to we compute 2nd derivative of log likelihood of prior on each variable? (optional)
            std::vector<prior_func> d2ll_dx2_prior;
            
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
            
            // Gradient
            std::vector<double> G;

            // Hessian
            std::vector<std::vector<double> > H;
            
            // How do we compute log likelihood?
            multivar_func ll_x;
            // How do we compute derivative of log likelihood?
            multivar_func_d dll_dx;
            // How do we compute 2nd derivative of log likelihood?
            multivar_func_d2 d2ll_dx2;
            
            // Does this solver have the need/ability to calculate the second derivative?    
            bool has_2d;

            std::vector<bool> has_prior;
            // Allow users to provide a Dirichlet prior over mixture components (optional)
            bool has_prior_mixcomp;
            std::vector<double> dirichlet_prior_mixcomp;
            
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
            
            // Sums to help when calculating 1st and 2nd derivatives of mixture proportions
            double mixcompsum;
            double mixcompsum_f;
            // Also store square & cube to cut down on function evaluations
            double mixcompsum_2;
            double mixcompsum_3;
            
            const void eval_funcs_bfgs(const std::vector<double>& x, 
                double& y, std::vector<double>& grad);
        
            // Evaluate log likelihood at a given parameter value 
            double eval_ll_x(int i);       
            
            // Evaluate derivative of log likelihood function at given parameter value
            void eval_dll_dx(int i);
            
            // Evaluate second derivative of log likelihood function at given parameter value
            void eval_d2ll_dx2(int i);
            
            void fill_results(double ll);
            
            void print_function_error();
            void print_function_error_prior(int idx);
            
            void init_params(std::vector<double> params_init);
            bool replace_params(std::vector<double>& params_init);

        public:
            
            void init(std::vector<double> params_init, multivar_func ll, 
                multivar_func_d dll, multivar_func_d2 d2ll);
            
            void init(std::vector<double> params_init, multivar_func ll,
                multivar_func_d dll);
            
            void add_one_param(double p); 
            multivar();

            bool add_prior(int idx, prior_func ll, prior_func dll, 
                prior_func dll2);
            bool add_prior(int idx, prior_func ll, prior_func dll);
 
            bool add_normal_prior(int idx, double mu, double sigma);
            bool add_normal_prior(int idx, double mu, double sigma, double a, double b);
            bool add_beta_prior(int idx, double alpha, double beta);
            bool add_poisson_prior(int idx, double lambda);

            bool add_prior_param(int idx, std::string name, double data);
            bool add_prior_param(int idx, std::string name, int data);

            bool add_mixcomp(std::vector<std::vector<double> >& data);
            bool add_mixcomp_fracs(std::vector<double>& fracs);
            bool add_mixcomp_prior(std::vector<double>& alphas);
            void randomize_mixcomps();
            std::vector<double> get_cur_mixprops();

            bool set_param(int idx, double val);
            
            bool add_param_grp(std::vector<double>& pg);
            void randomize_group(int idx);

            void constrain_pos(int idx);
            void constrain_01(int idx);
            
            bool solve();
            
            void print(int idx, double lower, double upper, double step);

            // Result
            std::vector<double> results;
            
            std::vector<double> results_mixcomp;

            // Standard error (or 0 if 2nd derivative unavailable or positive at root)
            std::vector<double> se;
    
            double eval_ll_all();


    };
}

#endif
