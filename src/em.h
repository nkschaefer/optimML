#ifndef _OPTIMML_EM_H
#define _OPTIMML_EM_H
#include <string>
#include <algorithm>
#include <vector>
#include <deque>
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
    /*    
    // Function to be used in solving a system of equations. 
    typedef std::function< double ( const std::vector<double>& ) > multivar_sys_func;
    // Derivative of above
    typedef std::function< void ( const std::vector<double>&, std::vector<double>& ) > multivar_sys_func_d;
    */

    // Fits mixture models using the EM algorithm. Instead of each component having its own parameters
    // that are updated by the method of moments (fast), however, each component can depend on 
    // a set of global parameters that are updated via maximum likelihood. 

    class em_solver{
        
        private:
            std::vector<double> params; 
            std::vector<double> params_extern;
            std::vector<double> params_orig;
            
            int nthreads;

            int n_params;
            std::deque<multivar_ml_solver* > solvers;
            //std::vector<multivar_ml_solver*> solvers;
            //multivar_ml_solver solver_global;
            multivar_ml_solver* solver_global;

            std::vector<multivar_func> dist_funcs;
            std::vector<multivar_func_d> dist_funcs_deriv;

            void init_responsibility_matrix(int no);            
            void free_responsibility_matrix();
            
            double llfunc(const std::vector<double>& p,
                const std::map<std::string, double>& dd,
                const std::map<std::string, int>& di);
            
            void dllfunc(const std::vector<double>& p,
                const std::map<std::string, double>& dd,
                const std::map<std::string, int>& di,
                std::vector<double>& r);
            
            multivar_func llfunc_wrapper;
            multivar_func_d dllfunc_wrapper;

            double delta_thresh;
            int maxiter;
            
            void E_step();
            double M_step();
            
            bool initialized;
            bool is_fit;
            bool hard;

            double weightsum;

            bool no_data_yet = true;
            std::vector<int> data_idx;
            std::vector<std::vector<double> > lls_tmp;
            std::vector<double> lls_tmp_rowsum;
            
            std::vector<double> log_component_weights;
            
            void compute_scores();

        public:
            
            std::vector<double> weights_global;
            std::vector<double> component_weights;
            std::vector<double> compsums;

            std::vector<std::string> component_names;
            
            int n_components; 
            int n_obs;
            void add_one_param(double p);
            void add_params(std::vector<double>& p);
            void set_param(int param_idx, double p);

            // Initialize with initial guesses of parameters
            em_solver(std::vector<double> params_init);
            em_solver();
            
            // Destructor
            ~em_solver();

            void add_param_grp(std::vector<double>& p);

            // Parameter values at optimum
            std::vector<double> results;
            
            // Responsibility matrix / obs-component weights
            double** responsibility_matrix;

            bool add_component(multivar_func func,
                multivar_func_d func_deriv);
            
            bool add_component(multivar_func func,
                multivar_func_d func_deriv,
                std::string s);
            
            bool rm_component(int ci);
            
            void reset_params();

            std::vector<double> frac_p_components();
            
            void elim_dists_by_count(int skipdist=-1);

            // Add data key/val to go with most recently-added equation
            bool add_data(std::string name, std::vector<double>& dat);
            bool add_data(std::string name, std::vector<int>& dat);
            
            bool add_data_fixed(std::string name, double dat);
            bool add_data_fixed(std::string name, int dat);
            
            bool add_prior(int idx, prior_func ll, prior_func dll);
 
            bool add_normal_prior(int idx, double mu, double sigma);
            bool add_normal_prior(int idx, double mu, double sigma, double a, double b);
            bool add_beta_prior(int idx, double alpha, double beta);
            bool add_poisson_prior(int idx, double lambda);

            bool add_prior_param(int idx, std::string name, double data);
            bool add_prior_param(int idx, std::string name, int data);           
            
            void set_threads(int nt);
            void set_bfgs_threads(int nt);
            void set_maxiter(int m);
            void set_hard(bool h);

            bool constrain_pos(int idx);
            bool constrain_01(int idx);
            
            // For adding global weights outside the EM framework 
            void add_weights(std::vector<double>& w);
            
            void set_delta(double d);
            
            void init();

            double fit();
            
            std::vector<int> rm_correlated_components();
            std::vector<short> assignments;

            double loglik;
            double bic;
            double aic;
            double icl;
            double entropy;
            std::vector<double> component_entropy;
            std::vector<double> component_ll;
            std::vector<double> weights_hard;
            std::vector<double> delta_icl;
            std::vector<double> delta_entropy;
            void print();
            void print_rm();
            void test_cutoffs();
    };

}

#endif
