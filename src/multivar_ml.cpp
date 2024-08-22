#include <string>
#include <algorithm>
#include <vector>
#include <iterator>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <functional>
#include <cstdlib>
#include <utility>
#include <math.h>
#include <time.h>
#include <random>
#include "functions.h"
#include "multivar_ml.h"
#include "stlbfgs/stlbfgs.h"

using std::cout;
using std::endl;
using namespace std;

// ===== multivar_ml_solver =====
// This class represents an optimizer for multivariate maximum likelihood functions.
// It requires functions to compute the log likelihood and the gradient.
// It uses LBFGS (via a third party library) to solve.

namespace optimML{
    
    multivar_ml_solver::multivar_ml_solver(){
        // Parent constructor should handle everything important
    }

    multivar_ml_solver::multivar_ml_solver(vector<double> params_init,
        multivar_func ll, multivar_func_d dll){
        init(params_init, ll, dll);
    }
    
    /**
     * When solving via BFGS, evaluate functions and deal with variable transformations
     * and resulting adjustments to derivatives.
     */
    const void multivar_ml_solver::eval_funcs_bfgs(const std::vector<double>& x_bfgs, 
        double& f_bfgs, std::vector<double>& g_bfgs){

        // Ignore the Hessian, but still calculate the gradient.
        // Zero out stuff
        for (int i = 0; i < n_param; ++i){
            // Copy param value back from BFGS solver
            x[i] = x_bfgs[i];
            G[i] = 0.0;
            dy_dt[i] = 0.0;
            if (i < n_param-nmixcomp){
                dy_dt_prior[i] = 0.0;
            }
            dt_dx[i] = 0.0;
        }
        dy_dp = 0.0;
        for (int i = 0; i < n_param_extern; ++i){
            dy_dt_extern[i] = 0.0;
        }
        if (nmixcomp > 0 && has_prior_mixcomp){
            for (int i = 0; i < nmixcomp; ++i){
                dy_dt_mixcomp_prior[i] = 0.0;
            }
        }

        // Un-transform variables and compute partial 1st and 2nd derivatives
        // wrt transformations
        
        // Note: 2nd derivatives of transformations of non-mixture component
        // variables do not depend on other variables, so off diagonal Hessian
        // entries are zero

        // Handle all non-mixture component variables
        map<int, double> grpsums;
        for (int i = 0; i < n_param_grp; ++i){
            grpsums.insert(make_pair(i, 0.0));
        } 

        for (int i = 0; i < n_param-nmixcomp; ++i){
            if (this->trans_log[i]){
                if (x[i] < xval_log_min || x[i] > xval_log_max){
                    x_skip[i] = true;
                } 
                else{
                    x_skip[i] = false;
                }
                x_t[i] = exp(x[i]);
                dt_dx[i] = x_t[i];
            }
            else if (this->trans_logit[i]){
                if (x[i] < xval_logit_min || x[i] > xval_logit_max){
                    x_skip[i] = true;
                }
                else{
                    x_skip[i] = false;
                }
                x_t[i] = expit(x[i]);
                dt_dx[i] = exp(-x[i]) / pow((exp(-x[i]) + 1), 2);
            }
            else if (param2grp.count(i) > 0){
                if (x[i] < xval_logit_min || x[i] > xval_logit_max){
                    x_skip[i] = true;
                }
                else{
                    x_skip[i] = false;
                }
                x_t[i] = expit(x[i]);
                grpsums[param2grp[i]] += x_t[i];
            }
            else{
                if (x[i] < xval_min || x[i] > xval_max){
                    x_skip[i] = true;
                }
                else{
                    x_skip[i] = false;
                }
                x_t[i] = x[i];
                dt_dx[i] = 1.0;
            }
            x_t_extern[i] = x_t[i];
        }

        for (map<int, int>::iterator pg = param2grp.begin(); pg != param2grp.end();
            ++pg){
            // Normalize each member of a sum-to-one group
            x_t[pg->first] /= grpsums[pg->second];
            x_t_extern[pg->first] = x_t[pg->first];
            
            // Handle derivatives of sum-to-one groups
            double e_negx1 = exp(-x[pg->first]);
            double e_negx1_p1_2 = pow(e_negx1 + 1, 2);
            double e_negx1_p1_3 = e_negx1_p1_2 * (e_negx1 + 1);
            double der_comp1 = e_negx1 / (e_negx1_p1_2 * grpsums[pg->second]) - 
                e_negx1 / (e_negx1_p1_3 * grpsums[pg->second] * grpsums[pg->second]);
            dt_dx[pg->first] = der_comp1;
        }
    
        // Handle all mixture component variables
        mixcompsum = 0.0;

        for (int i = n_param-nmixcomp; i < n_param; ++i){
            if (x[i] < xval_logit_min || x[i] > xval_logit_max){
                x_skip[i] = true;
            }
            else{
                x_skip[i] = false;    
            }
            x_t[i] = expit(x[i]);
            mixcompsum += x_t[i];
        }
        for (int i = n_param-nmixcomp; i < n_param; ++i){
            x_t[i] /= mixcompsum;
        }

        // Visit each data point
        double loglik = 0.0;
        for (int i = 0; i < n_data; ++i){
            
            // Update parameter maps that will be sent to functions
            prepare_data(i);
            
            // Handle p from mixture proportions, if we have mixture proportions
            // Also pre-calculate mixcompsum_f, a quantity used in derivatives of
            // transformation of mixture component variables
            mixcompsum_f = 0.0;
            if (nmixcomp > 0){
                double p = 0.0;
                for (int k = 0; k < nmixcomp; ++k){
                    p += mixcompfracs[i][k] * x_t[n_param-nmixcomp+k];
                    mixcompsum_f += mixcompfracs[i][k] / (exp(-x[k]) + 1);
                }
                x_t_extern[x_t_extern.size()-1] = p;
            }
            // Evaluate functions
            loglik += eval_ll_x(i);
            eval_dll_dx(i);
        }

        // Let prior distributions contribute and wrap up calculations
        // at the end, independent of data
        loglik += eval_ll_x(-1);
        eval_dll_dx(-1);
        

        // Make everything negative to reflect that we're minimizing instead of maximizing
        f_bfgs = -loglik;
        for (int i = 0; i < n_param; ++i){
            g_bfgs[i] = -G[i];
        }
    }  

    /**
     * Maximize the likelihood function using BFGS (allow stlbfgs library to 
     * handle this as a black box)
     */
    bool multivar_ml_solver::solve(){
        if (n_data == 0){
            // Attempt to dump in all fixed data to normal data structures
            if (!this->fixed_data_to_data()){
                fprintf(stderr, "ERROR: no data added\n");
                return false;
            }
        }
        G.clear();
        // Initialize data structures to store components of 1st and
        // 2nd derivatives 
        dt_dx.clear();
        dy_dt.clear();
        dy_dt_prior.clear();
         
        for (int i = 0; i < n_param; ++i){
            // Gradient
            G.push_back(0.0);
            
            // Partial derivatives
            dy_dt.push_back(1.0);
            dt_dx.push_back(1.0);

            // Prior derivatives
            if (i < n_param-nmixcomp){
                dy_dt_prior.push_back(1.0);
            }
        }
        
        std::function<void(const STLBFGS::vector&, double&, STLBFGS::vector&)> f = 
            [=](const STLBFGS::vector& a, double& b, STLBFGS::vector& c) {
            this->eval_funcs_bfgs(a, b, c);
        };
        
        STLBFGS::Optimizer opt{f};
        opt.verbose = false;
        
        opt.ftol = delta_thresh;
        opt.maxiter = maxiter;
        
        std::vector<double> xcopy = x;
        double res = opt.run(xcopy);
        for (int i = 0; i < n_param; ++i){
            x[i] = xcopy[i];
        }
        double ll = eval_ll_all();
        fill_results(ll);
        return true; 
    }
    
    void multivar_ml_solver::explore_starting_mixcomps_aux(set<int>& elim, 
        double& ll, 
        vector<double>& params,
        const vector<double>& params_orig){

        // How do we represent eliminated elements? 
        double perc_off = 0.001;
        
        int best_idx = -1;
        double best_ll = ll;
        vector<double> best_params;
        vector<double> best_params_orig;

        for (int elim_new = 0; elim_new < nmixcomp; ++elim_new){
            if (elim.find(elim_new) == elim.end()){
                vector<double> fracs(params_orig);
                if (fracs[elim_new] > perc_off){
                    // Alter
                    double orig = fracs[elim_new];
                    fracs[elim_new] = perc_off;
                    for (int x = 0 ; x < nmixcomp; ++x){
                        fracs[x] /= (1.0 - orig + perc_off);
                    }
                }
                this->add_mixcomp_fracs(fracs);
                this->solve();
                set<int> test = elim;
                test.insert(elim_new);
                fprintf(stderr, "ELIM");
                for (set<int>::iterator t = test.begin(); t != test.end(); ++t){
                    fprintf(stderr, " %d", *t);
                }
                fprintf(stderr, ": LL %f\n", log_likelihood);
                if (this->log_likelihood > best_ll){
                    best_idx = elim_new;
                    best_ll = log_likelihood;
                    best_params = results_mixcomp; 
                    best_params_orig = fracs;
                }
            }
        }
        /*
        if (best_ll > ll){ 
            ll = best_ll;
            params = best_params;
        }
        return;
        */
        if (best_idx == -1){
            // Can't do better than parent
            return;
        }
        else{
            // Explore children
            elim.insert(best_idx);
            ll = best_ll;
            params = best_params;
            if (elim.size() < nmixcomp){
                // Start from best fit params
                explore_starting_mixcomps_aux(elim, ll, params, best_params_orig);
            }
        }
    }

    /**
     * Seek out global maximum likelihood, when mixcomps are present,
     * in a more directed way: try even proportions, then
     * try eliminating one component at a time. Take the max of all these,
     * if it is from eliminating one component, then try eliminating
     * another, and so on, until we reach a global maximum.
     */
    bool multivar_ml_solver::explore_starting_mixcomps(){
        if (this->nmixcomp == 0){
            return false;
        }

        // Start with whatever we already have (it will be an 
        // even pool unless set differently by the user)
        vector<double> orig = get_cur_mixprops();
        solve();
        set<int> elim;
        double ll = log_likelihood;
        fprintf(stderr, "ELIM none: %f\n", ll);
        vector<double> params = results_mixcomp;
        explore_starting_mixcomps_aux(elim, ll, params, orig);
        /* 
        // Also try keeping only one of each
        double perc_off = 0.001;
        for (int i = 0; i < nmixcomp; ++i){
            vector<double> fracs(orig);
            fracs[i] = 1.0 - perc_off * (double)(nmixcomp-1);
            for (int j = 0; j < nmixcomp; ++j){
                if (j != i){
                    fracs[j] = perc_off;
                }
            }
            this->add_mixcomp_fracs(fracs);
            this->solve();
            if (this->log_likelihood > ll){
                ll = this->log_likelihood;
                params = this->results_mixcomp;
            }
        }
        */ 
        results_mixcomp = params;
        log_likelihood = ll;
        return true;
    }
    /**
     * LEGACY CODE
     * Might be useful for a future implementation of Newton-Raphson
     * 
     * Evaluate all functions when solving via a method other than BFGS
     * (i.e. Newton-Raphson); currently not used.
     *
    double multivar_ml_solver::eval_funcs(){
        // Zero out stuff
        for (int i = 0; i < n_param; ++i){
            G[i] = 0.0;
            dy_dt[i] = 0.0;
            if (i < n_param-nmixcomp){
                dy_dt_prior[i] = 0.0;
            }
            dt_dx[i] = 0.0;
            for (int j = 0; j < n_param; ++j){
                H[i][j] = 0.0;
                d2t_dx2[i][j] = 0.0;
            }
        }
        dy_dp = 0.0;
        d2y_dp2 = 0.0;
        if (nmixcomp > 0){
            for (int i = 0; i < n_param_extern-1; ++i){
                d2y_dtdp[i] = 0.0;
                d2y_dpdt[i] = 0.0;
            }
        }
        else{
            for (int i = 0; i < n_param_extern; ++i){
                dy_dt_extern[i] = 0.0;
                for (int j = 0; j < n_param_extern; ++j){
                    d2y_dt2_extern[i][j] = 0.0;
                }    
                // We won't use these, but doing this anyway
                d2y_dtdp[i] = 0.0;
                d2y_dpdt[i] = 0.0;
            }
        }
        if (nmixcomp > 0 && has_prior_mixcomp){
            for (int i = 0; i < nmixcomp; ++i){
                dy_dt_mixcomp_prior[i] = 0.0;
                d2y_dt2_mixcomp_prior[i] = 0.0;
            }
        }
        // Un-transform variables and compute partial 1st and 2nd derivatives
        // wrt transformations
        
        // Note: 2nd derivatives of transformations of non-mixture component
        // variables do not depend on other variables, so off diagonal Hessian
        // entries are zero

        // Handle all non-mixture component variables
        for (int i = 0; i < n_param-nmixcomp; ++i){
            if (this->trans_log[i]){
                if (x[i] < xval_log_min || x[i] > xval_log_max){
                    x_skip[i] = true;
                } 
                else{
                    x_skip[i] = false;
                }
                x_t[i] = exp(x[i]);
                dt_dx[i] = x_t[i];
                d2t_dx2[i][i] = x_t[i];
            }
            else if (this->trans_logit[i]){
                if (x[i] < xval_logit_min || x[i] > xval_logit_max){
                    x_skip[i] = true;
                }
                else{
                    x_skip[i] = false;
                }
                x_t[i] = expit(x[i]);
                dt_dx[i] = exp(-x[i]) / pow((exp(-x[i]) + 1), 2);
                double e_x = exp(x[i]);
                d2t_dx2[i][i] = -(e_x*(e_x - 1.0))/pow(e_x + 1.0, 3);
                
            }
            else{
                if (x[i] < xval_min || x[i] > xval_max){
                    x_skip[i] = true;
                }
                else{
                    x_skip[i] = false;
                }
                x_t[i] = x[i];
                dt_dx[i] = 1.0;
                d2t_dx2[i][i] = 1.0;
            }
            x_t_extern[i] = x_t[i];
        }
        // Handle all mixture component variables
        mixcompsum = 0.0;

        for (int i = n_param-nmixcomp; i < n_param; ++i){
            if (x[i] < xval_logit_min || x[i] > xval_logit_max){
                x_skip[i] = true;
            }
            else{
                x_skip[i] = false;    
            }
            x_t[i] = expit(x[i]);
            mixcompsum += x_t[i];
        }
        for (int i = n_param-nmixcomp; i < n_param; ++i){
            x_t[i] /= mixcompsum;
        }

        mixcompsum_2 = mixcompsum * mixcompsum;
        mixcompsum_3 = mixcompsum_2 * mixcompsum;
        // Visit each data point
        double loglik = 0.0;
        for (int i = 0; i < n_data; ++i){

            // Update parameter maps that will be sent to functions
            for (int j = 0; j < this->params_double_names.size(); ++j){
                *(this->param_double_ptr[j]) = (this->params_double_vals[j][i]);
            }
            for (int j = 0; j < this->params_int_names.size(); ++j){
                *(this->param_int_ptr[j]) = (this->params_int_vals[j][i]);
            }
            
            // Handle p from mixture proportions, if we have mixture proportions
            // Also pre-calculate mixcompsum_f, a quantity used in derivatives of
            // transformation of mixture component variables
            mixcompsum_f = 0.0;
            if (nmixcomp > 0){
                double p = 0.0;
                for (int k = 0; k < nmixcomp; ++k){
                    p += mixcompfracs[i][k] * x_t[n_param-nmixcomp+k];
                    mixcompsum_f += mixcompfracs[i][k] / (exp(-x[k]) + 1);
                }
                x_t_extern[x_t_extern.size()-1] = p;
            }
            // Evaluate functions
            loglik += eval_ll_x(i);
            eval_dll_dx(i);
            eval_d2ll_dx2(i);
        }
        // Let prior distributions contribute and wrap up calculations
        // at the end, independent of data
        loglik += eval_ll_x(-1);
        eval_dll_dx(-1);
        eval_d2ll_dx2(-1);
        
        //for (int j = 0; j < nmixcomp; ++j){
        //    if (x[n_param-nmixcomp+j] < -25 || x[n_param-nmixcomp+j] > 25){
        //        // It's effectively hit 0 or 1.
        //        G[n_param-nmixcomp+j] = 0.0;
        //        for (int k = 0; k < n_param; ++k){
        //            H[n_param-nmixcomp+j][k] = 0.0;
        //           H[k][n_param-nmixcomp+j] = 0.0;
        //        }
        //    }
        //}
        return loglik;
    }
    */

    /**
     * Check whether the function is concave (derivative root will be a maximum)
     *
     * This is done by seeing whether all eigenvalues of the Hessian are negative
     * or zero (negative semi-definite).
     *
     * If not, we assume we are converging toward a local minimum instead of a
     * local maximum and can change direction accordingly.
     *
     * NOTE: currently removed to eliminate dependency on CBLAS
     *
    bool multivar_ml_solver::check_negative_definite(){
        vector<double> eig;
        get_eigenvalues(H, eig);
        bool neg_def = true;
        for (int i = 0; i < eig.size(); ++i){
            if (eig[i] > 0){
                neg_def = false;
                break;
            }
        }
        return neg_def;
    }
    */

    /**
     * Returns a signal for whether or not to break out of the solve routine.
     *
     * NOTE: currently not used; not needed by BFGS
     *
    bool multivar_ml_solver::backtrack(vector<double>& delta_vec, 
        double& loglik, double& llprev, double& delta){

        // Strategy for backtracking from 9.7.1 Numerical Recipes
        
        // Note: this strategy was intended for root finding, but instead of looking at
        // whether the gradient is approaching zero, we look at whether the log likelihood
        // is increasing. Because of that, we swap out the gradient for the Hessian here,
        // and we replace function values with negative log likelihood values (since we
        // are maximizing instead of minimizing).

        double alpha = 1e-4;
        double lambda = 1.0;
        
        vector<double> x_orig;
        
        double llnew_alt = llprev;
        for (int i = 0; i < n_param; ++i){
            double x_old = x[i] - delta_vec[i];
            double x_new = x[i];
            llnew_alt += alpha * (x_new - x_old) * G[i];
        }

        if (loglik >= llnew_alt){
            return true;
        }

        for (int i = 0; i < n_param; ++i){
            x_orig.push_back(x[i] - delta_vec[i]);
        }
        
        // Did we accept one of the lambda values for a decreased step
        // size in the current direction?
        bool accept = false;

        fprintf(stderr, "  backtrack\n"); 
        
        // Backtrack.
        double g_0 = -llprev;
        double gprime_0 = 0.0;
        for (int i = 0; i < n_param; ++i){
            gprime_0 += G[i] * delta_vec[i];
        }
        
        double g_1 = -loglik;
        double lambda_2 = 1.0;
        double lambda_1 = -gprime_0 / (2*(g_1 - g_0 - gprime_0));
        
        double g_lambda2 = g_1;
        double g_lambda1;

        double lambda_min = 0.01;

        // Backtrack further.
        while (!accept && lambda_1 >= lambda_min){
            
            // Evaluate the function at the new value.
            for (int i = 0; i < n_param; ++i){
                x[i] = x_orig[i] + lambda_1 * delta_vec[i];
            }
            g_lambda1 = eval_ll_all();
            fprintf(stderr, "  lambda %f g %f\n", lambda_1, g_lambda1);
            if (g_lambda1 >= llnew_alt){
                accept = true;
                lambda = lambda_1;
                break;
            }
            else{    
                double coeff = 1.0/(lambda_1 - lambda_2);
                double x_1_1 = 1.0/(lambda_1 * lambda_1);
                double x_1_2 = -1.0/(lambda_2 * lambda_2);
                double x_2_1 = -lambda_2/(lambda_1 * lambda_1);
                double x_2_2 = lambda_1 / (lambda_2 * lambda_2);
                double y_1 = g_lambda1 - gprime_0*lambda_1 - g_0;
                double y_2 = g_lambda2 - gprime_0*lambda_2 - g_0;

                double a = x_1_1 * y_1 * coeff + x_1_2 * y_2 * coeff;
                double b = x_2_1 * y_1 * coeff + x_2_2 * y_2 * coeff;

                double mult = sqrt(b*b - 3.0*a*gprime_0);
                
                double soln = (-b + mult)/(3*a);
                if (soln <= 0.5*lambda_1 && soln >= 0.1*lambda_1){
                    // Update
                    lambda_2 = lambda_1;
                    g_lambda2 = g_lambda1;
                    lambda_1 = soln;
                }
                else{
                    break;
                }
            }
        }
        
        if (!accept){
            // Can't converge for some reason.
            // Revert to the previous iteration and bail out.
            fprintf(stderr, "  revert one iteration\n");
            for (int i = 0; i < n_param; ++i){
                this->x[i] = x_orig[i];
            } 
            loglik = eval_ll_all();
            fprintf(stderr, "LL %f\n", loglik);
            this->fill_results(loglik);
            return false;
        }
        else if (lambda < 1.0){
            fprintf(stderr, "  Accepted with lambda %f\n", lambda);
            // Adjust the last step.
            for (int i = 0; i < n_param; ++i){
                x[i] = x_orig[i] + lambda*delta_vec[i];
            }
            loglik = eval_ll_all();
            fprintf(stderr, "  LL %f\n", loglik);
            delta = loglik - llprev;
        }
        return true;
    }
    */
    
    /**
     * Maximize likelihood by finding roots of gradient function using Newton-Raphson
     *
     * NOTE: currently not used, in favor of BFGS, which does not need explicit
     * 2nd derivative evaluations and so requires fewer parameters, and is much
     * faster. Also avoids the need for CBLAS (for matrix inversion and computing
     * eigenvalues to check for negative definiteness). Code kept here for reference.
     *
    bool multivar_ml_solver::solve_newton(){
        if (!initialized){
            fprintf(stderr, "ERROR: not initialized\n");
            return false;
        }
        if (params_double_vals.size() == 0 && params_int_vals.size() == 0){
            fprintf(stderr, "ERROR: not initialized with data\n");
            return false;
        }
        
        // Initialize gradient and Hessian
        G.clear();
        H.clear();
        
        // Initialize data structures to store components of 1st and
        // 2nd derivatives 
        dt_dx.clear();
        d2t_dx2.clear();
        dy_dt.clear();
        dy_dt_prior.clear();
        d2y_dp2 = 0;
        d2y_dtdp.clear();
        d2y_dpdt.clear();
        
        this->xmax.clear();
        
        for (int i = 0; i < n_param; ++i){
            // Gradient
            G.push_back(0.0);
            
            // Hessian
            vector<double> H_row;
            for (int j = 0; j < n_param; ++j){
                H_row.push_back(0.0);
            }
            H.push_back(H_row);
            
            // Partial derivatives
            dy_dt.push_back(1.0);
            dt_dx.push_back(1.0);

            // Second derivatives
            vector<double> row;
            for (int j = 0; j < n_param; ++j){
                row.push_back(0.0);
            }    
            d2t_dx2.push_back(row);

            // Prior derivatives
            if (i < n_param-nmixcomp){
                dy_dt_prior.push_back(1.0);
            }

            xmax.push_back(0.0);
        }

        for (int i = 0; i < n_param_extern-1; ++i){
            d2y_dtdp.push_back(0.0);
            d2y_dpdt.push_back(0.0);
        }
        
        double delta = 999;
        int nits = 0;
        
        double llprev = 0.0;
        vector<double> delta_vec;
        
        bool mirrored_prev = false;

        while (delta > delta_thresh && (maxiter == -1 || nits < maxiter)){
            
            // Compute everything    
            double loglik = eval_funcs();
            
            // Store maximum value of log likelihood encountered so far
            // (and associated parameters), in case anything weird happens later
            if (nits == 0 || loglik > llmax){
                llmax = loglik;
                for (int z = 0; z < n_param; ++z){
                    this->xmax[z] = this->x[z];   
                }
            }

            // Check whether to terminate
            if (nits != 0){
                delta = loglik - llprev;
                if (mirrored_prev){
                    // Don't let it converge until we're back to seeking a 
                    // maximum
                    delta = 2*delta_thresh;
                }
                else{
                    bool success = backtrack(delta_vec, loglik, llprev, delta);
                    if (!success){
                        return false;
                    }
                }
            }
            
            // If not set to terminate yet, solve the system and update the x vector 
            if (delta > delta_thresh){
                // Make gradient negative
                for (int j = 0; j < n_param; ++j){
                    G[j] = -G[j];
                }
                // Solve Hx = -G
                delta_vec.clear();
                
                bool success = lstsq(H, G, delta_vec);
                if (!success){
                    fprintf(stderr, "least squares problem\n");
                    // Dump the Hessian, gradient, and parameters to screen.
                    fprintf(stderr, "H:\n");
                    for (int i = 0; i < n_param; ++i){
                        for (int j = 0; j < n_param; ++j){
                            fprintf(stderr, "%.2f ", H[i][j]);
                        }
                        fprintf(stderr, "\n");
                    }
                    fprintf(stderr, "\n");
                    for (int i = 0; i < n_param; ++i){
                        fprintf(stderr, "G[%d] = %.2f\n", i, G[i]);
                    }
                    fprintf(stderr, "\n");
                    for (int i = 0; i < n_param; ++i){
                        fprintf(stderr, "x[%d] = %.2f\n", i, x[i]);
                    }
                    exit(1);
                    return false;
                }

                // If the Hessian is not negative (semi) definite here,
                // then we're not approaching a maximum.
                
                // As a hacky solution, turn the delta vector around and
                // move away from the stationary point. Take this step
                // no matter what - and don't let the process converge
                // until we're negative definite again.

                bool nd = check_negative_definite();
                mirrored_prev = !nd;
                
                for (int j = 0; j < n_param; ++j){
                    if (nd){
                        x[j] += delta_vec[j];
                    }
                    else{
                        fprintf(stderr, "WARNING: not negative definite. Mirroring Newton step\n");
                        x[j] -= delta_vec[j];
                    }
                }
            }
            
            llprev = loglik;
            ++nits;
        }
        this->fill_results(llprev);
        fprintf(stderr, "LL: %f\n", llprev);
        
        if (maxiter != -1 && nits >= maxiter){
            // Did not converge.
            return false;
        }
        fprintf(stderr, "%d of %d iterations\n", nits, maxiter);
        return true;
    }
    */
}
 

