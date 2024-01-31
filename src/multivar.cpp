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
#include "lstsq.h"
#include "functions.h"
#include "eig.h"
#include "multivar.h"

using std::cout;
using std::endl;
using namespace std;

// ===== multivar =====
//
// A multi-variate Newton-Raphson solver designed to maximize the log
// likelihood of a function with known first and second derivatives. 
//
// Features that make it useful:
//
// Incorporates a slight modification of the backtracking method described
// in Chapter 9 of Numerical Recipes -- this catches situations where an
// iteration would jump too far and takes a fraction of the original
// step instead.
//
// Detects, at each step, whether the Hessian is negative-definite. If not,
// the algorithm is likely proceeding to a minimum instead of a maximum, so
// Newton steps are taken in the opposite direction (away from the minimum)
// until the Hessian becomes negative-definite.
// 
// Can contrain parameters to (0, infinity) (through log transformation)
// or (0,1) (through logit transformation) automatically.
//
// Can incorporate a set of parameters to model mixture components. This 
// applies where a quantity of interest can be observed that is thought 
// to be the sum of a known number of components, each existing at a 
// fraction of the total mixture. 
//
// For example, suppose you have measured a set of allele frequencies, 
// each denoted as f, and you think they come from a group that is a 
// mixture of 3 populations P1, P2, and P3, and for each f, the expected 
// frequency in population 1 is e1, the expected frequency in population 
// 2 is e2, and the expected frequency in population 3 is e3. Designate 
// a set of three variables p1, p2, and p3, where each reflects the percent 
// of the pool composed of individuals from the corresponding population, 
// and p1 + p2 + p3 = 1. Then, for each allele, you can compare the measured 
// quantity f to the expected quantity p1*e1 + p2*e2 + p3*e3.
//
// This class can handle these types of situations by logit-transforming 
// each p, then modeling each variable as the un-transformed version divided 
// by the sum of all un-transformed p. This prevents any p from slipping below
// 0 or above 1, or from the sum of all p deviating from 1.
//

// ----- Solve for maximum likelihood of multivariate equation with -----
// ----- known 1st and second derivatives                           -----

void multivar_ml_solver::init(vector<double> params_init, multivar_func ll,
    multivar_func_d dll, multivar_func_d2 dll2){
    
    initialized = true;
    srand(time(NULL));
     
    ll_x = ll;
    dll_dx = dll;
    d2ll_dx2 = dll2;
    

    for (int i = 0; i < params_init.size(); ++i){
        x.push_back(params_init[i]);
        xmax.push_back(0.0);
        x_t.push_back(params_init[i]);
        x_t_extern.push_back(params_init[i]);
        x_skip.push_back(false);

        results.push_back(0.0);
        se.push_back(0.0);

        has_prior.push_back(false);
        ll_x_prior.push_back(NULL);
        dll_dx_prior.push_back(NULL);
        d2ll_dx2_prior.push_back(NULL);

        trans_log.push_back(false);
        trans_logit.push_back(false);  
        
        map<string, double> m1;
        params_prior_double.push_back(m1);
        map<string, int> m2;
        params_prior_int.push_back(m2);

        
    }
    n_param = params_init.size();
    n_param_extern = params_init.size();

    n_data = 0;
    maxiter = 1000;
    delta_thresh = 0.01;
    log_likelihood = 0.0;
    
    nmixcomp = 0;
    
    xval_max = 1e15;
    xval_min = 1e-15;
    xval_log_min = log(xval_min);
    xval_log_max = log(xval_max);
    xval_logit_min = logit(xval_min);
    xval_logit_max = logit(1.0-xval_min);

}


multivar_ml_solver::multivar_ml_solver(vector<double> params_init,
    multivar_func ll, multivar_func_d dll, multivar_func_d2 dll2){
    init(params_init, ll, dll, dll2);  
}

multivar_ml_solver::multivar_ml_solver(){
    initialized = false;
}

void multivar_ml_solver::add_prior(int idx, multivar_prior_func ll,
    multivar_prior_func dll, multivar_prior_func dll2){
    
    if (!initialized){
        fprintf(stderr, "ERROR: not initialized\n");
        exit(1);
    }
    this->has_prior[idx] = true;
    this->ll_x_prior[idx] = &ll;
    this->dll_dx_prior[idx] = &dll;
    this->d2ll_dx2_prior[idx] = &dll2;

}

bool multivar_ml_solver::add_prior_param(int idx, string name, double data){
    if (!initialized){
        fprintf(stderr, "ERROR: not initialized\n");
        exit(1);
    }
    this->params_prior_double[idx].insert(make_pair(name, data));
    return true;
}

bool multivar_ml_solver::add_prior_param(int idx, string name, int data){
    if (!initialized){
        fprintf(stderr, "ERROR: not initialized\n");
        exit(1);
    }
    this->params_prior_int[idx].insert(make_pair(name, data));
    return true;
}

bool multivar_ml_solver::add_data(string name, std::vector<double>& dat){
    if (!initialized){
        fprintf(stderr, "ERROR: not initialized\n");
        exit(1);
    }
    // Make sure dimensions agree
    int nd = dat.size();
    if (this->n_data != 0 && this->n_data != nd){
        fprintf(stderr, "ERROR: data vectors do not have same dimensions\n");
        return false;
    }
    if (param_double_cur.count(name) > 0){
        fprintf(stderr, "ERROR: already has data keyed to %s\n", name.c_str());
        return false;
    }
    this->n_data = nd;
    this->params_double_names.push_back(name);
    this->params_double_vals.push_back(dat.data());
    this->param_double_cur.insert(make_pair(name, 0.0));
    this->param_double_ptr.push_back(&(this->param_double_cur.at(name)));
    return true;
}

bool multivar_ml_solver::add_data(string name, std::vector<int>& dat){
    if (!initialized){
        fprintf(stderr, "ERROR: not initialized\n");
        exit(1);
    }
    // Make sure dimensions agree
    int nd = dat.size();
    if (this->n_data != 0 && this->n_data != nd){
        fprintf(stderr, "ERROR: data vectors do not have same dimensions\n");
        return false;
    }
    if (param_int_cur.count(name) > 0){
        fprintf(stderr, "ERROR: already has data keyed to %s\n", name.c_str());
        return false;
    }
    this->n_data = nd;
    this->params_int_names.push_back(name);
    this->params_int_vals.push_back(dat.data());
    this->param_int_cur.insert(make_pair(name, 0.0));
    this->param_int_ptr.push_back(&(this->param_int_cur.at(name)));
    return true;
}

bool multivar_ml_solver::add_data_fixed(string name, double dat){
    if (!initialized){
        fprintf(stderr, "ERROR: not initialized\n");
        exit(1);
    }
    if (param_double_cur.count(name) > 0){
        fprintf(stderr, "ERROR: already has data keyed to %s\n", name.c_str());
        return false;
    }
    param_double_cur.insert(make_pair(name, dat));
    return true;
}

bool multivar_ml_solver::add_data_fixed(string name, int dat){
    if (!initialized){
        fprintf(stderr, "ERROR: not initialized\n");
        exit(1);
    }
    if (param_int_cur.count(name) > 0){
        fprintf(stderr, "ERROR: already has data keyeed to %s\n", name.c_str());
        return false;
    }
    param_int_cur.insert(make_pair(name, dat));
    return true;
}
/**
 * This class can also have a mixture of components as one of its variables.
 * A mixture of components is a set of n fractions between 0 and 1, which must
 * sum to 1. The user provides (via this function) rows of data corresponding
 * to the other data provided with add_param(). Each row of data is a coefficient
 * on each mixture proportion. In other words, to model a 3 way mixture of things,
 * provide here a vector of vectors, where each sub-vector is a row of data, containing
 * 3 elements: coefficient on proportion 1 (f1), coefficient on proportion 2 (f2), and
 * coefficient on proportion 3 (f3). The model will consider 3 new parameters c1, c2, and 
 * c3, where each is the fraction of the mixture made of each component, and where 
 * c1 + c2 + c3 = 1.0. Each row of data will then produce a new variable, 
 * p = f1c1 + f2c2 + f3c3, which will be exposed to the provided functions as the last
 * entry in x. 
 */
bool multivar_ml_solver::add_mixcomp(vector<vector<double> >& dat){
    if (!initialized){
        fprintf(stderr, "ERROR: not initialized\n");
        exit(1);
    }
    // Make sure dimensions agree.
    int nd = dat.size();
    if (this->n_data != 0 && this->n_data != nd){
        fprintf(stderr, "ERROR: data vectors do not have same dimensions\n");
        return false;
    }
    if (nd == 0){
        fprintf(stderr, "ERROR: component fraction matrix has no rows\n");
        return false;
    }
    int ncomp = dat[0].size();
    for (int i = 0; i < dat.size(); ++i){
        if (ncomp != dat[i].size()){
            fprintf(stderr, "ERROR: component fraction matrix has differing numbers of columns\n");
            this->mixcompfracs.clear();
            return false;
        }
        this->mixcompfracs.push_back(dat[i]);
        this->mixcomp_p.push_back(0.0);
    }

    // Outside functions will see this as a single variable
    // Add a single slot in the variable array for this new variable
    this->n_param_extern = n_param + 1;
    this->x_t_extern.push_back(0.0);
    
    // Initialize to an even pool
    this->nmixcomp = ncomp;
    for (int i = 0; i < ncomp; ++i){
        n_param++;
        x.push_back(logit(1.0 / (double)ncomp));
        x_t.push_back(0.0);
        x_skip.push_back(false);
    }
    return true;
}

bool multivar_ml_solver::add_mixcomp_fracs(vector<double>& fracs){
    if (!initialized){
        fprintf(stderr, "ERROR: not initialized\n");
        exit(1);
    }
    if (fracs.size() != nmixcomp){
        fprintf(stderr, "ERROR: number of mixture props does not match stored data\n");
        return false;
    }
    double sum = 0.0;
    for (int i = 0; i < fracs.size(); ++i){
        sum += fracs[i];
    }
    for (int i = 0; i < fracs.size(); ++i){
        x[n_param-nmixcomp+i] = logit(fracs[i] / sum);
    }
    return true;
}

void multivar_ml_solver::randomize_mixcomps(){
    if (!initialized){
        fprintf(stderr, "ERROR: not initialized\n");
        exit(1);
    }
    // Randomly re-sample starting mixture proportions. Equivalent to sampling from
    // a Dirichlet distribution with all alpha_i = 1
    double sum = 0.0;
    for (int i = 0; i < nmixcomp; ++i){
        double r = (double)rand() / (double)RAND_MAX;
        x[n_param-nmixcomp+i] = r;
        sum += r;
    }    
    for (int i = 0; i < nmixcomp; ++i){
        x[n_param-nmixcomp+i] = logit(x[n_param-nmixcomp+i]/sum);
    }
}

bool multivar_ml_solver::set_param(int idx, double val){
    if (!initialized){
        fprintf(stderr, "ERROR: not initialized\n");
        exit(1);
    }
    if (idx >= n_param-nmixcomp){
        fprintf(stderr, "ERROR: illegal parameter index\n");
        return false;
    }
    else if (this->trans_log[idx] && val <= 0){
        fprintf(stderr, "ERROR: illegal parameter value for positive-constrained variable\n");
        return false;
    }
    else if (this->trans_logit[idx] && (val <= 0 || val >= 1)){
        fprintf(stderr, "ERROR: illegal parameter value for logit-transformed variable\n");
        return false;
    }
    if (this->trans_log[idx]){
        x[idx] = log(val);
    }
    else if (this->trans_logit[idx]){
        x[idx] = logit(val);
    }
    else{
        x[idx] = val;
    }
    return true;
}

void multivar_ml_solver::constrain_pos(int idx){
    if (!initialized){
        fprintf(stderr, "ERROR: not initialized\n");
        exit(1);
    }
    // Un-transform if necessary
    if (this->trans_log[idx]){
        return;
    }
    else if (this->trans_logit[idx]){
        x[idx] = expit(x[idx]);
    }
    this->trans_logit[idx] = false;
    this->trans_log[idx] = true;
    if (x[idx] <= 0){
        fprintf(stderr, "ERROR: initial value %d out of domain for log transformation\n", idx);
        return;
    }
    x[idx] = log(x[idx]);
}

void multivar_ml_solver::constrain_01(int idx){
    if (!initialized){
        fprintf(stderr, "ERROR: not initialized\n");
        exit(1);
    }
    // Un-transform if necessary
    if (this->trans_logit[idx]){
        return;
    }
    else if (this->trans_log[idx]){
        x[idx] = exp(x[idx]);
    }
    this->trans_log[idx] = false;
    this->trans_logit[idx] = true;
    if (x[idx] <= 0 || x[idx] >= 1.0){
        fprintf(stderr, "ERROR: initial value %d out of domain for logit transformation\n", idx);
        return;
    }
    x[idx] = logit(x[idx]);
}

void multivar_ml_solver::set_delta(double d){
    if (!initialized){
        fprintf(stderr, "ERROR: not initialized\n");
        exit(1);
    }
    this->delta_thresh = d;
}

void multivar_ml_solver::set_maxiter(int m){
    if (!initialized){
        fprintf(stderr, "ERROR: not initialized\n");
        exit(1);
    }
    this->maxiter = m;
}

void multivar_ml_solver::print_function_error(){
    fprintf(stderr, "parameters:\n");
    for (int i = 0; i < x_t_extern.size(); ++i){
        fprintf(stderr, "%d): %f\n", i, x_t_extern[i]);
    }
    fprintf(stderr, "data:\n");
    for (map<string, double>::iterator p = this->param_double_cur.begin(); 
        p != this->param_double_cur.end(); ++p){
        fprintf(stderr, "  %s = %f\n", p->first.c_str(), p->second);
    }
    for (map<string, int>::iterator i = this->param_int_cur.begin();
        i != this->param_int_cur.end(); ++i){
        fprintf(stderr, "  %s = %d\n", i->first.c_str(), i->second);
    }
}

void multivar_ml_solver::print_function_error_prior(int idx){
    fprintf(stderr, "parameter:\n");
    fprintf(stderr, "%d): %f\n", idx, x_t_extern[idx]);
    fprintf(stderr, "data:\n");

    for (map<string, double>::iterator p = this->params_prior_double[idx].begin(); 
        p != this->params_prior_double[idx].end(); ++p){
        fprintf(stderr, "  %s = %f\n", p->first.c_str(), p->second);
    }
    for (map<string, int>::iterator i = this->params_prior_int[idx].begin();
        i != this->params_prior_int[idx].end(); ++i){
        fprintf(stderr, "  %s = %d\n", i->first.c_str(), i->second);
    }
}

// Evaluate log likelihood at current vector of values
double multivar_ml_solver::eval_ll_x(int i){
    double f_x = 0.0;
    if (i >= 0){
        // Get log likelihood of one (current) row of data
        double ll = ll_x(x_t_extern, this->param_double_cur, this->param_int_cur);
        if (isnan(ll) || isinf(ll)){
            fprintf(stderr, "ERROR: illegal value returned by log likelihood function\n");
            print_function_error();
            exit(1);
        }
        f_x += ll;
    }
    else{
        for (int j = 0; j < n_param-nmixcomp; ++j){
            if (this->has_prior[j]){
                double ll = (*ll_x_prior[j])(x_t_extern[j], this->params_prior_double[j], this->params_prior_int[j]);
                if (isnan(ll) || isinf(ll)){
                    fprintf(stderr, "ERROR: illegal value returned by prior log likelihood function on \
parameter %d\n", j);
                    print_function_error_prior(j);
                    exit(1);
                }
                f_x += ll;
            }
        }
    }
    return f_x;
}



double multivar_ml_solver::eval_ll_all(){
    double loglik = 0.0;

    // Transform all non-mixture component variables
    for (int i = 0; i < n_param-nmixcomp; ++i){
        if (this->trans_log[i]){
            x_t[i] = exp(x[i]);
        }
        else if (this->trans_logit[i]){
            x_t[i] = expit(x[i]);
        }
        else{
            x_t[i] = x[i];
        }
        x_t_extern[i] = x_t[i];
    }
    
    // Transform all mixture component variables
    mixcompsum = 0.0;
    for (int i = n_param-nmixcomp; i < n_param; ++i){
        x_t[i] = expit(x[i]);
        mixcompsum += x_t[i];
    }
    for (int i = n_param-nmixcomp; i < n_param; ++i){
        x_t[i] /= mixcompsum;
    }
    
    for (int i = 0; i < n_data; ++i){

        // Update parameter maps that will be sent to functions
        for (int j = 0; j < this->params_double_names.size(); ++j){
            *(this->param_double_ptr[j]) = (this->params_double_vals[j][i]);
        }
        for (int j = 0; j < this->params_int_names.size(); ++j){
            *(this->param_int_ptr[j]) = (this->params_int_vals[j][i]);
        }
        
        // Handle p from mixture proportions, if we have mixture proportions
        if (nmixcomp > 0){
            double p = 0.0;
            for (int k = 0; k < nmixcomp; ++k){
                p += mixcompfracs[i][k] * x_t[n_param-nmixcomp+k];
            }
            x_t_extern[x_t_extern.size()-1] = p;
        }
        
        // Evaluate functions
        loglik += eval_ll_x(i);
    }

    loglik += eval_ll_x(-1);
    return loglik;
}

// Evaluate derivative of log likelihood at current vector; store in 
// gradient vector.
void multivar_ml_solver::eval_dll_dx(int i){
    if (i >= 0){
        for (int j = 0; j < n_param_extern; ++j){
            double dy_dt_this = dll_dx(x_t_extern, this->param_double_cur, this->param_int_cur, j);
            if (isnan(dy_dt_this) || isinf(dy_dt_this)){
                fprintf(stderr, "ERROR: invalid value returned by gradient function: parameter %d\n", j);
                print_function_error();
                exit(1);
            }
            if (nmixcomp > 0 && j == n_param_extern-1){
                dy_dp = dy_dt_this;
            }
            else{
                dy_dt[j] = dy_dt_this;
                if (x_skip[j]){
                    G[j] = 0.0;
                    dt_dx[j] = 0.0;
                }
                else{
                    G[j] += dy_dt[j] * dt_dx[j];
                }
            }
        }
        if (nmixcomp > 0){
            double p = x_t_extern[x_t_extern.size()-1];
            for (int j = 0; j < nmixcomp; ++j){
                if (x_skip[n_param-nmixcomp+j]){
                    G[n_param-nmixcomp+j] = 0.0;
                    dt_dx[n_param-nmixcomp+j] = 0.0;
                }
                else{
                    double e_negx = exp(-x[n_param-nmixcomp+j]);
                    double e_negx_p1 = e_negx + 1.0;
                    double e_negx_p1_2 = e_negx_p1*e_negx_p1;
                    dt_dx[n_param-nmixcomp + j] = ((e_negx)/(e_negx_p1_2 * mixcompsum)) * 
                        (this->mixcompfracs[i][j] - mixcompsum_f/mixcompsum);
                    G[n_param-nmixcomp+j] += dy_dp * dt_dx[n_param-nmixcomp + j];
                }
            }
        }
    }
    else{
        for (int j = 0; j < n_param-nmixcomp; ++j){
            //G[j] = dy_dt[j] * dt_dx[j];
            if (this->has_prior[j]){
                if (x_skip[j]){
                    // Keep it at zero
                }
                else{
                    double dllprior = (*dll_dx_prior[j])(x_t[j], this->params_prior_double[j], 
                        this->params_prior_int[j]);
                    if (isnan(dllprior) || isinf(dllprior)){
                        fprintf(stderr, "ERROR: illegal value returned by prior gradient function on \
parameter %d\n", j);
                        print_function_error_prior(j); 
                        exit(1);
                    }
                    dy_dt_prior[j] = dllprior;
                    G[j] += dy_dt_prior[j] * dt_dx[j]; 
                }
            }
        }
        /*
        for (int j = n_param-nmixcomp; j < n_param; ++j){
            fprintf(stderr, "dy_dp = %f dt_dx[%d] = %f\n", dy_dp, j, dt_dx[j]);
            G[j] = dy_dp * dt_dx[j]; 
        }
        */
    }
}

// Evaluate second derivative at current parameter values; store results in
// Hessian matrix.
void multivar_ml_solver::eval_d2ll_dx2(int i){

    if (i >= 0){

        // Handle second derivatives involving pairs of non-mix comps
        // Also compute second derivatives wrt p (the summary mixture parameter) 
        for (int j = 0; j < n_param_extern; ++j){
            for (int k = 0; k < n_param_extern; ++k){
                
                bool j_is_p = nmixcomp > 0 && j == n_param_extern-1;
                bool k_is_p = nmixcomp > 0 && k == n_param_extern-1;
                                
                double deriv2 = d2ll_dx2(x_t_extern, this->param_double_cur, this->param_int_cur, j, k);
                if (isnan(deriv2) || isinf(deriv2)){
                    fprintf(stderr, "ERROR: illegal value returned by 2nd derivative function on \
parameters: %d %d\n", j, k);
                    print_function_error();
                    exit(1);
                }

                if (j_is_p && k_is_p){
                    d2y_dp2 = deriv2;
                }
                else if (j_is_p){
                    if (!x_skip[k]){
                        d2y_dpdt[k] = deriv2;
                    }
                }
                else if (k_is_p){
                    if (!x_skip[j]){
                        d2y_dtdp[j] = deriv2;
                    }
                }
                else{
                    if (!x_skip[j] && !x_skip[k]){
                        H[j][k] += deriv2 * d2t_dx2[j][k];
                    }
                }
            }
        }
        if (nmixcomp > 0){
            double p = x_t_extern[x_t_extern.size()-1];
            for (int j = 0; j < nmixcomp; ++j){
                int j_i = n_param-nmixcomp + j;
                
                if (x_skip[j_i]){
                    continue;
                }

                // Handle second derivatives involving mix comps + non mix comps
                
                for (int k = 0; k < n_param-nmixcomp; ++k){
                    if (!x_skip[k]){
                        H[j_i][k] += dt_dx[j_i] * d2y_dpdt[k];
                        H[k][j_i] += d2y_dtdp[k] * dt_dx[j_i];   
                    }
                }
                
                double exp_negx_p1_j = exp(-x[j_i]) + 1;
                double exp_negx_p1_j_2 = exp_negx_p1_j * exp_negx_p1_j;
                
                // Handle second derivatives involving pairs of mix comps

                for (int k = 0; k < nmixcomp; ++k){
                    int k_i = n_param-nmixcomp + k;
                    
                    if (x_skip[k_i]){
                        continue;
                    }

                    double exp_negx_p1_k = exp(-x[k_i]) + 1;
                    double exp_negx_p1_k_2 = exp_negx_p1_k * exp_negx_p1_k;
                    if (j == k){
                        double exp_negx = exp(-x[j_i]);
                        double exp_neg2x = exp(-2*x[j_i]);
                        double exp_negx_p1_j_3 = exp_negx_p1_j_2 * exp_negx_p1_j;
                        double exp_negx_p1_j_4 = exp_negx_p1_j_3 * exp_negx_p1_j;
                        d2t_dx2[j_i][k_i] = 0;
                        d2t_dx2[j_i][k_i] +=  -(exp_negx * mixcompfracs[i][j]) / 
                            (exp_negx_p1_j_2 * mixcompsum);
                        d2t_dx2[j_i][k_i] += (2*exp_neg2x*mixcompfracs[i][j]) / 
                            (exp_negx_p1_j_3 * mixcompsum);
                        d2t_dx2[j_i][k_i] -= (exp_neg2x * mixcompfracs[i][j]) / 
                            (exp_negx_p1_j_4 * mixcompsum_2);
                        d2t_dx2[j_i][k_i] += (exp_negx * mixcompsum_f) / 
                            (exp_negx_p1_j_2 * mixcompsum_2);
                        d2t_dx2[j_i][k_i] -= (2*exp_neg2x * mixcompsum_f) / 
                            (exp_negx_p1_j_3 * mixcompsum_2);
                        d2t_dx2[j_i][k_i] += (2*exp_neg2x * mixcompsum_f) / 
                            (exp_negx_p1_j_4 * mixcompsum_3);
                        d2t_dx2[j_i][k_i] += (exp_neg2x * mixcompfracs[i][j]) / 
                            (exp_negx_p1_j_4 * mixcompsum_2);
                        
                        /* 
                        if (isnan(d2t_dx2[j_i][k_i]) || isinf(d2t_dx2[j_i][k_i])){
                            
                            d2t_dx2[j_i][k_i] = 0.0;
                            fprintf(stderr, "h\n");
                            fprintf(stderr, "%f\n", exp_negx);
                            fprintf(stderr, "%f\n", exp_neg2x);
                            fprintf(stderr, "%f %f %f\n", exp_negx_p1_j_2,
                                exp_negx_p1_j_3, exp_negx_p1_j_4);
                            fprintf(stderr, "%f %f %f\n", mixcompsum, mixcompsum_2,
                                mixcompsum_3);
                            exit(1);
                        }
                        */

                        H[j_i][k_i] += d2y_dp2 * dt_dx[j_i] * dt_dx[k_i] + 
                            dy_dp * d2t_dx2[j_i][k_i];
                        
                    }
                    else{
                        double exp_negx_jminusk = exp(-x[j] - x[k]);
                        d2t_dx2[j_i][k_i] = 0;
                        d2t_dx2[j_i][k_i] += -(mixcompfracs[i][j] * exp_negx_jminusk) / 
                            (exp_negx_p1_j_2 * exp_negx_p1_k_2 * mixcompsum_2);
                        d2t_dx2[j_i][k_i] -= (mixcompfracs[i][k] * exp_negx_jminusk) / 
                            (exp_negx_p1_j_2 * exp_negx_p1_k_2 * mixcompsum_2);
                        d2t_dx2[j_i][k_i] += (2 * exp_negx_jminusk * mixcompsum_f) / 
                            (exp_negx_p1_j_2 * exp_negx_p1_k_2 * mixcompsum_3);
                        
                        /* 
                        if (isinf(d2t_dx2[j_i][k_i]) || isnan(d2t_dx2[j_i][k_i])){
                            d2t_dx2[j_i][k_i] = 0.0;
                        }
                        */

                        H[j_i][k_i] += d2y_dp2 * dt_dx[j_i] * dt_dx[k_i] + 
                            dy_dp * d2t_dx2[j_i][k_i];
                        
                    }
                }
            }
        }
    }
    else{
        for (int j = 0; j < n_param-nmixcomp; ++j){
            /*
            for (int k = 0; k < n_param-nmixcomp; ++k){
                H[j][k] *= dt_dx[j] * dt_dx[k];
                if (j == k){
                    H[j][j] += dy_dt[j] * d2t_dx2[j][j];
                }
            }
            */
            if (this->has_prior[j] && !x_skip[j]){
                double d2llprior = (*d2ll_dx2_prior[j])(x_t[j], this->params_prior_double[j], 
                    this->params_prior_int[j]) * dt_dx[j] * dt_dx[j] + dy_dt_prior[j] * d2t_dx2[j][j]; 
                if (isnan(d2llprior) || isinf(d2llprior)){
                    fprintf(stderr, "ERROR: illegal value returned by 2nd derivative function for prior \
on parameter %d\n", j);
                    print_function_error_prior(j);
                    exit(1);
                }
                H[j][j] += d2llprior;
            }
        }
        /*
        for (int j = 0; j < nmixcomp; ++j){
            int j_i = n_param-nmixcomp+j;
            // Handle 2nd derivatives involving all other mix props (and including self)
            for (int k = 0; k < nmixcomp; ++k){
                int k_i = n_param-nmixcomp+k;
                H[j_i][k_i] += d2y_dp2 * dt_dx[j_i] * dt_dx[k_i] + dy_dp * d2t_dx2[j_i][k_i]; 
            }
            // Handle 2nd derivatives involving all other variables
            for (int k = 0; k < n_param-nmixcomp; ++k){
                H[j_i][k] += dt_dx[j_i] * d2y_dpdt[k];
                H[k][j_i] += d2y_dtdp[k] * dt_dx[j_i];
            }
        }
        */
    }
}

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
        // We won't use these, but doing this anyway
        for (int i = 0; i < n_param_extern; ++i){
            d2y_dtdp[i] = 0.0;
            d2y_dpdt[i] = 0.0;
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
    /*
    for (int j = 0; j < nmixcomp; ++j){
        if (x[n_param-nmixcomp+j] < -25 || x[n_param-nmixcomp+j] > 25){
            // It's effectively hit 0 or 1.
            G[n_param-nmixcomp+j] = 0.0;
            for (int k = 0; k < n_param; ++k){
                H[n_param-nmixcomp+j][k] = 0.0;
                H[k][n_param-nmixcomp+j] = 0.0;
            }
        }
    }
    */

    return loglik;
}

void multivar_ml_solver::fill_results(double llprev){
    // Store un-transformed results
    for (int j = 0; j < n_param-nmixcomp; ++j){
        if (trans_log[j]){
            results[j] = exp(x[j]);
        }
        else if (trans_logit[j]){
            results[j] = expit(x[j]);
        }
        else{
            results[j] = x[j];
        } 
    }
    results_mixcomp.clear();
    double mcsum = 0.0;
    for (int j = 0; j < nmixcomp; ++j){
        double val = expit(x[n_param-nmixcomp+j]);
        mcsum += val;
        results_mixcomp.push_back(val);
    }
    for (int j = 0; j < nmixcomp; ++j){
        results_mixcomp[j] /= mcsum;
    }
    this->log_likelihood = llprev;
}

/**
 * Check whether the function is concave (derivative root will be a maximum)
 *
 * This is done by seeing whether all eigenvalues of the Hessian are negative
 * or zero (negative semi-definite).
 *
 * If not, we assume we are converging toward a local minimum instead of a
 * local maximum and can change direction accordingly.
 *
 */
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

/**
 * Returns a signal for whether or not to break out of the solve routine.
 */
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
        /*
        g_lambda1 = -loglik;
        for (int i = 0; i < n_param; ++i){
            g_lambda1 += G[i] * lambda_1 * delta_vec[i];
        }
        g_lambda1 = -g_lambda1;
        */
        fprintf(stderr, "  lambda %f g %f\n", lambda_1, g_lambda1);
        /*
        for (int i = 0; i < n_param; ++i){
            x[i] = x_orig[i] + lambda_1 * delta_vec[i];        
        }
        double lltest = eval_ll_all();
        */
        if (g_lambda1 >= llnew_alt){
        //if (lltest >= llnew_alt){     
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

bool multivar_ml_solver::solve(){
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
        //fprintf(stderr, "LL %f -> %f\n", llprev, loglik);
        
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

