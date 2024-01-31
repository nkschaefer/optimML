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
#include "optim.h"
#include "eig.h"

using std::cout;
using std::endl;
using namespace std;

// ===== optim =====
// Functions related to optimization problems

// ----- Brent's method root finder

// Set default values.
void brentSolver::init(){
    this->has_d2ll = false;
    this->n_data = 0; 
    this->trans_log = false;
    this->trans_logit = false;
    this->maxiter = 100;
    this->delta_thresh = 1e-6; 
    this->root = 0.0;
    this->se = 0.0;
    this->root_found = false;
    this->se_found = false;
    this->has_prior = false;
}

// Initialize without knowledge of second derivative
brentSolver::brentSolver(brent_func ll, brent_func dll){
    this->init();
    this->ll_x = ll;
    this->dll_dx = dll;
    this->has_d2ll = false;
}

// Initialize with second derivative
brentSolver::brentSolver(brent_func ll, brent_func dll, brent_func dll2){
    this->init();
    this->ll_x = ll;
    this->dll_dx = dll;
    this->d2ll_dx2 = dll2;
    this->has_d2ll = true;
}

// Add in a prior without knowledge of second derivative
void brentSolver::add_prior(brent_func ll, brent_func dll){
    if (this->has_d2ll){
        fprintf(stderr, "WARNING: second derivative supplied for log likelihood \
function but not for prior.\n");
        fprintf(stderr, "Estimation of standard error will be disabled.\n");
        this->has_d2ll = false;
    }
    this->ll_x_prior = ll;
    this->dll_dx_prior = dll;
    this->has_prior = true;
}

// Add in a prior with knowledge of second derivative
void brentSolver::add_prior(brent_func ll, brent_func dll, brent_func dll2){
    if (!this->has_d2ll){
        fprintf(stderr, "WARNING: second derivative supplied for prior but not \
for log likelihood function.\n");
        fprintf(stderr, "Estimation of standard error will be disabled.\n");
        this->has_d2ll = false;
    }
    this->ll_x_prior = ll;
    this->dll_dx_prior = dll;
    this->d2ll_dx2_prior = dll2;
    this->has_prior = true;
}

// Constrain independent variable to be positive
void brentSolver::constrain_pos(){
    this->trans_log = true;
    this->trans_logit = false;
}

// Constrain independent variable to be in (0,1)
void brentSolver::constrain_01(){
    this->trans_log = false;
    this->trans_logit = true;
}

// Set delta thresh
void brentSolver::set_delta(double delt){
    this->delta_thresh = delt;
}

// Set maximum iterations
void brentSolver::set_maxiter(int i){
    this->maxiter = i;
}

// Add in data for log likelihood function
bool brentSolver::add_param(string name, std::vector<double>& dat){
    // Make sure dimensions agree
    int nd = dat.size();
    if (this->n_data != 0 && this->n_data != nd){
        fprintf(stderr, "ERROR: data vectors do not have same dimensions\n");
        return false;
    }
    this->n_data = nd;
    this->params_double_names.push_back(name);
    this->params_double_vals.push_back(dat.data());
    this->param_double_cur.insert(make_pair(name, 0.0));
    this->param_double_ptr.push_back(&(this->param_double_cur.at(name)));
    //this->params_double.insert(make_pair(name, dat));    
    return true;
}
bool brentSolver::add_param(string name, std::vector<int>& dat){
    // Make sure dimensions agree
    int nd = dat.size();
    if (this->n_data != 0 && this->n_data != nd){
        fprintf(stderr, "ERROR: data vectors do not have same dimensions\n");
        return false;
    }
    this->n_data = nd;
    this->params_int_names.push_back(name);
    this->params_int_vals.push_back(dat.data());
    this->param_int_cur.insert(make_pair(name, 0.0));
    this->param_int_ptr.push_back(&(this->param_int_cur.at(name)));
    //this->params_int.insert(make_pair(name, dat));
    return true;
}

// Add in data for prior
bool brentSolver::add_prior_param(string name, double dat){
    this->params_prior_double.insert(make_pair(name, dat));
    return true;
}
bool brentSolver::add_prior_param(string name, int dat){
    this->params_prior_int.insert(make_pair(name, dat));
    return true;
}

/**
 * For debugging: prints log likelihood, derivative, and possibly second derivative
 * over the specified range.
 */
void brentSolver::print(double lower, double upper, double step){
    for (double x = lower; x <= upper; x += step){
        double ll = eval_ll_x(x);
        double deriv1 = eval_dll_dx(x);
        if (this->has_d2ll){
            double deriv2 = eval_d2ll_dx2(x);
            fprintf(stdout, "%f\t%f\t%f\t%f\n", x, ll, deriv1, deriv2);
        }
        else{
            fprintf(stdout, "%f\t%f\t%f\n", x, ll, deriv1);
        }
    }
}

// Evaluate log likelihood at a value
double brentSolver::eval_ll_x(double x){
    double x_t = x;
    if (this->trans_log){
        x_t = exp(x);
    }
    else if (this->trans_logit){
        x_t = expit(x);
    }
    double f_x = 0.0;

    for (int i = 0; i < this->n_data; ++i){
        for (int j = 0; j < this->params_double_names.size(); ++j){
            *(this->param_double_ptr[j]) = (this->params_double_vals[j][i]);
        }
        for (int j = 0; j < this->params_int_names.size(); ++j){
            *(this->param_int_ptr[j]) = (this->params_int_vals[j][i]);
        }
        f_x += ll_x(x_t, this->param_double_cur, this->param_int_cur);
    }
    if (this->has_prior){
        f_x += ll_x_prior(x_t, this->params_prior_double, this->params_prior_int);
    }
    return f_x;
}

// Evaluate derivative of log likelihood at a value
double brentSolver::eval_dll_dx(double x){
    
    double x_t = x;
    double df_dt_x = 1.0; 
    if (this->trans_log){
        x_t = exp(x);
        // f(x) = e^x -> df_dx = e^x
        df_dt_x = x_t;
    }
    else if (this->trans_logit){
        x_t = expit(x);
        // f(x) = 1/(1 + e^(-x)) -> df_dx = exp(-x) / ((exp(-x)) + 1)^2
        df_dt_x = exp(-x) / pow(((exp(-x)) + 1), 2);
    }
    
    double f_x = 0.0;
    for (int i = 0; i < this->n_data; ++i){
        for (int j = 0; j < this->params_double_names.size(); ++j){
            *(this->param_double_ptr[j]) = (this->params_double_vals[j][i]);
        }
        for (int j = 0; j < this->params_int_names.size(); ++j){
            *(this->param_int_ptr[j]) = (this->params_int_vals[j][i]);
        }
        f_x += dll_dx(x_t, this->param_double_cur, this->param_int_cur) * df_dt_x;
    }
    if (this->has_prior){
        f_x += dll_dx_prior(x_t, this->params_prior_double, this->params_prior_int) * df_dt_x; 
    }
    return f_x;
}

// Evaluate second derivative of log likelihood at a value
double brentSolver::eval_d2ll_dx2(double x){
    double x_t = x;
    double d2f_dt2_x = 1.0; 
    if (this->trans_log){
        x_t = exp(x);
        // f(x) = e^x -> d2f_dx2 = e^x
        d2f_dt2_x = x_t;
    }
    else if (this->trans_logit){
        x_t = expit(x);
        // f(x) = 1/(1 + e^(-x)) -> d2f_dx2 = -(exp(x)*(exp(x) - 1))/(exp(x) + 1)^3
        double e_x = exp(x);
        double d2f_dt2_x = (e_x*(e_x - 1.0))/pow(e_x + 1.0, 3);
    }
    double f_x = 0.0;
    for (int i = 0; i < this->n_data; ++i){
        for (int j = 0; j < this->params_double_names.size(); ++j){
            *(this->param_double_ptr[j]) = (this->params_double_vals[j][i]);
        }
        for (int j = 0; j < this->params_int_names.size(); ++j){
            *(this->param_int_ptr[j]) = (this->params_int_vals[j][i]);
        }
        f_x += d2ll_dx2(x_t, this->param_double_cur, this->param_int_cur) * d2f_dt2_x;
    }
    if (this->has_prior){
        f_x += d2ll_dx2_prior(x_t, this->params_prior_double, this->params_prior_int) * d2f_dt2_x; 
    }
    return f_x;
}


/**
 * Brent's method for finding a root of a 1-parameter function
 * in a fixed interval.
 *
 * When an update would send the parameter out of the interval, defaults to
 * the bisection method for that iteration, as described in chapter 9.3 of
 * Numerical Recipes in C++, 3rd edition, by Press, Teukolsky, Vetterling, 
 * & Flannery: https://numerical.recipes/book.html
 *
 * Optionally allows parameters to be constrained (by log-transformation or
 * logit-transformation), according to class parameters.
 */
double brentSolver::solve(double lower, double upper){
    // Attempt to make interval feasible if transformations are being used.
    if (this->trans_log && lower == 0){
        lower += this->delta_thresh;
    }
    else if (this->trans_logit && (lower == 0 || upper == 1)){
        if (lower == 0){
            lower += this->delta_thresh;
        }
        if (upper == 1){
            upper -= this->delta_thresh;
        }
    }
    if ((this->trans_log && (lower <= 0 || upper <= 0)) ||
        (this->trans_logit && (lower <= 0 || upper >= 1))){
        fprintf(stderr, "ERROR: bounds of (%f, %f) are not compatible with the \
transformation of the data.\n", lower, upper);
        this->root_found = false;
        this->se_found = false;
        this->root = 0.0;
        this->se = 0.0;
        return log(0.0);
    }
    
    // Initial bounds (a & c) and guess of root (b)
    double a = lower;
    double b = (lower+upper)/2.0;
    double c = upper;
    
    if (this->trans_log){
        // each variable is log(x), operate on e(x)
        a = log(lower);
        c = log(upper);
        //b = (a+c)/2.0;
        //b = log(b);
        
        if (a > c){
            double tmp = a;
            a = c;
            c = tmp;
        }
        b = (a+c)/2.0;
    }
    else if (this->trans_logit){
        // Each variable is logit(x), operate on expit(x)
        a = logit(lower);
        c = logit(upper);
        //b = (a+c)/2.0;
        //b = logit(b);
        
        if (a > c){
            double tmp = a;
            a = c;
            c = tmp;
        }
        b = (a+c)/2.0;
    }

    double delta = 999;
    int nits = 0;
    
    // Functions evaluated at each point
    double f_a = 0.0;
    double f_b = 0.0;
    double f_c = 0.0;
    
    // This section is equivalent to eval_dll_dx above, but more efficient
    // since we need to compute three values at once -- just loop over the 
    // data once
    
    double a_t = a;
    double b_t = b;
    double c_t = c;
    
    double df_dt_a = 1.0;
    double df_dt_b = 1.0;
    double df_dt_c = 1.0;

    if (this->trans_log){
        a_t = exp(a);
        b_t = exp(b);
        c_t = exp(c);
        
        // f(x) = e^x -> df_dx = e^x
        df_dt_a = a_t;
        df_dt_b = b_t;
        df_dt_c = c_t;
    }
    else if (this->trans_logit){
        a_t = expit(a);
        b_t = expit(b);
        c_t = expit(c);
        
        // f(x) = 1/(1 + e^(-x)) -> df_dx = exp(-x) / ((exp(-x)) + 1)^2
        df_dt_a = exp(-a) / pow(((exp(-a)) + 1), 2);
        df_dt_b = exp(-b) / pow(((exp(-b)) + 1), 2);
        df_dt_c = exp(-c) / pow(((exp(-c)) + 1), 2);
    }
    
    for (int i = 0; i < this->n_data; ++i){
        for (int j = 0; j < this->params_double_names.size(); ++j){
            *(this->param_double_ptr[j]) = (this->params_double_vals[j][i]);
        }
        for (int j = 0; j < this->params_int_names.size(); ++j){
            *(this->param_int_ptr[j]) = (this->params_int_vals[j][i]);
        }
        f_a += this->dll_dx(a_t, this->param_double_cur, this->param_int_cur) * df_dt_a;
        f_b += this->dll_dx(b_t, this->param_double_cur, this->param_int_cur) * df_dt_b;
        f_c += this->dll_dx(c_t, this->param_double_cur, this->param_int_cur) * df_dt_c;
    }
    if (this->has_prior){
        f_a += this->dll_dx_prior(a_t, this->params_prior_double, this->params_prior_int) * df_dt_a;
        f_b += this->dll_dx_prior(b_t, this->params_prior_double, this->params_prior_int) * df_dt_b;
        f_c += this->dll_dx_prior(c_t, this->params_prior_double, this->params_prior_int) * df_dt_c;
    }
    
    // Account for possible weirdness at boundaries (i.e. if likelihood is beta and range is (0,1)
    if (isinf(f_a) || isnan(f_a)){
        if (this->trans_log){
            a = log(exp(a) + this->delta_thresh);
        }
        else if (this->trans_logit){
            a = logit(expit(a) + this->delta_thresh);
        }
        else{
            a += this->delta_thresh;
        }
        f_a = eval_dll_dx(a);
    }
    if (isinf(f_c) || isnan(f_c)){
        if (this->trans_log){
            c = log(exp(c) - this->delta_thresh);
        }
        else if (this->trans_logit){
            c = logit(expit(c) - this->delta_thresh);
        }
        else{
            c -= this->delta_thresh;
        }
        f_c = eval_dll_dx(c);
    }
    

    // Function must have opposite signs on either side of the interval
    // for a root to exist.  
    bool root_in_interval = (f_a < 0 && f_c > 0) || (f_a > 0 && f_c < 0);
    
    if (!root_in_interval){

        this->root_found = false;
        this->se_found = false;
        this->root = 0.0;
        this->se = 0.0;

        // No root. Pick whichever boundary has lower log likelihood.
        double ll_a = eval_ll_x(a);
        double ll_c = eval_ll_x(c);
        
        a_t = a;
        c_t = c;
    
        if (this->trans_log){
            a_t = exp(a);
            c_t = exp(c);
        }
        else if (this->trans_logit){
            a_t = expit(a);
            c_t = expit(c);
        }

        if (ll_a > ll_c){
            return a_t;
        }
        else{
            return c_t;
        }
    }

    while (delta > this->delta_thresh && nits < this->maxiter){
        ++nits;

        double f_a_safe = f_a;
        double f_b_safe = f_b;
        double f_c_safe = f_c;
        
        // Prevent overflow/underflow by dividing all numbers by
        // smallest absolute value
        double minabs = abs(f_a);
        if (abs(f_b) < minabs){
            minabs = abs(f_b);
        }
        if (abs(f_c) < minabs){
            minabs = abs(f_c);
        }
        f_a_safe /= minabs;
        f_b_safe /= minabs;
        f_c_safe /= minabs;
        
        double R = f_b_safe/f_c_safe;
        double S = f_b_safe/f_a_safe;
        double T = f_a_safe/f_c_safe;
        double P = S*(T*(R-T)*(c-b) - (1.0-R)*(b-a));
        double Q = (T-1.0)*(R-1.0)*(S-1.0);
        
        if (Q == 0 || (nits == 1 && P/Q < this->delta_thresh) || isnan(P/Q) || 
            (b + P/Q) <= a || (b + P/Q) >= c){

            // Use bisection.
            if (f_a < 0){
                if (f_b < 0){
                    a = b;
                    f_a = f_b;
                }
                else{
                    c = b;
                    f_c = f_b;
                }
            }
            else{
                // c < 0
                if (f_b < 0){
                    c = b;
                    f_c = f_b;
                }
                else{
                    a = b;
                    f_a = f_b;
                }
            }
            delta = abs(b - (a+c)/2.0 + a);
            b = (a+c)/2.0;
            f_b = eval_dll_dx(b);
        }    
        else{
            delta = abs(P/Q);
            double x = b + P/Q;
            if (f_a < 0){
                if (f_b < 0){
                    a = b;
                    f_a = f_b;
                }
                else{
                    c = b;
                    f_c = f_b;
                }
            }
            else{
                // f_c < 0
                if (f_b < 0){
                    c = b;
                    f_c = f_b;
                }
                else{
                    a = b;
                    f_a = f_b;
                }
            }
            b = x;
            f_b = eval_dll_dx(b);
        }
    }
    
    this->root_found = true;
    b_t = b;

    // Transform back, if necessary
    if (this->trans_log){
        b_t = exp(b);
    }
    else if (this->trans_logit){
        b_t = expit(b);
    }
    this->root = b_t;

    if (this->has_d2ll){
        
        // Try to get standard error here
        double y = eval_d2ll_dx2(b);
        
        if (y < 0.0){
            double se = sqrt(-1.0/y);
            this->se = se;
            this->se_found = true;
        }
        else{
            this->se_found = false;
            this->se = 0.0;
        }
    }
    return b_t;
}

// ----- Solve for maximum likelihood of multivariate equation with -----
// ----- known 1st and second derivatives                           -----

multivar_ll_solver::multivar_ll_solver(vector<double> params_init,
    multivar_func ll, multivar_func_d dll, multivar_func_d2 dll2){
    
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

void multivar_ll_solver::add_prior(int idx, multivar_prior_func ll,
    multivar_prior_func dll, multivar_prior_func dll2){
    
    this->has_prior[idx] = true;
    this->ll_x_prior[idx] = &ll;
    this->dll_dx_prior[idx] = &dll;
    this->d2ll_dx2_prior[idx] = &dll2;

}

bool multivar_ll_solver::add_prior_param(int idx, string name, double data){
    this->params_prior_double[idx].insert(make_pair(name, data));
    return true;
}

bool multivar_ll_solver::add_prior_param(int idx, string name, int data){
    this->params_prior_int[idx].insert(make_pair(name, data));
    return true;
}

bool multivar_ll_solver::add_param(string name, std::vector<double>& dat){
    // Make sure dimensions agree
    int nd = dat.size();
    if (this->n_data != 0 && this->n_data != nd){
        fprintf(stderr, "ERROR: data vectors do not have same dimensions\n");
        return false;
    }
    this->n_data = nd;
    this->params_double_names.push_back(name);
    this->params_double_vals.push_back(dat.data());
    this->param_double_cur.insert(make_pair(name, 0.0));
    this->param_double_ptr.push_back(&(this->param_double_cur.at(name)));
    return true;
}

bool multivar_ll_solver::add_param(string name, std::vector<int>& dat){
    // Make sure dimensions agree
    int nd = dat.size();
    if (this->n_data != 0 && this->n_data != nd){
        fprintf(stderr, "ERROR: data vectors do not have same dimensions\n");
        return false;
    }
    this->n_data = nd;
    this->params_int_names.push_back(name);
    this->params_int_vals.push_back(dat.data());
    this->param_int_cur.insert(make_pair(name, 0.0));
    this->param_int_ptr.push_back(&(this->param_int_cur.at(name)));
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
bool multivar_ll_solver::add_mixcomp(vector<vector<double> >& dat){
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

bool multivar_ll_solver::add_mixcomp_fracs(vector<double>& fracs){
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

void multivar_ll_solver::randomize_mixcomps(){

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

bool multivar_ll_solver::set_param(int idx, double val){
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

void multivar_ll_solver::constrain_pos(int idx){
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

void multivar_ll_solver::constrain_01(int idx){
    
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

void multivar_ll_solver::set_delta(double d){
    this->delta_thresh = d;
}

void multivar_ll_solver::set_maxiter(int m){
    this->maxiter = m;
}

void multivar_ll_solver::print_function_error(){
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

void multivar_ll_solver::print_function_error_prior(int idx){
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
double multivar_ll_solver::eval_ll_x(int i){
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



double multivar_ll_solver::eval_ll_all(){
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
void multivar_ll_solver::eval_dll_dx(int i){
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
void multivar_ll_solver::eval_d2ll_dx2(int i){

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

double multivar_ll_solver::eval_funcs(){
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

void multivar_ll_solver::fill_results(double llprev){
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
bool multivar_ll_solver::check_negative_definite(){
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
bool multivar_ll_solver::backtrack(vector<double>& delta_vec, 
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

bool multivar_ll_solver::solve(){
    
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
    
    vector<double> G_prev;
    
    bool mirrored_prev = false;

    while (delta > delta_thresh && (maxiter == -1 || nits < maxiter)){
        
        G_prev = G;

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
    return true;
}


// ----- Optimization of log likelihood of mixture proportions

/**
 * Cost function least squares
 */
double mixPropOpt::y_ls(double x, const vector<double>& params){
    return pow(x - params[0], 2);
}

/**
 * dy_dx least squares
 */
double mixPropOpt::dy_dx_ls(double x, const vector<double>& params){
    return 2*x - 2*params[0];
}

/**
 * d^2y/dx^2 least squares
 */
double mixPropOpt::d2y_dx2_ls(double x, const vector<double>& params){
    return 2.0;
}

/**
 * Log likelihood Normal
 *
 * Note that for truncated normal, derivatives don't depend on the
 * normalization, so optimizing a normal is the same as optimizing
 * a truncated normal
 */
double mixPropOpt::y_norm(double x, const vector<double>& params){
    double mu = params[0];
    double sigma = params[1];
    return -0.5*pow((x-mu)/sigma, 2) - log(sigma) - log(sqrt(2*M_PI));
}

/**
 * dy_dx Normal
 */
double mixPropOpt::dy_dx_norm(double x, const vector<double>& params){
    double mu = params[0];
    double sigma = params[1];
    return (x-mu)/(sigma * sigma);
}

/**
 * d^2y/dx^2 Normal
 */
double mixPropOpt::d2y_dx2_norm(double x, const vector<double>& params){
    return 1.0/(params[1] * params[1]);
}

/**
 * Log likelihood Beta
 */
double mixPropOpt::y_beta(double x, const vector<double>& params){
    double alpha = params[0];
    double beta = params[1];
    return (alpha-1.0)*log(x) + (beta-1.0)*log(1.0-x) + lgamma(alpha+beta) - 
        lgamma(alpha) - lgamma(beta);
}

/**
 * dy_dx Beta
 */
double mixPropOpt::dy_dx_beta(double x, const vector<double>& params){
    double alpha = params[0];
    double beta = params[1];
    return (alpha- 1.0)/x - (beta - 1.0)/(1.0-x);
}

/**
 * d^2y_dx^2 Beta
 */
double mixPropOpt::d2y_dx2_beta(double x, const vector<double>& params){
    double alpha = params[0];
    double beta = params[1];
    return (1.0-alpha)/(x*x) + (1.0-beta)/pow(1.0-x, 2);
}

// Check the dimensions of a b matrix.
int mixPropOpt::nb(vector<vector<double> >& b){
    int nb = -1;
    for (int i = 0; i < b.size(); ++i){
        if (nb == -1){
            nb = b[i].size();
        }
        else if (nb != b[i].size()){
            fprintf(stderr, "ERROR: inconsistent dimensions in b matrix\n");
            exit(1);
        }
    }
    return nb;
}

/**
 * Initialize object
 */
void mixPropOpt::init(vector<vector<double> >& A,
    vector<vector<double> >& b,
    optim_func ll,
    optim_func deriv,
    optim_func deriv2){
    
    if (this->initialized){
        fprintf(stderr, "ERROR: cannot re-initialize mixPropOpt object\n");
        exit(1);
    }

    this->y = ll;
    this->dy_dx = deriv;
    this->d2y_dx2 = deriv2;
    
    this->A = &A;
    this->b = &b;

    // How many Xs are there?
    int n_x = -1;
    for (int i = 0; i < A.size(); ++i){
        if (n_x == -1){
            n_x = A[i].size();
        }
        else if (n_x != A[i].size()){
            fprintf(stderr, "ERROR: A does not have consistent number of columns; cannot solve.\n");
            this->initialized = false;
            return;
        }
    }
     
    this->n_params = n_x;

    // Initialize x vector (assume an even pool to start)
    for (int i = 0; i < n_x; ++i){
        double prob = 1.0/(double)n_x;
        this->params.push_back(logit(prob));
    }

    // Initialize Hessian and gradient
    for (int i = 0; i < n_x; ++i){
        this->G.push_back(0.0);
        vector<double> H_row;
        for (int j = 0; j < n_x; ++j){
            H_row.push_back(0.0);
        }
        this->H.push_back(H_row);
    }

    this->delta_thresh = 0.1;
    this->prop_delta_thresh = 1e-4;
    this->maxits = 500;

    this->initialized = true;
}

mixPropOpt::mixPropOpt(vector<vector<double> >& A,
    vector<vector<double> >& b,
    optim_func ll,
    optim_func deriv,
    optim_func deriv2){
    this->initialized = false;
    this->init(A, b, ll, deriv, deriv2);
}

mixPropOpt::mixPropOpt(vector<vector<double> >& A,
    vector<vector<double> >& b,
    string name){
    this->initialized = false;
    // Handle presets.
    if (name == "ls" || name == "LS" || name == "ols" || name == "OLS"){
        int nb_this = nb(b);
        if (nb_this != 1){
            fprintf(stderr, "ERROR: least squares: b has %d columns instead of expected 1\n", nb_this);
            this->initialized = false;
            return;
        }
        this->init(A, b, mixPropOpt::y_ls, mixPropOpt::dy_dx_ls, mixPropOpt::d2y_dx2_ls);
        
        // Need a different delta threshold here - instead of tracking change in log likelihood,
        // tracking change in sum of squared residuals
        this->delta_thresh = 0.001;
    }
    else if (name == "normal" || name == "Normal" || name == "gaussian" ||
        name == "Gaussian" || name == "gauss" || name == "Gauss" || 
        name == "norm" || name == "Norm"){
        int nb_this = nb(b);
        if (nb_this != 2){
            fprintf(stderr, "ERROR: Normal: b has %d columns instead of expected 2\n", nb_this);
            this->initialized = false;
            return;
        }
        this->init(A, b, mixPropOpt::y_norm, mixPropOpt::dy_dx_norm, mixPropOpt::d2y_dx2_norm);
    }
    else if (name == "beta" || name == "Beta"){
        int nb_this = nb(b);
        if (nb_this != 2){
            fprintf(stderr, "ERROR: Beta: b has %d columns instead of expected 2\n", nb_this);
            this->initialized = false;
            return;
        }
        this->init(A, b, mixPropOpt::y_beta, mixPropOpt::dy_dx_beta, mixPropOpt::d2y_dx2_beta);
    }
}

void mixPropOpt::set_delta(double thresh){
    this->delta_thresh = thresh;
}

void mixPropOpt::set_prop_delta(double thresh){
    this->prop_delta_thresh = thresh;
}

void mixPropOpt::set_maxits(int m){
    this->maxits = m;
}

void mixPropOpt::set_init_props(vector<double>& props){
    if (props.size() != this->params.size()){
        fprintf(stderr, "ERROR: initializing starting proportions with different number \
of proportions than parameters\n");
        exit(1);
    }
    // Ensure they sum to 1
    double tot = 0.0;
    for (int i = 0; i < props.size(); ++i){
        tot += props[i];
    }
    if (tot == 0){
        fprintf(stderr, "ERROR: starting proportions can not sum to 0\n");
        exit(1);
    }
    for (int i = 0; i < props.size(); ++i){
        this->params[i] = logit(props[i] / tot);
    }
}

/**
 * Returns log likelihood (or cost function) evaluated at optimum
 */
double mixPropOpt::fit(){
    if (!this->initialized){
        fprintf(stderr, "ERROR: mixPropOpt object not initialized\n");
        return false;
    }

    double delta = 9999;
    double prop_delta = 9999;
    int its = 0;
    double loglik_prev = 0.0;

    while (delta > this->delta_thresh && prop_delta > this->prop_delta_thresh && 
        its < this->maxits){
        
        double loglik = 0.0;

        // Zero out Hessian and gradient
        if (its > 0){
            for (int i = 0; i < this->n_params; ++i){
                this->G[i] = 0.0;
                for (int j = 0; j < this->n_params; ++j){
                    this->H[i][j] = 0.0;
                }
            }
        }
        
        // Compute sum of back-transformed parameters
        double s1 = 0.0;
        for (int i = 0; i < this->n_params; ++i){
            s1 += expit(params[i]);
        }
        
        // Pre-compute the square and cube
        double s1_2 = s1*s1;
        double s1_3 = s1_2*s1;

        // Pre-compute the un-transformed (probability) version of each parameter 
        vector<double> param_frac;
        for (int i = 0; i < this->params.size(); ++i){
            double p = expit(this->params[i]) / s1;
            param_frac.push_back(p);
        }
        
        // r = row index
        for (int r = 0; r < A->size(); ++r){
            
            // Derivative of the transformed value x = A[r][0]*x[0] + A[r][1]*x[1] ...
            // with respect to transformed value of individual vector element
            vector<double> dx_db;

            // Same as above but second partial derivatives
            vector<vector<double> > d2x_db2;

            // This is the summed contribution of each sample * expectation from each sample
            double x = 0.0;
            
            // s2 = another sum that will come up often 
            double s2 = 0.0;
            
            // c = column index
            for (int c = 0; c < n_params; ++c){
                double f_i = (*A)[r][c];
                s2 += f_i/(exp(-params[c]) + 1);
            }

            for (int c = 0; c < n_params; ++c){
                            
                double f_i = (*A)[r][c];
                int i = c;

                x += f_i * param_frac[c];
                
                double exp_b_i = exp(-params[i]);
                double exp_b_i_p1 = exp_b_i + 1.0;
                double exp_b_i_p1_2 = exp_b_i_p1 * exp_b_i_p1;
                double exp_b_i_p1_3 = exp_b_i_p1_2 * exp_b_i_p1;
                double exp_b_i_p1_4 = exp_b_i_p1_3 * exp_b_i_p1;
                
                double exp_2b_i = exp(-2.0*params[i]);
                
                // dx wrt this entry in beta (param) vector
                double dx_db_i = (exp_b_i)/(exp_b_i_p1_2 * s1) * (f_i - s2/s1);
                
                // d^2x wrt this entry in beta param vector, then entry
                // corresponding to idx in this array
                vector<double> d2x_db2_row;
                for (int c2 = 0; c2 < n_params; ++c2){
                    d2x_db2_row.push_back(0.0);
                }
                
                // Handle d^2x / db_i^2
                d2x_db2_row[i] += (-exp_b_i*f_i)/(exp_b_i_p1_2*s1);
                d2x_db2_row[i] += (2*exp_2b_i*f_i)/(exp_b_i_p1_3*s1);
                d2x_db2_row[i] += -(exp_2b_i*f_i)/(exp_b_i_p1_4*s1_2);
                d2x_db2_row[i] += (exp_b_i * s2)/(exp_b_i_p1_2*s1_2);
                d2x_db2_row[i] += -(2*exp_2b_i * s2)/(exp_b_i_p1_3*s1_2);
                d2x_db2_row[i] += (2*exp_2b_i * s2)/(exp_b_i_p1_4*s1_3);
                d2x_db2_row[i] += (exp_2b_i * f_i)/(exp_b_i_p1_4*s1_2);
                
                // Handle all d^2x / db_idb_j
                for (int j = 0; j < n_params; ++j){
                    if (j != i){
                        double f_j = (*A)[r][j];
                        double exp_bi_bj = exp(-params[i] - params[j]);
                        double exp_b_j_p1_2 = pow(exp(-params[j]) + 1, 2);
                        d2x_db2_row[j] += -(f_i * exp_bi_bj)/(exp_b_i_p1_2 * exp_b_j_p1_2 * s1_2);
                        d2x_db2_row[j] += -(f_j * exp_bi_bj)/(exp_b_i_p1_2 * exp_b_j_p1_2 * s1_2);
                        d2x_db2_row[j] += (2*exp_bi_bj*s2)/(exp_b_i_p1_2 * exp_b_j_p1_2 * s1_3);
                    }
                }
                
                d2x_db2.push_back(d2x_db2_row);
                dx_db.push_back(dx_db_i);
            }
            double y_row = this->y(x, (*b)[r]);    
            double dy_dx_row = this->dy_dx(x, (*b)[r]);
            double d2y_dx2_row = this->d2y_dx2(x, (*b)[r]);
            
            // Update log likelihood / loss function
            loglik += y_row;

            // Update gradient & Hessian
            for (int j = 0; j < this->n_params; ++j){
                G[j] += (dy_dx_row * dx_db[j]);
                for (int k = 0; k < this->n_params; ++k){
                    H[j][k] += d2y_dx2_row * dx_db[j] * dx_db[k] + dy_dx_row * d2x_db2[j][k];        
                }
            }

        } // rows of A
        
        // Make gradient negative
        for (int i = 0; i < G.size(); ++i){
            G[i] = -G[i];
        }

        // Solve
        vector<double> deltavec;
        bool success = lstsq(H, G, deltavec);
        if (!success){
            fprintf(stderr, "error with matrix operation\n");
            return 0.0;
        }

        double maxd = 0.0;
        double x_denom_old = 0.0;
        double x_denom_new = 0.0;
        vector<double> params_new;
        for (int i = 0; i < params.size(); ++i){
            double pnew = params[i] + deltavec[i];
            //fprintf(stderr, "param %d) %f -> %f\n", i, params[i], pnew);
            params_new.push_back(pnew);
            x_denom_new += expit(pnew);
            x_denom_old += expit(params[i]);
        }
        for (int i = 0; i < params.size(); ++i){
            double prob_old = expit(params[i]) / x_denom_old;
            double prob_new = expit(params_new[i]) / x_denom_new;
            //fprintf(stderr, "prob %d) %f -> %f\n", i, prob_old, prob_new);
            double pdelt = abs(prob_new - prob_old);
            if (pdelt > maxd){
                maxd = pdelt;
            }
            params[i] = params_new[i];
        }
        //fprintf(stderr, "max delta %f\n", maxd);
        fprintf(stderr, "mp LL %f -> %f\n", loglik_prev, loglik);
        //fprintf(stderr, "\n");
        
        //delta = maxd;
        delta = loglik - loglik_prev;
        prop_delta = maxd;
        loglik_prev = loglik;
        ++its;
    }
    
    fprintf(stderr, "mixPropOpt converged in %d iterations\n", its);

    // Now that it has converged, convert parameters to probabilities.
    double x_denom = 0.0;
    for (int i = 0; i < params.size(); ++i){
        x_denom += expit(params[i]);
    }
    this->results.clear();
    for (int i = 0; i < params.size(); ++i){
        this->results.push_back(expit(params[i]) / x_denom);
    }
    return loglik_prev;
}

