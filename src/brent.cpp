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
#include "functions.h"
#include "brent.h"

using std::cout;
using std::endl;
using namespace std;

// ----- Brent's method root finder

/**
 * Helper for truncated normal LL calculations
 */
double brentSolver::phi(double x){
    return  0.5*(1 + erf(x / sqrt(2)));
}

/**
 * Helper for normal LL calculations
 */
double brentSolver::dnorm(double x, double mu, double sigma){
    static double sq2pi = sqrt(2.0 * M_PI);
    return -0.5 * pow((x-mu)/sigma, 2) - log(sigma * sq2pi);
}

/**
 * Built-in prior function for input variables: (truncated) normal
 */
double brentSolver::ll_prior_normal(double x, map<string, double>& params_d,
    map<string, int>& params_i){
    bool trunc = false;
    double trunc_low;
    double trunc_high;
    if (params_d.count("a") > 0 && params_d.count("b") > 0){
        trunc_low = params_d["a"];
        trunc_high = params_d["b"];
        trunc = true;
    }
    if (trunc){
        if (x < trunc_low || x > trunc_high){
            return log(0);
        }
    }
    double mu = params_d["mu"];
    double sigma = params_d["sigma"];
    return dnorm(x, mu, sigma) - log(phi((1.0-mu)/sigma) - phi((0.0-mu)/sigma));
}

double brentSolver::dll_prior_normal(double x, map<string, double>& params_d,
    map<string, int>& params_i){
    // Truncation no longer matters for derivatives wrt independent variable
    double mu = params_d["mu"];
    double sigma = params_d["sigma"];
    return -((x-mu)/(sigma*sigma));
}

double brentSolver::d2ll_prior_normal(double x, map<string, double>& params_d,
    map<string, int>& params_i){
    double mu = params_d["mu"];
    double sigma = params_d["sigma"];
    return -1.0/(sigma*sigma);
}

double brentSolver::ll_prior_beta(double x, map<string, double>& params_d,
    map<string, int>& params_i){
    double a = params_d["alpha"];
    double b = params_d["beta"];
    return (a-1.0)*log(x) + (b-1)*log(1.0-x) - (lgamma(a) + lgamma(b) - lgamma(a+b));
}

double brentSolver::dll_prior_beta(double x, map<string, double>& params_d,
    map<string, int>& params_i){
    double a = params_d["alpha"];
    double b = params_d["beta"];
    return (a-1.0)/x - (b-1.0)/(1.0-x);
}

double brentSolver::d2ll_prior_beta(double x, map<string, double>& params_d,
    map<string, int>& params_i){
    double a = params_d["alpha"];
    double b = params_d["beta"];
    return (1.0-a)/(x*x) + (1.0-b)/((1.0-x)*(1.0-x));
}


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
    this->log_likelihood = 0.0;
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

void brentSolver::add_normal_prior(double mu, double sigma){
    add_prior(ll_prior_normal, dll_prior_normal, d2ll_prior_normal);
    add_prior_param("mu", mu);
    add_prior_param("sigma", sigma);
}

/**
 * Truncated normal distribution on (a, b)
 */
void brentSolver::add_normal_prior(double mu, double sigma, double a, double b){
    add_prior(ll_prior_normal, dll_prior_normal, d2ll_prior_normal);
    add_prior_param("mu", mu);
    add_prior_param("sigma", sigma);
    add_prior_param("a", a);
    add_prior_param("b", b);
}

void brentSolver::add_beta_prior(double alpha, double beta){
    add_prior(ll_prior_beta, dll_prior_beta, d2ll_prior_beta);
    add_prior_param("alpha", alpha);
    add_prior_param("beta", beta);
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
bool brentSolver::add_data(string name, std::vector<double>& dat){
    // Make sure dimensions agree
    int nd = dat.size();
    if (this->n_data != 0 && this->n_data != nd){
        fprintf(stderr, "ERROR: data vectors do not have same dimensions\n");
        return false;
    }
    if (param_double_cur.count(name) > 0){
        //fprintf(stderr, "WARNING: %s already keyed to data. Overwriting\n", name.c_str());
        int idx = -1;
        for (int i = 0; i < params_double_names.size(); ++i){
            if (params_double_names[i] == name){
                idx = i;
                break;
            }
        }
        params_double_vals[idx] = dat.data();
        //return false;
        return true;
    }
    this->n_data = nd;
    this->params_double_names.push_back(name);
    this->params_double_vals.push_back(dat.data());
    this->param_double_cur.insert(make_pair(name, 0.0));
    this->param_double_ptr.push_back(&(this->param_double_cur.at(name)));
    return true;
}

bool brentSolver::add_data(string name, std::vector<int>& dat){
    // Make sure dimensions agree
    int nd = dat.size();
    if (this->n_data != 0 && this->n_data != nd){
        fprintf(stderr, "ERROR: data vectors do not have same dimensions\n");
        return false;
    }
    if (this->param_int_cur.count(name) > 0){
        fprintf(stderr, "ERROR: %s already keyed to data\n", name.c_str());
        return false;
    }
    this->n_data = nd;
    this->params_int_names.push_back(name);
    this->params_int_vals.push_back(dat.data());
    this->param_int_cur.insert(make_pair(name, 0.0));
    this->param_int_ptr.push_back(&(this->param_int_cur.at(name)));
    return true;
}

bool brentSolver::add_data_fixed(string name, double dat){
    if (param_double_cur.count(name) > 0){
        fprintf(stderr, "ERROR: %s already keyed to data\n", name.c_str());
        return false;
    }
    param_double_cur.insert(make_pair(name, dat));
    return true;
}

bool brentSolver::add_data_fixed(string name, int dat){
    if (param_int_cur.count(name) > 0){
        fprintf(stderr, "ERROR: %s already keyed to data\n", name.c_str());
        return false;
    }
    param_int_cur.insert(make_pair(name, dat));
    return true;
}

bool brentSolver::add_weights(vector<double>& weights){
    if (this->n_data != 0 && this->n_data != weights.size()){
        fprintf(stderr, "ERROR: dimension of weights does not equal dimension of data\n");
        return false;
    }
    this->weights = weights;
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
    if (trans_log && (lower <= 0)){
        fprintf(stderr, "ERROR: range given is out of bounds for log transformation\n");
        return;
    }
    else if (trans_logit && (lower <= 0 || upper >= 1)){
        fprintf(stderr, "ERROR: range given is out of bounds for logit transformation\n");
        return;
    }
    for (double x = lower; x <= upper; x += step){
        double x_t = x;
        if (this->trans_log){
            x_t = log(x);
        }
        else if (this->trans_logit){
            x_t = logit(x);
        }
        double ll = eval_ll_x(x_t);
        double deriv1 = eval_dll_dx(x_t);
        /*
        if (this->trans_log){
            double df_dt_x = exp(x);
            deriv1 /= df_dt_x;
        }
        else if (this->trans_logit){
            double df_dt_x = exp(-x) / pow(((exp(-x)) + 1), 2);
            deriv1 /= df_dt_x;
        }
        */
        if (this->has_d2ll){
            double deriv2 = eval_d2ll_dx2(x_t);
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
        double llx = ll_x(x_t, this->param_double_cur, this->param_int_cur);
        if (isnan(llx) || isinf(llx)){
            fprintf(stderr, "ERROR: nan or inf from log likelihood function\n");
            fprintf(stderr, "parameter: %f\n", x_t);
            fprintf(stderr, "data:\n");
            for (map<string, double>::iterator it = param_double_cur.begin(); it != 
                param_double_cur.end(); ++it){
                fprintf(stderr, "%s = %f\n", it->first.c_str(), it->second);
            }
            for (map<string, int>::iterator it = param_int_cur.begin(); it != 
                param_int_cur.end(); ++it){
                fprintf(stderr, "%s = %d\n", it->first.c_str(), it->second);
            }
            exit(1);
        }
        if (this->weights.size() > 0){
            llx *= weights[i];
        }
        f_x += llx;
    }
    if (this->has_prior){
        double llx = ll_x_prior(x_t, this->params_prior_double, this->params_prior_int);
        if (isinf(llx) || isnan(llx)){
            fprintf(stderr, "ERROR: illegal value from prior function\n");
            fprintf(stderr, "parameter: %f\n", x_t);
            fprintf(stderr, "prior dist parameters:\n");
            for (map<string, double>::iterator it = params_prior_double.begin(); it != 
                params_prior_double.end(); ++it){
                fprintf(stderr, "%s = %f\n", it->first.c_str(), it->second);
            }
            for (map<string, int>::iterator it = params_prior_int.begin(); it != 
                params_prior_int.end(); ++it){
                fprintf(stderr, "%s = %d\n", it->first.c_str(), it->second);
            }
            exit(1);
        }
        f_x += llx;
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
        double w = 1.0;
        if (this->weights.size() > 0){
            w = this->weights[i];
        }
        f_x += dll_dx(x_t, this->param_double_cur, this->param_int_cur) * w * df_dt_x;
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
        double w = 1.0;
        if (this->weights.size() > 0){
            w = this->weights[i];
        }
        f_x += d2ll_dx2(x_t, this->param_double_cur, this->param_int_cur) * w * d2f_dt2_x;
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
    if (this->n_data == 0){
        fprintf(stderr, "ERROR: no data added\n");
        return 0.0;
    }

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
        this->log_likelihood = 0.0;
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
            this->log_likelihood = ll_a;
            return a_t;
        }
        else{
            this->log_likelihood = ll_c;
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
    this->log_likelihood = eval_ll_x(b);
    return b_t;
}

