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

optimML::brent_solver::brent_solver(univar_func ll, univar_func dll){
    se_found = false;
    se = 0;    
    init(ll, dll);
}

optimML::brent_solver::brent_solver(univar_func ll, univar_func dll, univar_func dll2){
    se_found = false;
    se = 0;
    init(ll, dll, dll2);
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
double optimML::brent_solver::solve(double lower, double upper){
    if (this->n_data == 0){
        fprintf(stderr, "ERROR: no data added\n");
        return 0.0;
    }

    // Attempt to make interval feasible if transformations are being used.
    if (this->trans_log && lower == 0){
        lower += this->xval_precision;
    }
    else if (this->trans_logit && (lower == 0 || upper == 1)){
        if (lower == 0){
            lower += this->xval_precision;
        }
        if (upper == 1){
            upper -= this->xval_precision;
        }
    }
    if ((this->trans_log && (lower <= 0 || upper <= 0)) ||
        (this->trans_logit && (lower <= 0 || upper >= 1))){
        fprintf(stderr, "ERROR: bounds of (%f, %f) are not compatible with the \
transformation of the data.\n", lower, upper);
        this->root_found = false;
        this->se_found = false;
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
        b = log(b);
        /*
        if (a > c){
            double tmp = a;
            a = c;
            c = tmp;
        }
        b = (a+c)/2.0;
        */
    }
    else if (this->trans_logit){
        // Each variable is logit(x), operate on expit(x)
        a = logit(lower);
        c = logit(upper);
        b = logit(b);
        //b = (a+c)/2.0;
        //b = logit(b);
        /*
        if (a > c){
            double tmp = a;
            a = c;
            c = tmp;
        }
        b = (a+c)/2.0;
        */
    }

    double delta = 999;
    int nits = 0;
    
    double df_dt_a = 1.0;
    double df_dt_b = 1.0;
    double df_dt_c = 1.0;

    if (this->trans_log){
        // f(x) = e^x -> df_dx = e^x
        df_dt_a = exp(a);
        df_dt_b = exp(b);
        df_dt_c = exp(c);
    }
    else if (this->trans_logit){
        // f(x) = 1/(1 + e^(-x)) -> df_dx = exp(-x) / ((exp(-x)) + 1)^2
        df_dt_a = exp(-a) / pow(((exp(-a)) + 1), 2);
        df_dt_b = exp(-b) / pow(((exp(-b)) + 1), 2);
        df_dt_c = exp(-c) / pow(((exp(-c)) + 1), 2);
    }
    
    double f_a = eval_dll_dx(a);
    double f_b = eval_dll_dx(b);
    double f_c = eval_dll_dx(c);
    
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
        this->se = 0.0;

        // No root. Pick whichever boundary has lower log likelihood.
        double ll_a = eval_ll_x(a);
        double ll_c = eval_ll_x(c);
        
        double a_t = a;
        double c_t = c;
    
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
    double b_t = b;

    // Transform back, if necessary
    if (this->trans_log){
        b_t = exp(b);
    }
    else if (this->trans_logit){
        b_t = expit(b);
    }

    if (this->has_d2ll_dx2){
        
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

