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
#include <climits>
#include "functions.h"
#include "brent.h"

using std::cout;
using std::endl;
using namespace std;

// ----- Brent's method root finder

optimML::brent_solver::brent_solver(univar_func ll){
    se_found = false;
    se = 0;
    init(ll);
    no_deriv = true;
    root = false;
}

optimML::brent_solver::brent_solver(univar_func ll, univar_func dll){
    se_found = false;
    se = 0;    
    init(ll, dll);
    no_deriv = false;
    root = false;
}

optimML::brent_solver::brent_solver(univar_func ll, univar_func dll, univar_func dll2){
    se_found = false;
    se = 0;
    init(ll, dll, dll2);
    no_deriv = false;
    root = false;
}

/**
 * One iteration of golden section search.
 *
 * This is used when maximizing without derivative information.
 *
 */
bool optimML::brent_solver::golden(){
    static double const1 = (3.0 - sqrt(5))/2.0;
    static double const2 = (sqrt(5) - 1.0)/2.0;
    
    double x;
    if ( b < (a+c)/2.0){
        x = a + const1*(c-a);
    }
    else{
        x = a + const2*(c-a);
    } 
    
    if (abs(x-b) < xval_precision){
        b = (x+b)/2.0;
        return true;
    }
    else{
        double f_x = eval_ll_x(x);
        
        step_2ago = step_prev;

        if (b < x){
            if (f_b > f_x){
                // (a,b,x)
                step_prev = c-x;
                c = x;
                f_c = f_x;
            }
            else{
                // (b,x,c)
                step_prev = b-a;
                a = b;
                f_a = f_b;
                b = x;
                f_b = f_x;
            }
        }
        else{
            if (f_b > f_x){
                // (x,b,c)
                step_prev = x-a;
                a = x;
                f_a = f_x;
            }
            else{
                // (a,x,b)
                step_prev = c-b;
                c = b;
                f_c = f_b;
                b = x;
                f_b = f_x;
            }
        }
        return false;
        
    }
}

/**
 * Fits a quadratic polynomial and finds a candidate step to take
 */
double optimML::brent_solver::quadfit(bool& success){
    
    double x = f_b/f_c;
    double y = f_b/f_a;
    double z = f_a/f_c;
    double num = y*(z*(x-z)*(c-b) - (1.0-x)*(b-a));
    double denom = (z-1.0)*(x-1.0)*(y-1.0);
    
    if (denom == 0){
        success = false;
        return 0;
    }
    else{
        double new_b = b + num/denom;
        if (new_b > a && new_b < c){
            success = true;
            return num/denom;
        }
        else{
            success = false;
            return 0;
        }
    }
}

/**
 * If finding a root of a function instead of maximizing it, this function
 * performs one iteration - the fallback is bisection, rather than
 * golden section search.
 */
bool optimML::brent_solver::interpolate_root(){
    if (b - a < xval_precision){
        return true;
    }
    bool interp_success;
    double step = quadfit(interp_success);
    if (interp_success){
        double x = b + step;
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
        f_b = eval_ll_x(b);
    }
    else{
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
        step = b - (a+c)/2.0 + a;
        b = (a+c)/2.0;
        f_b = eval_ll_x(b);
    }
    if (abs(step) < xval_precision){
        return true;
    }
    return false;
}

/**
 * One iteration of Brent's method for maximization, without derivative
 * information, which uses golden section search as a fallback.
 */
bool optimML::brent_solver::interpolate(){
    if (b - a < xval_precision){
        return true;
    }
    bool interp_success;
    double step = -quadfit(interp_success);
    if (interp_success && (step_2ago == 0.0 || abs(step) < 0.5*step_2ago)){
        double x = step + b;
        if (abs(x-b) < xval_precision){
            b = (x + b)/2.0;
            return true;
        }
        else{
            double f_x = eval_ll_x(x);
            step_2ago = step_prev;
            if (x < b){
                if (f_b > f_x){
                    // (x,b,c)
                    step_prev = x-a;
                    a = x;
                    f_a = f_x;
                }
                else{
                    // (a,x,b)
                    step_prev = c-b;
                    c = b;
                    f_c = f_b;
                    b = x;
                    f_b = f_x;
                }
            }
            else{
                if (f_b > f_x){
                    // (a,b,x)
                    step_prev = c-x;
                    c = x;
                    f_c = f_x;
                }
                else{
                    // (b,x,c)
                    step_prev = b-a;
                    a = b;
                    f_a = f_b;
                    b = x;
                    f_b = f_x;
                }
            }
            return false;
        }
    }
    else{
        // Conditions not met. Fall back to golden section search.
        if (this->golden()){
            return true;
        }
        return false;
    }
    return false;
}

/**
 * One iteration of Brent's method for maximization using first derivative information
 */
bool optimML::brent_solver::interpolate_der(){
    
    if (b - a < xval_precision){
        return true;
    }

    bool interp_success;
    
    double step = -quadfit(interp_success);
    
    if (interp_success && (step_2ago == 0.0 || abs(step) < 0.5*step_2ago)){
        double x = step + b;
        double f_x = eval_ll_x(x);
        if (abs(x-b) < xval_precision){
            b = (x + b)/2.0;
            return true;
        }
        else{
            step_2ago = step_prev;
            
            if (x < b){
                if (f_b > f_x){
                    // (x,b,c)
                    step_prev = x-a;
                    a = x;
                    f_a = f_x;
                }
                else{
                    // (a,x,b)
                    step_prev = c-b;
                    c = b;
                    f_c = f_b;
                    b = x;
                    f_b = f_x;
                }
            }
            else{
                if (f_b > f_x){
                    // (a,b,x)
                    step_prev = c-x;
                    c = x;
                    f_c = f_x;
                }
                else{
                    // (b,x,c)
                    step_prev = b-a;
                    a = b;
                    f_a = f_b;
                    b = x;
                    f_b = f_x;
                }
            }
            return false;
        }
    }
    else{
        // Conditions not met. Compute derivative and use it to guide which of the two
        // intervals (bisection method) to search.
        double df_db = eval_dll_dx(b);
        step_2ago = step_prev;

        if (df_db < 0){
            // Interval (a,b)
            // (a,x,b)
            step_prev = c-b;
            double x = (a+b)/2.0;
            double f_x = eval_ll_x(x);
            c = b;
            b = x;
            f_c = f_b;
            f_b = f_x;
        }
        else{
            // Interval (b,c)
            // (b,x,c)
            step_prev = b-a;
            double x = (b+c)/2.0;
            double f_x = eval_ll_x(x);
            a = b;
            b = x;
            f_a = f_b;
            f_b = f_x;
        }
        return false;
    }
}

/**
 * Sets the solver to do root finding (of first/LL function) instead of
 * maximization
 */
void optimML::brent_solver::set_root(){
    this->root = true;
}

/**
 * Sets the solver to do maximization instead of root finding
 * Which method is used depends on whether derivative is supplied
 */
void optimML::brent_solver::set_max(){
    this->root = false;
}

/**
 * Finds the solution (either a root or maximum) in the given interval.
 */
double optimML::brent_solver::solve(double lower, double upper){
    if (this->n_data == 0){
        // Check to see if the user added fixed data that can be transformed into regular data.
        if (!this->fixed_data_to_data()){
            fprintf(stderr, "ERROR: no data added\n");
            return 0.0;
        }
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
    
    // Initial bounds (a & c) and guess of maximum (b)
    a = lower;
    b = (lower+upper)/2.0;
    c = upper;
    
    if (this->trans_log){
        // each variable is log(x), operate on e(x)
        a = log(lower);
        c = log(upper);
        b = log(b);
    }
    else if (this->trans_logit){
        // Each variable is logit(x), operate on expit(x)
        a = logit(lower);
        c = logit(upper);
        b = logit(b);
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
    
    if (root || !no_deriv){
        if (root){
            f_a = eval_ll_x(a);
            f_b = eval_ll_x(b);
            f_c = eval_ll_x(c);
        }
        else{
            f_a = eval_dll_dx(a);
            f_b = eval_dll_dx(b);
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
    }

    if (!root){ 
        // If root, we already calculated these earlier.
        f_a = eval_ll_x(a);
        f_b = eval_ll_x(b);
        f_c = eval_ll_x(c);
    }

    bool converged = false;
    step_prev = 0;
    step_2ago = 0;

    while (!converged && (maxiter < 0 || nits < maxiter)){
        if (root){
            // We're trying to find the root of the function, with no derivative.
            converged = interpolate_root();
        }
        else{
            if (no_deriv){
                // Use method that does not rely on derivative
                converged = interpolate();
            }
            else{
                // Use derivate information
                converged = interpolate_der();
            }
        }
        ++nits;
    }
    //fprintf(stderr, "nits %d\n", nits);
    
    this->root_found = converged;
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
