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
    rhs = 0.0;
    reconcile_func_set = false;
}

optimML::brent_solver::brent_solver(univar_func ll, univar_func dll){
    se_found = false;
    se = 0;    
    init(ll, dll);
    no_deriv = false;
    root = false;
    rhs = 0.0;
    reconcile_func_set = false;
}

optimML::brent_solver::brent_solver(univar_func ll, univar_func dll, univar_func dll2){
    se_found = false;
    se = 0;
    init(ll, dll, dll2);
    no_deriv = false;
    root = false;
    rhs = 0.0;
    reconcile_func_set = false;
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
 * Quadratic interpolation routine for root finding
 */
double optimML::brent_solver::quadfit2(bool& success){
    if (f_a == f_b || f_a == f_c || f_b == f_c){
        success = false;
        return 0;
    }
    if (a == c || a == b || b == c){
        success = false;
        return 0;
    }
    double x = f_b/f_c;
    double y = f_b/f_a;
    double z = f_a/f_c;
   
    double m = (c-a)/2.0;
    
    double num = y*((c-b)*z*(z-x) - (b-a)*(x-1));
    double denom = (z-1)*(x-1)*(y-1);
    if (denom == 0){
        success = false;
        return 0;
    }
    else{
        double step = num/denom;
        if (b + step > a && b + step < c){
            success = true;
            return step;
        }
        else{
            success = false;
            return 0.0;
        }
    }
}

/**
 * If finding a root of a function instead of maximizing it, this function
 * performs one iteration - the fallback is bisection, rather than
 * golden section search.
 */
bool optimML::brent_solver::interpolate_root(){
    if (f_b == 0 || c-b < xval_precision){
        return true;
    }
    if (b - a < xval_precision){
        return true;
    }
    bool interp_success;
    double step = quadfit2(interp_success);
    if (interp_success){
        double x = b + step;
        double f_x = eval_ll_x(x);
        if (x < b){
            if (f_x * f_a < 0){
                c = b;
                f_c = f_b;
                b = x;
                f_b = f_x;
            }
            else{
                a = x;
                f_a = f_x;
            }
        }
        else{
            if (f_x * f_c < 0){
                a = b;
                f_a = f_b;
                b = x;
                f_b = f_x;
            }
            else{
                c = x;
                f_c = f_x;
            }
        }
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
        step = (a+c)/2.0 - b;
        b += step;
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
 * Sets an optional constant (not dependent on data) to subtract before
 * finding root, at each iteration
 */
void optimML::brent_solver::set_root_rhs(double r){
    if (!this->root){
        fprintf(stderr, "ERROR: can only set RHS when finding a root, not maximum.\n");
        return;
    }
    this->rhs = r;
}

/**
 * Adds another evaluation function in the case of root finding. Output
 * from each function needs to be reconciled via a "reconciliation" function
 * that must be added.
 */
void optimML::brent_solver::add_root_function(univar_func f){
    if (!this->root){
        fprintf(stderr, "ERROR: can only add additional eval functions when root finding.\n");
        return;
    }
    if (this->additional_funcs.size() == 0){
        // If first, need to add an entry for the "default" eval function
        this->additional_funcs_out.push_back(0.0);
    }
    this->additional_funcs.push_back(f);
    this->additional_funcs_out.push_back(0.0);
}

/**
 * Adds the function needed to reconcile output from multiple functions
 * when root finding.
 */
void optimML::brent_solver::set_root_reconcile_function(reconcile_fun f){
    if (!this->root){
        fprintf(stderr, "ERROR: can only add a reconciliation function when root finding\n");
        fprintf(stderr, "  with multiple evaluation functions.\n");
        return;
    }
    this->reconcile_func = f;
    this->reconcile_func_set = true;
}

/**
 * Sets the solver to do maximization instead of root finding
 * Which method is used depends on whether derivative is supplied
 */
void optimML::brent_solver::set_max(){
    this->root = false;
}

/**
 * If we've tried solving (given initial bounds), and the interval
 * does not bracket a maximum, attempt to find a new interval that
 * does.
 */
bool optimML::brent_solver::bracket_max(int attempt_no){
    
    if (attempt_no > 10){
        return false;
    }

    // Define a step size
    double stepsize = (b-a);
    if (c-b < stepsize){
        stepsize = c-b;
    }
    // Make the next step 2x as big
    stepsize *= 2.0;
    
    if (f_c > f_a){
        // Move to the right.
        a = b;
        f_a = f_b;
        b = c;
        f_b = f_c;
        c = c + stepsize;
        f_c = eval_ll_x(c);
        if (f_b > f_a && f_b > f_c){
            return true;
        }
        else{
            // Try again
            return bracket_max(attempt_no+1);
        }
    }
    else{
        // Move to the left.
        c = b;
        f_c = f_b;
        b = a;
        f_b = f_a;
        a = a - stepsize;
        f_a = eval_ll_x(a);
        if (f_b > f_a && f_b > f_c){
            return true;
        }
        else{
            return bracket_max(attempt_no+1);
        }
    }
}

/**
 * If we've tried solving (given initial bounds), and the interval
 * does not bracket a root, attempt to find a new interval that does.
 */
bool optimML::brent_solver::bracket_root(int attempt_no){
    if (attempt_no > 10){
        return false;
    }

    // Define a step size
    double stepsize = (b-a);
    if (c-b < stepsize){
        stepsize = c-b;
    }
    // Make the next step 2x as big
    stepsize *= 2.0;
    
    if (abs(f_c) < abs(f_a)){
        // Look right
        a = b;
        f_a = f_b;
        b = c;
        f_b = f_c;
        c += stepsize;
        f_c = eval_ll_x(c);
        if (f_a * f_c < 0){
            return true;
        }
        else{
            return bracket_root(attempt_no+1);
        }
    }
    else{
        // Look left
        c = b;
        f_c = f_b;
        b = a;
        f_b = f_a;
        a -= stepsize;
        f_a = eval_ll_x(a);
        if (f_a * f_c < 0){
            return true;
        }
        else{
            return bracket_root(attempt_no+1);
        }
    }
}

/**
 * Override of parent routine that allows for additional stuff when root finding.
 * 
 * NOTE: none of this is applicable to maximization, and this cannot handle
 * fancy stuff when dealing with priors or derivatives.
 */
double optimML::brent_solver::eval_ll_x(double x){
    // Call parent routine
    double result = univar::eval_ll_x(x);
    
    if (this->root){
        // Check for additional stuff to do.
        if (this->additional_funcs.size() > 0){
            
            if (!this->reconcile_func_set){
                fprintf(stderr, "ERROR: multiple functions set for root finding, but\
 no function provided to reconcile their output. Please call set_root_reconcile_function()\
 before solving.\n");
                exit(1);
            }

            univar_func backup = ll_x;
            additional_funcs_out[0] = result;
            
            for (int i = 0; i < additional_funcs.size(); ++i){
                ll_x = additional_funcs[i];
                double val = univar::eval_ll_x(x);
                additional_funcs_out[i+1] = val;
            }

            ll_x = backup;

            // Reconcile outputs
            // Provide reconciliation function with current back-transformed version of
            // the independent variable, and a vector of output from each function
            // evaluated at the variable value.
            result = this->reconcile_func(x_t, additional_funcs_out);

            // Since we're root finding, subtract rhs
            result -= rhs;

        }
    }

    return result;
}

/**
 * Finds the solution (either a root or maximum) in the given interval.
 */
double optimML::brent_solver::solve(double lower, double upper, bool attempt_bracket){
    if (this->n_data == 0){
        // Check to see if the user added fixed data that can be transformed into regular data.
        if (!this->fixed_data_to_data()){
            fprintf(stderr, "ERROR: no data added\n");
            return 0.0;
        }
    }
    
    if (nthread > 0){
        create_threads();
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
    else if (this->trans_bounds && (lower == bound_low || upper == bound_high)){
        if (lower == bound_low){
            lower += this->xval_precision;
        }
        if (upper == bound_high){
            upper -= this->xval_precision;
        }
    }
    if ((this->trans_log && (lower <= 0 || upper <= 0)) ||
        (this->trans_logit && (lower <= 0 || upper >= 1)) ||
        (this->trans_bounds && (lower <= bound_low || upper >= bound_high))){
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
    else if (this->trans_bounds){
        // Transform variables
        a = log(lower-bound_low) - log(bound_high-lower);
        c = log(upper-bound_low) - log(bound_high-upper);
        b = log(b-bound_low) - log(bound_high-b);
        /*
        a = bound_low + (bound_high-bound_low)/(1.0 + exp(-a));
        b = bound_low + (bound_high-bound_low)/(1.0 + exp(-b));
        c = bound_low + (bound_high-bound_low)/(1.0 + exp(-c));
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
    else if (this->trans_logit || this->trans_bounds){
        // f(x) = 1/(1 + e^(-x)) -> df_dx = exp(-x) / ((exp(-x)) + 1)^2
        df_dt_a = exp(-a) / pow(((exp(-a)) + 1), 2);
        df_dt_b = exp(-b) / pow(((exp(-b)) + 1), 2);
        df_dt_c = exp(-c) / pow(((exp(-c)) + 1), 2);
    }
    if (this->trans_bounds){
        df_dt_a *= (bound_high-bound_low);
        df_dt_b *= (bound_high-bound_low);
        df_dt_c *= (bound_high-bound_low);
        /*
        df_dt_a = 1.0/(a-bound_low) + 1.0/(bound_high-a);
        df_dt_b = 1.0/(b-bound_low) + 1.0/(bound_high-b);
        df_dt_c = 1.0/(c-bound_low) + 1.0/(bound_high-c);
        */
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
            if (attempt_bracket){
                if (root){
                    root_in_interval = bracket_root(0);
                }
                else{
                    root_in_interval = bracket_max(0);
                }
            }
        }
        
        if (!root_in_interval){

            this->root_found = false;
            this->se_found = false;
            this->se = 0.0;

            // No root. Indicate failure and return whichever boundary
            // is better.

            double ll_a = eval_ll_x(a);
            double ll_c = eval_ll_x(c);
            
            double a_t = a;
            double c_t = c;
        
            if (this->trans_log){
                a_t = exp(a);
                c_t = exp(c);
            }
            else if (this->trans_logit || this->trans_bounds){
                a_t = expit(a);
                c_t = expit(c);
            }
            if (this->trans_bounds){
                a_t *= (bound_high-bound_low);
                a_t += bound_low;
                c_t *= (bound_high-bound_low);
                c_t += bound_low;
                /*
                a_t = log(a - bound_low) - log(bound_high - a);
                c_t = log(c - bound_low) - log(bound_high - c);
                */
            }
            
            if (root){
                // Choose lower function eval (looking for zero)
                if (ll_a < ll_c){
                    this->log_likelihood = ll_a;
                    return a_t;
                }
                else{
                    this->log_likelihood = ll_c;
                    return c_t;
                }
            }
            else{
                // Choose higher function eval (looking for maximum)
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
    }

    if (!root){ 
        // If root, we already calculated these earlier.
        f_a = eval_ll_x(a);
        f_b = eval_ll_x(b);
        f_c = eval_ll_x(c);
        
        if (no_deriv){
            // If here, we have not yet checked to make sure a maximum exists.
            if (!(f_b > f_a && f_b > f_c)){
                // Need to bracket.
                bool bracket_found = false;
                if (attempt_bracket){
                    bracket_found = bracket_max(0);
                }
                if (!bracket_found){
                    // Bail.
         
                    this->root_found = false;
                    this->se_found = false;
                    this->se = 0.0;

                    // No root. Indicate failure and return whichever boundary
                    // is better.

                    double ll_a = eval_ll_x(a);
                    double ll_c = eval_ll_x(c);
                    
                    double a_t = a;
                    double c_t = c;
                
                    if (this->trans_log){
                        a_t = exp(a);
                        c_t = exp(c);
                    }
                    else if (this->trans_logit || this->trans_bounds){
                        a_t = expit(a);
                        c_t = expit(c);
                    }
                    if (this->trans_bounds){
                        a_t *= (bound_high-bound_low);
                        a_t += bound_low;
                        c_t *= (bound_high-bound_low);
                        c_t += bound_low;
                        //a_t = log(a - bound_low) - log(bound_high - a);
                        //c_t = log(c - bound_low) - log(bound_high - c);
                    }
                
                    // Choose higher function eval (looking for maximum)
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
        }
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
    else if (this->trans_logit || this->trans_bounds){
        b_t = expit(b);
    }
    if (this->trans_bounds){
        b_t *= (bound_high-bound_low);
        b_t += bound_low;
        //b_t = log(b - bound_low) - log(bound_high - b);
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

