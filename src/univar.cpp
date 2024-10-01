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
#include "univar.h"

using std::cout;
using std::endl;
using namespace std;

namespace optimML{
    
    /**
     * Constructor
     */
    univar::univar(){
        root_found = false;
        has_dll_dx = false;
        has_d2ll_dx2 = false;
        has_prior = false;
        trans_log = false;
        trans_logit = false;
       
        xval_precision = 1e-8;
        initialized = false;

        cur_ll_x = 0;
        cur_dll_dx = 0;
        cur_d2ll_dx2 = 0;
        
        threads_init = false; 
        thread_compute_ll = false;
        thread_compute_dll = false;
        thread_compute_d2ll = false;
    }

    univar::~univar(){
        if (threads_init){
            delete ll_mutex;
            delete dll_mutex;
            delete d2ll_mutex;
        }
    }

    void univar::init(univar_func ll_x){
        this->ll_x = ll_x;
        initialized = true; 
    }

    void univar::init(univar_func ll_x, univar_func dll_dx){
        this->ll_x = ll_x;
        this->dll_dx = dll_dx;
        this->has_dll_dx = true;
        initialized = true;
    }

    void univar::init(univar_func ll_x, univar_func dll_dx, univar_func d2ll_dx2){
        this->ll_x = ll_x;
        this->dll_dx = dll_dx;
        this->d2ll_dx2 = d2ll_dx2;
        this->has_dll_dx = true;
        this->has_d2ll_dx2 = true;
        initialized = true;
    }
    
    void univar::dump_cur_params(){
        fprintf(stderr, "data:\n");
        for (map<string, double>::iterator it = param_double_cur.begin(); it != 
            param_double_cur.end(); ++it){
            fprintf(stderr, "%s = %f\n", it->first.c_str(), it->second);
        }
        for (map<string, int>::iterator it = param_int_cur.begin(); it != 
            param_int_cur.end(); ++it){
            fprintf(stderr, "%s = %d\n", it->first.c_str(), it->second);
        }
        throw optimML::OPTIMML_MATH_ERR; 
    }
    
    void univar::dump_prior_params(){
        fprintf(stderr, "prior params:\n");
        for (map<string, double>::iterator it = params_prior_double.begin(); it != 
            params_prior_double.end(); ++it){
            fprintf(stderr, "%s = %f\n", it->first.c_str(), it->second);
        }
        for (map<string, int>::iterator it = params_prior_int.begin(); it != 
            params_prior_int.end(); ++it){
            fprintf(stderr, "%s = %d\n", it->first.c_str(), it->second);
        }
        throw optimML::OPTIMML_MATH_ERR;
    }
    
    double univar::eval_ll_x(double x){
        eval_funcs(x, true, false, false);
        return cur_ll_x;
    }

    double univar::eval_dll_dx(double x){
        eval_funcs(x, false, true, false);
        return cur_dll_dx;
    }

    double univar::eval_d2ll_dx2(double x){
        eval_funcs(x, false, false, true);
        return cur_d2ll_dx2;
    }

    /**
     * Evaluate all functions at a given value.
     */
    void univar::eval_funcs(double x, bool eval_ll_x, bool eval_dll_dx, bool eval_d2ll_dx2){
        if (eval_dll_dx && !has_dll_dx){
            fprintf(stderr, "ERROR: cannot compute first derivative; not given\n");
            exit(1);
        }
        else if (eval_d2ll_dx2 && !has_d2ll_dx2){
            fprintf(stderr, "ERROR: cannot compute second derivative; not given\n");
            exit(1);
        }

        x_t = x;
        df_dt_x = 1.0; 
        d2f_dt2_x = 1.0; 
        
        if (this->trans_log){
            x_t = exp(x);
            // f(x) = e^x -> df_dx = e^x
            df_dt_x = x_t;
            // f(x) = e^x -> d2f_dx2 = e^x
            d2f_dt2_x = x_t;
        }
        else if (this->trans_logit){
            x_t = expit(x);
            // f(x) = 1/(1 + e^(-x)) -> df_dx = exp(-x) / ((exp(-x)) + 1)^2
            df_dt_x = exp(-x) / pow(((exp(-x)) + 1), 2);
            // f(x) = 1/(1 + e^(-x)) -> d2f_dx2 = -(exp(x)*(exp(x) - 1))/(exp(x) + 1)^3
            double e_x = exp(x);
            d2f_dt2_x = (e_x*(e_x - 1.0))/pow(e_x + 1.0, 3);
        }
        
        cur_ll_x = 0;
        cur_dll_dx = 0;
        cur_d2ll_dx2 = 0;
        
        if (nthread > 0){
            thread_compute_ll = eval_ll_x;
            thread_compute_dll = eval_dll_dx;
            thread_compute_d2ll = eval_d2ll_dx2;
            launch_threads();
        }

        for (int i = 0; i < this->n_data; ++i){
            
            if (nthread > 0){
                this->add_job(i);
            }
            else{
                this->prepare_data(i);
                
                double w = 1.0;
                if (this->weights.size() > 0){
                    w = weights[i];
                }
                
                if (eval_ll_x){
                    double y = ll_x(x_t, this->param_double_cur, this->param_int_cur);
                    if (isnan(y) || isinf(y)){
                        fprintf(stderr, "ERROR: nan or inf from log likelihood function\n");
                        fprintf(stderr, "parameter: %f\n", x_t);
                        dump_cur_params(); 
                    }
                    cur_ll_x += y * w;
                }
                if (eval_dll_dx && this->has_dll_dx){
                    double yprime = dll_dx(x_t, this->param_double_cur, this->param_int_cur);
                    if (isnan(yprime) || isinf(yprime)){
                        fprintf(stderr, "ERROR: nan or inf from derivative LL function\n");
                        fprintf(stderr, "parameter: %f\n", x_t);
                        dump_cur_params(); 
                    }
                    cur_dll_dx += yprime * w * df_dt_x;
                }
                if (eval_d2ll_dx2 && this->has_d2ll_dx2){
                    double yprime2 = d2ll_dx2(x_t, this->param_double_cur, this->param_int_cur);
                    if (isnan(yprime2) || isinf(yprime2)){
                        fprintf(stderr, "ERROR: nan or inf from 2nd derivative LL function\n");
                        fprintf(stderr, "parameter: %f\n", x_t);
                        dump_cur_params();  
                    }
                    cur_d2ll_dx2 += yprime2 * w * d2f_dt2_x;
                }
            }
        }
        
        if (nthread > 0){
            close_pool();
            cur_ll_x += ll_threads;
            cur_dll_dx += dll_threads;
            cur_d2ll_dx2 += d2ll_threads;
        } 

        // Handle priors
        if (this->has_prior){
            if (eval_ll_x){
                double yprior = ll_x_prior(x_t, this->params_prior_double, this->params_prior_int);
                if (isinf(yprior) || isnan(yprior)){
                    fprintf(stderr, "ERROR: illegal value from prior function\n");
                    fprintf(stderr, "parameter: %f\n", x_t);
                    dump_prior_params();
                }
                cur_ll_x += yprior;
            }
            if (eval_dll_dx && this->has_dll_dx){
                double yprime_prior = dll_dx_prior(x_t, this->params_prior_double, this->params_prior_int);
                if (isinf(yprime_prior) || isnan(yprime_prior)){
                    fprintf(stderr, "ERROR: illegal value from first derivative prior function\n");
                    fprintf(stderr, "parameter: %f\n", x_t);
                    dump_prior_params();
                }
                cur_dll_dx += yprime_prior * df_dt_x;
            }
            if (eval_d2ll_dx2 && this->has_d2ll_dx2){
                
                double yprime2_prior = d2ll_dx2_prior(x_t, this->params_prior_double, this->params_prior_int);
                if (isinf(yprime2_prior) || isnan(yprime2_prior)){
                    fprintf(stderr, "ERROR: illegal value from second derivative prior function\n");
                    fprintf(stderr, "parameter: %f\n", x_t);
                    dump_prior_params();  
                } 
                cur_d2ll_dx2 += yprime2_prior * d2f_dt2_x; 
            }
        }
    }
     
    void univar::add_prior(univar_func ll){
        if (this->has_dll_dx || this->has_d2ll_dx2){
            fprintf(stderr, "WARNING: function has derivatives but no derivative \
supplied for prior. Derivatives will not be calculated.\n");
            this->has_dll_dx = false;
            this->has_d2ll_dx2 = false;
        }
        this->ll_x_prior = ll;
        this->has_prior = true;
    }

    void univar::add_prior(univar_func ll, univar_func dll){
        if (this->has_d2ll_dx2){
            fprintf(stderr, "WARNING: function has second derivative but none supplied \
for prior. Second derivatives will not be calculated.\n");
            this->has_d2ll_dx2 = false;
        }
        if (this->has_dll_dx){
            this->dll_dx_prior = dll;
        }
        this->ll_x_prior = ll;
        this->has_prior = true;
    }
    
    void univar::add_prior(univar_func ll, univar_func dll, univar_func d2ll){
        if (this->has_dll_dx){
            this->dll_dx_prior = dll;
        }
        if (this->has_d2ll_dx2){
            this->d2ll_dx2_prior = d2ll;
        }
        this->ll_x_prior = ll;
        this->has_prior = true;
    }

    // Add in data for prior
    bool univar::add_prior_param(string name, double dat){
        this->params_prior_double.insert(make_pair(name, dat));
        return true;
    }
    bool univar::add_prior_param(string name, int dat){
        this->params_prior_int.insert(make_pair(name, dat));
        return true;
    }

    /**
     * Normal distribution
     */
    void univar::add_normal_prior(double mu, double sigma){
        add_prior(ll_prior_normal, dll_prior_normal, d2ll_prior_normal);
        add_prior_param("mu", mu);
        add_prior_param("sigma", sigma);
    }
     
    /**
     * Truncated normal distribution on (a, b)
     */
    void univar::add_normal_prior(double mu, double sigma, double a, double b){
        add_prior(ll_prior_normal, dll_prior_normal, d2ll_prior_normal);
        add_prior_param("mu", mu);
        add_prior_param("sigma", sigma);
        add_prior_param("a", a);
        add_prior_param("b", b);
    }

    void univar::add_beta_prior(double alpha, double beta){
        add_prior(ll_prior_beta, dll_prior_beta, d2ll_prior_beta);
        add_prior_param("alpha", alpha);
        add_prior_param("beta", beta);
    }
     
    // Constrain independent variable to be positive
    void univar::constrain_pos(){
        this->trans_log = true;
        this->trans_logit = false;
    }

    // Constrain independent variable to be in (0,1)
    void univar::constrain_01(){
        this->trans_log = false;
        this->trans_logit = true;
    }
    
    // Change the tolerance for stopping iteration.
    void univar::set_epsilon(double e){
        xval_precision = e;
    }
    
    void univar::create_threads(){
        if (threads_init){
            fprintf(stderr, "ERROR: threads already initialized.\n");
            exit(1);
        }
        
        // Set up data stuff for threads
        solver::create_threads();
        
        ll_mutex = new mutex;
        dll_mutex = new mutex;
        d2ll_mutex = new mutex;

        threads_init = true;
    }
    
    void univar::launch_threads(){
        solver::launch_threads();
        ll_threads = 0.0;
        dll_threads = 0.0;
        d2ll_threads = 0.0;
    }
    
    void univar::worker(int thread_idx){
        while(true){
            
            //if (this->job_inds.size() == 0 && this->terminate_threads){
            //    return;
            //}
            int jid = get_next_job();
            if (jid == -1){
                return;
            }
            
            // Prepare data (thread-specific)
            prepare_data(jid, thread_idx);
            
            double w = 1.0;
            if (this->weights.size() > 0){
               w = weights[jid];
            }

            if (thread_compute_ll){
                double y = ll_x(x_t, this->params_double_cur_thread[thread_idx], 
                    this->params_int_cur_thread[thread_idx]);
                if (isnan(y) || isinf(y)){
                    fprintf(stderr, "ERROR: nan or inf from log likelihood function\n");
                    fprintf(stderr, "parameter: %f\n", x_t);
                    dump_cur_params(); 
                }
                unique_lock<mutex> lock(*ll_mutex);
                ll_threads += y * w;
            }
            if (thread_compute_dll){
                double yprime = dll_dx(x_t, this->params_double_cur_thread[thread_idx], 
                    this->params_int_cur_thread[thread_idx]);

                if (isnan(yprime) || isinf(yprime)){
                    fprintf(stderr, "ERROR: nan or inf from derivative LL function\n");
                    fprintf(stderr, "parameter: %f\n", x_t);
                    dump_cur_params();
                }
                unique_lock<mutex> lock(*dll_mutex);
                dll_threads += yprime * w * df_dt_x;
            }
            if (thread_compute_d2ll){
                double yprime2 = d2ll_dx2(x_t, this->params_double_cur_thread[thread_idx], 
                    this->params_int_cur_thread[thread_idx]);

                if (isnan(yprime2) || isinf(yprime2)){
                    fprintf(stderr, "ERROR: nan or inf from 2nd derivative LL function\n");
                    fprintf(stderr, "parameter: %f\n", x_t);
                    dump_cur_params();  
                }
                unique_lock<mutex> lock(*d2ll_mutex);
                d2ll_threads += yprime2 * w * d2f_dt2_x;
            }
        }
    }

    /**
     * For debugging: prints log likelihood, derivative, and possibly second derivative
     * over the specified range.
     */
    void univar::print(double lower, double upper, double step){
        if (trans_log && (lower <= 0)){
            lower = xval_precision;
        }
        if (trans_logit && lower <= 0){
            lower = xval_precision;
        }
        if (trans_logit && upper >= 1){
            upper = 1.0-xval_precision;
        }
        for (double x = lower; x <= upper; x += step){
            double x_t = x;
            if (this->trans_log){
                x_t = log(x);
            }
            else if (this->trans_logit){
                x_t = logit(x);
            }
            eval_funcs(x_t, true, has_dll_dx, has_d2ll_dx2);
            if (has_dll_dx && has_d2ll_dx2){
                fprintf(stdout, "%f\t%f\t%f\t%f\n", x, cur_ll_x, cur_dll_dx,
                    cur_d2ll_dx2);
            }
            else if (has_dll_dx){
                fprintf(stdout, "%f\t%f\t%f\n", x, cur_ll_x, cur_dll_dx);
            }
            else{
                fprintf(stdout, "%f\t%f\n", x, cur_ll_x);
            }
        }
    }

    double univar::solve(double lower, double upper){
        // To be implemented
        return 0.0;
    }
}

