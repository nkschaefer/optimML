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
#include <mutex>
#include <condition_variable>
#include "functions.h"
#include "solver.h"
using std::cout;
using std::endl;
using namespace std;

// ===== base class for all solvers/optimizers =====
// Implements some base methods and is extended by multivar_ml_solver and univar_solver

namespace optimML{
    
    /**
     * Constructor
     */
    solver::solver(){
        // Set default values.
        n_data = 0;
        maxiter = 1000;
        delta_thresh = 0.01;
        log_likelihood = 0.0;
        initialized = false;
        fixed_data_dumped = false;
        
        nthread = 0;
        terminate_threads = false;
        pool_open = false;
        threads_init = false;
    }
    
    /**
     * Destructor
     */
    solver::~solver(){
        param_double_cur.clear();
        param_int_cur.clear();
        params_double_names.clear();
        params_int_names.clear();
        param_double_ptr.clear();
        param_int_ptr.clear();
        params_double_vals.clear();
        params_int_vals.clear();
        
        if (threads_init){
            delete queue_mutex;
            delete has_jobs;
        }
    }
    

    /**
     * Helper for truncated normal LL calculations
     */
    double solver::phi(double x){
        return  0.5*(1 + erf(x / sqrt(2)));
    }

    /**
     * Helper for normal LL calculations
     */
    double solver::dnorm(double x, double mu, double sigma){
        static double sq2pi = sqrt(2.0 * M_PI);
        return -0.5 * pow((x-mu)/sigma, 2) - log(sigma * sq2pi);
    }

    /**
     * Built-in prior function for input variables: (truncated) normal
     */
    double solver::ll_prior_normal(double x, const map<string, double>& params_d,
        const map<string, int>& params_i){
        bool trunc = false;
        double trunc_low;
        double trunc_high;
        if (params_d.count("a") > 0 && params_d.count("b") > 0){
            trunc_low = params_d.at("a");
            trunc_high = params_d.at("b");
            trunc = true;
        }
        if (trunc){
            if (x < trunc_low || x > trunc_high){
                return log(0);
            }
        }
        double mu = params_d.at("mu");
        double sigma = params_d.at("sigma");
        return dnorm(x, mu, sigma) - log(phi((1.0-mu)/sigma) - phi((0.0-mu)/sigma));
            
    }

    double solver::dll_prior_normal(double x, const map<string, double>& params_d,
        const map<string, int>& params_i){
        // Truncation no longer matters for derivatives wrt independent variable
        double mu = params_d.at("mu");
        double sigma = params_d.at("sigma");
        return -((x-mu)/(sigma*sigma));
    }

    double solver::d2ll_prior_normal(double x, const map<string, double>& params_d,
        const map<string, int>& params_i){
        double mu = params_d.at("mu");
        double sigma = params_d.at("sigma");
        return -1.0/(sigma*sigma);
    }

    double solver::ll_prior_beta(double x, const map<string, double>& params_d,
        const map<string, int>& params_i){
        double a = params_d.at("alpha");
        double b = params_d.at("beta");
        return (a-1.0)*log(x) + (b-1)*log(1.0-x) - (lgamma(a) + lgamma(b) - lgamma(a+b));
    }

    double solver::dll_prior_beta(double x, const map<string, double>& params_d,
        const map<string, int>& params_i){
        double a = params_d.at("alpha");
        double b = params_d.at("beta");
        return (a-1.0)/x - (b-1.0)/(1.0-x);
    }

    double solver::d2ll_prior_beta(double x, const map<string, double>& params_d,
        const map<string, int>& params_i){
        double a = params_d.at("alpha");
        double b = params_d.at("beta");
        return (1.0-a)/(x*x) + (1.0-b)/((1.0-x)*(1.0-x));
    }
    
    double solver::ll_prior_poisson(double x, const map<string, double>& params_d,
        const map<string, int>& params_i){
        double l = params_d.at("lambda");
        
        // Compute factorial using Stirling's approximation
        double lxfac = 0.5*(log(2) + log(M_PI) + log(x)) + 
            x*(log(x) - log(exp(1)));

        return x*log(l) + -l - lxfac; 
    } 

    double solver::dll_prior_poisson(double x, const map<string, double>& params_d,
        const map<string, int>& params_i){
        double l = params_d.at("lambda");
        return x/l - 1.0;
    }

    double solver::d2ll_prior_poisson(double x, const map<string, double>& params_d,
        const map<string, int>& params_i){
        double l = params_d.at("lambda");
        return -x/(l*l);
    }

    /**
     * Placeholder to fill in prior function arrays until replaced with 
     * a real function
     */
    double solver::dummy_prior_func(double x, const map<string, double>& params_d,
        const map<string, int>& params_i){
        return 0.0;
    }
    
    /**
     *  WARNING: this does not copy the data into the solver object. In other words,
     *  you must keep a valid copy of the data outside of this class -- you cannot
     *  declare vectors in a loop and add them using this function, as they will be
     *  garbage collected before the solver can run.
     */    
    bool solver::add_data(string name, std::vector<double>& dat){
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
        
        if (threads_init){
            for (int i = 0; i < nthread; ++i){
                params_double_cur_thread[i].insert(make_pair(name, 0.0));
                double* ptr = &params_double_cur_thread[i].at(name);
                param_double_ptr_thread[i].push_back(ptr);   
            }
        }
        return true;
    }

    bool solver::add_data(string name, std::vector<int>& dat){
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
        if (threads_init){
            for (int i = 0; i < nthread; ++i){
                params_int_cur_thread[i].insert(make_pair(name, 0.0));
                int* ptr = &params_int_cur_thread[i].at(name);
                param_int_ptr_thread[i].push_back(ptr);
            }
        }
        return true;
    }
    
    bool solver::add_data_fixed(string name, double dat){
        if (!initialized){
            fprintf(stderr, "ERROR: not initialized\n");
            exit(1);
        }
        if (param_double_cur.count(name) > 0){
            fprintf(stderr, "ERROR: already has data keyed to %s\n", name.c_str());
            return false;
        }
        param_double_cur.insert(make_pair(name, dat));
        if (threads_init){
            for (int i = 0; i < nthread; ++i){
                params_double_cur_thread[i].insert(make_pair(name, dat));
            }
        }
        return true;
    }

    bool solver::add_data_fixed(string name, int dat){
        if (!initialized){
            fprintf(stderr, "ERROR: not initialized\n");
            exit(1);
        }
        if (param_int_cur.count(name) > 0){
            fprintf(stderr, "ERROR: already has data keyeed to %s\n", name.c_str());
            return false;
        }
        param_int_cur.insert(make_pair(name, dat));
        if (threads_init){
            for (int i = 0; i < nthread; ++i){
                params_int_cur_thread[i].insert(make_pair(name, dat));
            }
        }
        return true;
    }
    
    /**
     * If the user has only provided fixed data (and no regular type data), 
     * treat the fixed data like regular data.
     */
    bool solver::fixed_data_to_data(){
        if (this->n_data == 0){

            vector<string> dat_d_names;
            
            // Everything in param_double_cur and param_int_cur must have come from fixed data.
            for (map<string, double>::iterator c = param_double_cur.begin(); c !=
                param_double_cur.end(); ){
                if (threads_init){
                    for (int i = 0; i < nthread; ++i){
                        params_double_cur_thread[i].erase(c->first);
                    }
                }
                dat_d_names.push_back(c->first);
                data_d_tmp.push_back(vector<double>{ c->second });
                param_double_cur.erase(c++);
            }

            vector<string> dat_i_names;

            for (map<string, int>::iterator c = param_int_cur.begin(); c != 
                param_int_cur.end(); ){
                if (threads_init){
                    for (int i = 0; i < nthread; ++i){
                        params_int_cur_thread[i].erase(c->first);
                    }
                }
                dat_i_names.push_back(c->first);
                data_i_tmp.push_back(vector<int>{ c->second });
                param_int_cur.erase(c++);
            }

            for (int i = 0; i < data_d_tmp.size(); ++i){
                this->add_data(dat_d_names[i], data_d_tmp[i]);
            }
            for (int i = 0; i < data_i_tmp.size(); ++i){
                this->add_data(dat_i_names[i], data_i_tmp[i]);
            }
            fixed_data_dumped = true;
        }

        return this->n_data > 0;
    }
    
    bool solver::create_threads(){
        if (threads_init){
            fprintf(stderr, "ERROR: threads already initialized.\n");
            return false;
        }
        
        // Create a distinct data set per thread
        set<string> name_d_copied;
        set<string> name_i_copied;

        for (int i = 0; i < nthread; ++i){
            map<string, double> m;
            map<string, int> m2;
            params_double_cur_thread.push_back(m);
            params_int_cur_thread.push_back(m2);
            vector<double*> v;
            param_double_ptr_thread.push_back(v);
            vector<int*> v2;
            param_int_ptr_thread.push_back(v2);
            
            for (int j = 0; j < params_double_names.size(); ++j){
                params_double_cur_thread[i].insert(make_pair(params_double_names[j], 0.0));
                double* ptr = &params_double_cur_thread[i].at(params_double_names[j]);
                param_double_ptr_thread[i].push_back(ptr);
                if (i == 0){
                    name_d_copied.insert(params_double_names[j]);
                }
            }

            for (map<string, double>::iterator c = param_double_cur.begin(); c != 
                param_double_cur.end(); ++c){
                if (name_d_copied.find(c->first) == name_d_copied.end()){
                    // Fixed data
                    params_double_cur_thread[i].insert(make_pair(c->first, c->second));
                }
            }
            
            for (int j = 0; j < params_int_names.size(); ++j){
                params_int_cur_thread[i].insert(make_pair(params_int_names[j], 0));
                int* ptr = &params_int_cur_thread[i].at(params_int_names[j]);
                param_int_ptr_thread[i].push_back(ptr);
                if (i == 0){
                    name_i_copied.insert(params_int_names[j]);
                }
            }

            for (map<string, int>::iterator c = param_int_cur.begin(); c != 
                param_int_cur.end(); ++c){
                if (name_i_copied.find(c->first) == name_i_copied.end()){
                    // Fixed data
                    params_int_cur_thread[i].insert(make_pair(c->first, c->second));
                }
            }
        } 
         
        queue_mutex = new mutex;
        has_jobs = new condition_variable;
    
        threads_init = true;
        return true;
    }
    
    void solver::launch_threads(){
        if (!initialized || !threads_init){
            fprintf(stderr, "ERROR: not initialized\n");
            exit(1);
        }
        terminate_threads = false;
        pool_open = true;
        for (int i = 0; i < nthread; ++i){
            thread* t = new thread(&solver::worker, this, i);
            this->threads.push_back(t);
        }
    }

    void solver::add_job(int i){
        if (!pool_open){
            fprintf(stderr, "ERROR: thread pool not active\n");
            exit(1);
        }
        unique_lock<mutex> lock(*this->queue_mutex);
        this->job_inds.push_back(i);
        this->has_jobs->notify_one();
    }
    
    int solver::get_next_job(){
        if (!pool_open){
            fprintf(stderr, "ERROR: thread pool not active\n");
            exit(1);
        }
        unique_lock<mutex> lock(*this->queue_mutex);
        this->has_jobs->wait(lock, [this]{ return job_inds.size() > 0 ||
            terminate_threads;});
        if (this->job_inds.size() == 0 && this->terminate_threads){
            return -1;
        }
        int jid = this->job_inds[0];
        this->job_inds.pop_front();
        return jid;
    }

    void solver::worker(int thread_idx){
        // To be implemented by child classes - get a job ID and 
        // process the data
    }
    
    void solver::close_pool(){
        {
            unique_lock<mutex> lock(*queue_mutex);
            terminate_threads = true;
        }
        has_jobs->notify_all();
        for (int i = 0; i < nthread; ++i){
            threads[i]->join();
            delete threads[i];
        }
        threads.clear();
        pool_open = false;
        
        // Delete thread data?
    }

    bool solver::add_weights(std::vector<double>& weights){
        if (!initialized){
            fprintf(stderr, "ERROR: not initialized\n");
            exit(1);
        }
        if (n_data > 0 && weights.size() != this->n_data){
            fprintf(stderr, "ERROR: weight vector not same length as data\n");
            return false;
        } 
        this->weights = weights;
        return true;
    }

    void solver::set_delta(double d){
        if (!initialized){
            fprintf(stderr, "ERROR: not initialized\n");
            exit(1);
        }
        this->delta_thresh = d;
    }

    void solver::set_maxiter(int m){
        if (!initialized){
            fprintf(stderr, "ERROR: not initialized\n");
            exit(1);
        }
        this->maxiter = m;
    }
    
    void solver::set_threads(int nt){
        if (threads_init){
            fprintf(stderr, "ERROR: cannot change thread number after threads initialized\n");
            exit(1);
        }
        // 1 thread might as well be 0 - don't launch thread stuff for that
        if (nt <= 1){
            nt = 0;
        }
        nthread = nt;
    }
    
    void solver::prepare_data(int i, int thread_idx){
        // Update parameter maps that will be sent to functions
        if (threads_init && thread_idx >= 0){
            int jid = i;
            for (int j = 0; j < this->params_double_names.size(); ++j){
                *(this->param_double_ptr_thread[thread_idx][j]) = 
                    (this->params_double_vals[j][jid]);
            }
            for (int j = 0; j < this->params_int_names.size(); ++j){
                *(this->param_int_ptr_thread[thread_idx][j]) = 
                    (this->params_int_vals[j][jid]);
            }
        }
        else{
            for (int j = 0; j < this->params_double_names.size(); ++j){
                *(this->param_double_ptr[j]) = (this->params_double_vals[j][i]);
            }
            for (int j = 0; j < this->params_int_names.size(); ++j){
                *(this->param_int_ptr[j]) = (this->params_int_vals[j][i]);
            }
        }
    }
    
    void solver::fill_results(double ll){
        log_likelihood = ll;
    }
}

