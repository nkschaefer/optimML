#ifndef _OPTIMML_SOLVER_H
#define _OPTIMML_SOLVER_H
#include <string>
#include <algorithm>
#include <vector>
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
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>

// ===== Base class for solvers =====

namespace optimML{
    
    // Exception code for math issues
    const int OPTIMML_MATH_ERR = 123;

    // Function for prior distributions over individual x variables
    typedef std::function< double( double,
        const std::map<std::string, double>&,
        const std::map<std::string, int>& ) > prior_func;

    class solver{
        protected:
            
            bool initialized;
            
            // Pre-set prior functions        
            static double phi(double x);
            static double dnorm(double x, double mu, double sigma);
            static double ll_prior_normal(double x, const std::map<std::string, double>& params_d,
                const std::map<std::string, int>& params_i);
            static double dll_prior_normal(double x, const std::map<std::string, double>& params_d,
                const std::map<std::string, int>& params_i);
            static double d2ll_prior_normal(double x, const std::map<std::string, double>& params_d,
                const std::map<std::string, int>& params_i);
            static double ll_prior_beta(double x, const std::map<std::string, double>& params_d,
                const std::map<std::string, int>& params_i);
            static double dll_prior_beta(double x, const std::map<std::string, double>& params_d,
                const std::map<std::string, int>& params_i);
            static double d2ll_prior_beta(double x, const std::map<std::string, double>& params_d,
                const std::map<std::string, int>& params_i);
            static double ll_prior_poisson(double x, const std::map<std::string, double>& params_d,
                const std::map<std::string, int>& params_i);
            static double dll_prior_poisson(double x, const std::map<std::string, double>& params_d,
                const std::map<std::string, int>& params_i);
            static double d2ll_prior_poisson(double x, const std::map<std::string, double>& params_d,
                const std::map<std::string, int>& params_i); 

            static double dummy_prior_func(double x, const std::map<std::string, double>& params_d,
                const std::map<std::string, int>& params_i);   

            // Optional: observation weights for weighted ML calculation
            std::vector<double> weights;    
            
            // How many data points are there?
            int n_data;
            
            std::vector<std::string> params_double_names;
            std::vector<double*> params_double_vals;
            
            std::vector<std::string> params_int_names;
            std::vector<int*> params_int_vals;
            
            std::map<std::string, double> param_double_cur;
            std::map<std::string, int> param_int_cur;
           
            // Pointer to location in above maps
            // Position is same as in params_double_names and params_double_vals
            // For the jth variable, ith data point:
            // name = params_double_names[j]
            // val = *params_double_vals[j][i]
            // pointer to update in map: param_double_ptr[j]

            std::vector<double*> param_double_ptr;
            std::vector<int*> param_int_ptr;

            // Maximum iterations
            int maxiter;

            // Delta threshold
            double delta_thresh;   
            
            // To be called before function evaluations: fill data structures with current
            // data points, given index            
            void prepare_data(int i, int thread_idx = -1);
            
            // Make things user-friendly after finding a solution and before exiting
            void fill_results(double ll);
            
            // If the user has added no normal/row based data but only fixed data,
            // dump all the fixed stuff into the data structures for row-based data.
            // Return true if we end up with row-based data, false otherwise.        
            bool fixed_data_to_data();
            
            // For use with above method: if user provided only fixed data, we need to store
            // it somewhere.
            std::vector<std::vector<double> > data_d_tmp;
            std::vector<std::vector<int> > data_i_tmp;
            
            bool fixed_data_dumped;
            
            // ----- Multithreading-related

            int nthread;
            std::deque<std::map<std::string, double> > params_double_cur_thread;
            std::deque<std::map<std::string, int> > params_int_cur_thread;
            std::deque<std::vector<double*> > param_double_ptr_thread;
            std::deque<std::vector<int*> > param_int_ptr_thread;
            
            std::mutex* queue_mutex;
            std::condition_variable* has_jobs;
            bool terminate_threads;
            std::vector<std::thread*> threads;
            bool pool_open;
            std::deque<int> job_inds;
            bool threads_init;
            
            bool create_threads();
            void launch_threads();
            void add_job(int idx);
            int get_next_job();
            virtual void worker(int thread_idx);
            void close_pool();

        public:
            
            solver();
            ~solver();

            bool add_data(std::string name, std::vector<double>& data);
            bool add_data(std::string name, std::vector<int>& data);
            //bool add_data(std::string name, std::map<int, double>& data, int ndata);
            //bool add_data(std::string name, std::map<int, int>& data, int ndata);
            bool add_data_fixed(std::string name, double data);
            bool add_data_fixed(std::string name, int data);
            
            bool add_weights(std::vector<double>& weights);
            
            void set_delta(double delt);
            void set_maxiter(int i);   
            void set_threads(int nt);
            
            double log_likelihood;
    };
}

#endif
