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
#include "multivar.h"

using std::cout;
using std::endl;
using namespace std;

// ===== multivar =====
// A class representing multivariate solvers. Does not implement a useful
// solve() method -- users must use child classes instead of this one.

namespace optimML{
    
    void multivar::dummy_d2_func(const vector<double>& params,
        const map<string, double>& params_d, const map<string, int>& params_i, 
        vector<vector<double> >& results){
        // Do nothing
    }

    multivar::multivar(){
        srand(time(NULL));   
        
        n_data = 0;
        nmixcomp = 0;
        has_prior_mixcomp = false;
        n_param_grp = 0;
        xval_max = 1e15;
        xval_min = 1e-15;
        xval_log_min = log(xval_min);
        xval_log_max = log(xval_max);
        xval_logit_min = logit(xval_min);
        xval_logit_max = logit(1.0-xval_min);
        
        initialized = false;
    }
    
    void multivar::add_one_param(double param){
        x.push_back(param);
        x_t.push_back(param);
        x_t_extern.push_back(param);
        x_skip.push_back(false);
        dy_dt_extern.push_back(0.0);

        /*
        vector<double> d2y_dt2_row;
        for (int j = 0; j < params_init.size(); ++j){
            d2y_dt2_row.push_back(0.0);
        }
        d2y_dt2_extern.push_back(d2y_dt2_row);
        */

        results.push_back(0.0);
        se.push_back(0.0);

        has_prior.push_back(false);
        ll_x_prior.push_back(dummy_prior_func);
        dll_dx_prior.push_back(dummy_prior_func);
        d2ll_dx2_prior.push_back(dummy_prior_func);

        trans_log.push_back(false);
        trans_logit.push_back(false);  
        
        map<string, double> m1;
        params_prior_double.push_back(m1);
        map<string, int> m2;
        params_prior_int.push_back(m2);
        
        n_param++;
        n_param_extern++;
    }

    void multivar::init_params(vector<double> params_init){
        
        n_param = 0;
        n_param_extern = 0;

        for (int i = 0; i < params_init.size(); ++i){
            add_one_param(params_init[i]);     
        }   
    }
    
    bool multivar::replace_params(vector<double>& params_new){
        if (params_new.size() != this->x.size()){
            fprintf(stderr, "ERROR: different number of parameters given\n");
            return false;    
        }
        for (int i = 0; i < params_new.size(); ++i){
            this->x[i] = params_new[i];
            this->x_t[i] = params_new[i];
            this->x_t_extern[i] = params_new[i];
            this->x_skip[i] = false;
            this->results[i] = 0;
            if (this->trans_log[i]){
                this->trans_log[i] = false;
                this->trans_logit[i] = false;
                constrain_pos(i);    
            }
            else if (this->trans_logit[i]){
                this->trans_log[i] = false;
                this->trans_logit[i] = false;
                constrain_01(i);
            }
        }
        return true;
    }

    void multivar::init(vector<double> params_init, multivar_func ll,
        multivar_func_d dll, multivar_func_d2 dll2){
        if (!initialized){
            init_params(params_init);
        }
        else{
            replace_params(params_init);
        }
        ll_x = ll;
        dll_dx = dll;
        d2ll_dx2 = dll2;
        has_2d = true;
        initialized = true;
    }

    void multivar::init(vector<double> params_init, multivar_func ll,
        multivar_func_d dll){
        if (!initialized){
            init_params(params_init);
        }
        else{
            replace_params(params_init);
        }
        ll_x = ll;
        dll_dx = dll;
        d2ll_dx2 = dummy_d2_func;
        has_2d = false;
        initialized = true;
    }
    
    bool multivar::add_prior(int idx, prior_func ll,
        prior_func dll, prior_func dll2){
        
        if (!initialized){
            fprintf(stderr, "ERROR: not initialized\n");
            return false;
        }
        if (idx >= n_param-nmixcomp){
            fprintf(stderr, "ERROR: index %d out of bounds\n", idx);
            return false;
        }
        this->has_prior[idx] = true;
        this->ll_x_prior[idx] = ll;
        this->dll_dx_prior[idx] = dll;
        this->d2ll_dx2_prior[idx] = dll2;
        return true;
    }

    bool multivar::add_prior(int idx, prior_func ll,
        prior_func dll){
        
        return add_prior(idx, ll, dll, dummy_prior_func);
    }

    bool multivar::add_prior_param(int idx, string name, double data){
        if (!initialized){
            fprintf(stderr, "ERROR: not initialized\n");
            exit(1);
        }
        this->params_prior_double[idx].insert(make_pair(name, data));
        return true;
    }

    bool multivar::add_prior_param(int idx, string name, int data){
        if (!initialized){
            fprintf(stderr, "ERROR: not initialized\n");
            exit(1);
        }
        this->params_prior_int[idx].insert(make_pair(name, data));
        return true;
    }
    
    /**
     * Add a group of parameters where each must be between 0 and 1 (exclusive)
     * and all must sum to 1. This could, for example, be the parameter vector
     * for a multinomial distribution.
     *
     * NOTE: this is distinct from the idea of "mixture components" (see below).
     * You can have an arbitrary number of groups added by this function but only
     * one set of mixture components.
     */ 
    bool multivar::add_param_grp(vector<double>& pg){
        // Ensure they sum to 1
        double tot = 0.0;
        for (int i = 0; i < pg.size(); ++i){
            tot += pg[i];
            if (pg[i] <= 0){
                fprintf(stderr, "ERROR: parameters in param group cannot be <= 0\n");
                return false;
            }
        }
        int group_idx = this->n_param_grp;
        for (int i = 0; i < pg.size(); ++i){
            int param_idx = this->x.size();
            this->param2grp.insert(make_pair(param_idx, group_idx));
            this->add_one_param(logit(pg[i] / tot));
        }
        this->n_param_grp++;
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
    bool multivar::add_mixcomp(vector<vector<double> >& dat){
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
        this->dy_dt_extern.push_back(0.0);
        vector<double> lastrow;
        for (int i = 0; i < d2y_dt2_extern.size(); ++i){
            d2y_dt2_extern[i].push_back(0.0);    
            lastrow.push_back(0.0);
        }
        lastrow.push_back(0.0);
        d2y_dt2_extern.push_back(lastrow);

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

    /**
     * If we are modeling mixture components, allow the user to add a Dirichlet prior
     * over mixture components. The parameters of the distribution should be provided
     * here.
     */
    bool multivar::add_mixcomp_prior(std::vector<double>& alphas){
        if (!initialized){
            fprintf(stderr, "ERROR: not initialized\n");
            exit(1);
        }
        if (this->nmixcomp == 0){
            fprintf(stderr, "ERROR: dimension of prior parameters (%ld) does not equal the dimension of mixture components (%d)\n", alphas.size(), nmixcomp);
            return false;
        }
        for (int i = 0; i < alphas.size(); ++i){
            if (alphas[i] == 0.0){
                fprintf(stderr, "ERROR: illegal value for Dirichlet parameter %d\n", i);
                return false;
            }
        }
        this->dirichlet_prior_mixcomp.clear();
        this->dirichlet_prior_mixcomp = alphas;

        // Create space in arrays that will store partial derivatives from function evaluations
        this->dy_dt_mixcomp_prior.clear();
        this->d2y_dt2_mixcomp_prior.clear();
        for (int i = 0; i < alphas.size(); ++i){
            dy_dt_mixcomp_prior.push_back(0.0);
            d2y_dt2_mixcomp_prior.push_back(0.0);
        }
        has_prior_mixcomp = true;
        return true;
    }

    bool multivar::add_mixcomp_fracs(vector<double>& fracs){
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

    void multivar::randomize_mixcomps(){
        if (!initialized){
            fprintf(stderr, "ERROR: not initialized\n");
            exit(1);
        }
        if (has_prior_mixcomp){
            // Draw randomly from the prior Dirichlet distribution to obtain starting values.
            // Sample a Gamma variable for each alpha, then set proportion to each alpha / alpha_sum
            vector<double> rands;
            double randsum = 0.0;
            for (int i = 0; i < nmixcomp; ++i){
                
                default_random_engine generator(time(NULL));
                gamma_distribution<double> dist(dirichlet_prior_mixcomp[i], 1.0);
                double samp = dist(generator);
                randsum += samp;
                rands.push_back(samp);
            }
            for (int i = 0; i < nmixcomp; ++i){
                x[n_param-nmixcomp+i] = logit(rands[i]/randsum);
            }
        }
        else{
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
    }
    
    vector<double> multivar::get_cur_mixprops(){
        if (!initialized){
            fprintf(stderr, "ERROR: not initialized\n");
            vector<double> v;
            return v;
        } 
        else if (this->nmixcomp == 0){
            fprintf(stderr, "ERROR: no mixcomps added\n");
            vector<double> v;
            return v;
        }
        vector<double> curprops;
        double mcsum = 0.0;
        for (int j = 0; j < nmixcomp; ++j){
            double val = expit(x[n_param-nmixcomp+j]);
            mcsum += val;
            curprops.push_back(val);
        }
        for (int j = 0; j < nmixcomp; ++j){
            curprops[j] /= mcsum;
        }
        return curprops;
    }    

    /**
     * Evaluate optional Dirichlet prior on mixture components
     */
    double multivar::ll_mixcomp_prior(){
        if (!has_prior_mixcomp){
            return 0.0;
        }
        double ll = 0.0;
        double alphasum = 0.0;
        double alphasum_neg = 0.0;
        for (int i = 0; i < nmixcomp; ++i){
            double x_i = x_t[n_param-nmixcomp+i];
            ll += (dirichlet_prior_mixcomp[i] - 1.0)*log(x_i);
            alphasum += dirichlet_prior_mixcomp[i];
            alphasum_neg += lgamma(dirichlet_prior_mixcomp[i]);
        }
        ll += lgamma(alphasum);
        ll -= alphasum_neg;
        return ll;
    }

    /**
     * Evaluate 1st derivative of optional Dirichlet prior on mixture components
     */
    void multivar::dll_mixcomp_prior(){
        if (!has_prior_mixcomp){
            return;
        }
        // Fill dy_dt_mixcomp_prior
        for (int i = 0; i < nmixcomp; ++i){
            if (!x_skip[n_param-nmixcomp+i]){
                double x_i = x_t[n_param-nmixcomp+i];
                dy_dt_mixcomp_prior[i] = (dirichlet_prior_mixcomp[i] - 1.0)/x_i;
            }
        }
    }

    /**
     * Evaluate 2nd derivative of optional Dirichlet prior on mixture components
     */
    void multivar::d2ll_mixcomp_prior(){
        if (!has_prior_mixcomp){
            return;
        }
        // Fill d2y_dt2_mixcomp_prior
        for (int i = 0; i < nmixcomp; ++i){
            if (!x_skip[n_param-nmixcomp+i]){
                double x_i = x_t[n_param-nmixcomp+i];
                d2y_dt2_mixcomp_prior[i] = (1.0 - dirichlet_prior_mixcomp[i])/(x_i*x_i);
            }
        }
    }

    bool multivar::add_normal_prior(int idx, double mu, double sigma){
        if (sigma == 0){
            fprintf(stderr, "ERROR: sigma cannot equal 0\n");
            return false;
        }
        if (!add_prior(idx, ll_prior_normal, dll_prior_normal, d2ll_prior_normal)){
            return false;
        }
        if (!add_prior_param(idx, "mu", mu) || !add_prior_param(idx, "sigma", sigma)){
            return false;
        }
        return true;
    }

    bool multivar::add_normal_prior(int idx, double mu, double sigma, double a, double b){
        if (b <= a){
            fprintf(stderr, "ERROR: lower bound of truncated normal set above upper bound\n");
            return false;
        }
        else if (sigma == 0){
            fprintf(stderr, "ERROR: sigma cannot equal 0\n");
            return false;
        }
        if (!add_prior(idx, ll_prior_normal, dll_prior_normal, d2ll_prior_normal)){
            return false;
        }
        if (!add_prior_param(idx, "mu", mu) || !add_prior_param(idx, "sigma", sigma) || 
            !add_prior_param(idx, "a", a) || !add_prior_param(idx, "b", b)){
            return false;
        }
        return true;
    }

    bool multivar::add_beta_prior(int idx, double alpha, double beta){
        if (alpha == 0 || beta == 0){
            fprintf(stderr, "ERROR: alpha and beta cannot equal 0\n");
            return false;
        }
        if (!add_prior(idx, ll_prior_beta, dll_prior_beta, d2ll_prior_beta)){
            return false;
        }
        if (!add_prior_param(idx, "alpha", alpha) || !add_prior_param(idx, "beta", beta)){
            return false;
        }
        return true;
    }
    
    bool multivar::add_poisson_prior(int idx, double lambda){
        if (lambda <= 0){
            fprintf(stderr, "ERROR: lambda must be >= 0\n");
            return false;
        }
        if (!add_prior(idx, ll_prior_poisson, dll_prior_poisson, d2ll_prior_poisson)){
            return false;
        }
        if (!add_prior_param(idx, "lambda", lambda)){
            return false;
        }
        return true;
    }

    bool multivar::set_param(int idx, double val){
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
   
    void multivar::constrain_pos(int idx){
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
        else if (this->param2grp.count(idx) > 0){
            x[idx] = expit(x[idx]);
            param2grp.erase(idx);
        }
        this->trans_logit[idx] = false;
        this->trans_log[idx] = true;
        if (x[idx] <= 0){
            fprintf(stderr, "ERROR: initial value %d out of domain for log transformation\n", idx);
            fprintf(stderr, "value: %f\n", x[idx]);
            return;
        }
        x[idx] = log(x[idx]);
    }

    void multivar::constrain_01(int idx){
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
        else if (this->param2grp.count(idx) > 0){
            x[idx] = expit(x[idx]);
            param2grp.erase(idx);
        }
        this->trans_log[idx] = false;
        this->trans_logit[idx] = true;
        if (x[idx] <= 0 || x[idx] >= 1.0){
            fprintf(stderr, "ERROR: initial value %d out of domain for logit transformation\n", idx);
            fprintf(stderr, "value: %f\n", x[idx]);
            return;
        }
        x[idx] = logit(x[idx]);
    }

    /**
     * When something goes wrong in function evaluation, send a message to stderr.
     */
    void multivar::print_function_error(){
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

    /**
     * When something goes wrong in evaluating a prior function, send a message
     * to stderr.
     */
    void multivar::print_function_error_prior(int idx){
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
    
    /**
     * Evaluate log likelihood at current vector of values
     * index is either an index into the data (gives us a vector of data values
     * for a single observation)
     * or -1 (tells us we're done with the data and it's time to apply the prior)
     */
    double multivar::eval_ll_x(int i){
        double f_x = 0.0;
        if (i >= 0){
            // Get log likelihood of one (current) row of data
            double ll = ll_x(x_t_extern, this->param_double_cur, this->param_int_cur);
            if (isnan(ll) || isinf(ll)){
                fprintf(stderr, "ERROR: illegal value returned by log likelihood function\n");
                print_function_error();
                exit(1);
            }
            if (this->weights.size() > 0){
                ll *= this->weights[i];
            }
            f_x += ll;
        }
        else{
            for (int j = 0; j < n_param-nmixcomp; ++j){
                if (this->has_prior[j]){
                    double ll = ll_x_prior[j](x_t_extern[j], 
                        this->params_prior_double[j], this->params_prior_int[j]);
                    if (isnan(ll) || isinf(ll)){
                        fprintf(stderr, "ERROR: illegal value returned by prior log likelihood function on \
    parameter %d\n", j);
                        print_function_error_prior(j);
                        exit(1);
                    }
                    f_x += ll;
                }
            }
            if (nmixcomp > 0 && has_prior_mixcomp){
                // Evaluate Dirichlet prior on mixture components
                f_x += ll_mixcomp_prior();
            }
        }
        return f_x;
    }

    /**
     * Evaluate the log likelihood function across every current data point.
     */
    double multivar::eval_ll_all(){

        double loglik = 0.0;
        
        map<int, double> grpsums;
        for (int i = 0; i < n_param_grp; ++i){
            grpsums.insert(make_pair(i, 0.0));
        }

        // Transform all non-mixture component variables
        for (int i = 0; i < n_param-nmixcomp; ++i){
            if (this->trans_log[i]){
                x_t[i] = exp(x[i]);
            }
            else if (this->trans_logit[i]){
                x_t[i] = expit(x[i]);
            }
            else if (param2grp.count(i) > 0){
                x_t[i] = expit(x[i]);
                grpsums[param2grp[i]] += x_t[i];
            }
            else{
                x_t[i] = x[i];
            }
            x_t_extern[i] = x_t[i];
        }

        for (map<int, int>::iterator pg = param2grp.begin(); pg != param2grp.end(); ++pg){
            x_t[pg->first] /= grpsums[pg->second];
            x_t_extern[pg->first] = x_t[pg->first];
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
            prepare_data(i);
            
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

    /**
     * Evaluate derivative of log likelihood at current vector; store in 
     * gradient vector.
     */
    void multivar::eval_dll_dx(int i){
        if (i >= 0){
            for (int z = 0; z < n_param_extern; ++z){
                this->dy_dt_extern[z] = 0.0;
            }
            dll_dx(x_t_extern, this->param_double_cur, this->param_int_cur, dy_dt_extern);
            for (int j = 0; j < n_param_extern; ++j){
                if (isnan(dy_dt_extern[j]) || isinf(dy_dt_extern[j])){
                    fprintf(stderr, "ERROR: invalid value returned by gradient function: parameter %d\n", j);
                    print_function_error();
                    exit(1);
                }
                double w = 1.0;
                if (this->weights.size() > 0){
                    w = this->weights[i];
                }
                if (nmixcomp > 0 && j == n_param_extern-1){
                    dy_dp = dy_dt_extern[j] * w;
                }
                else{
                    dy_dt[j] = dy_dt_extern[j];
                    if (x_skip[j]){
                        G[j] = 0.0;
                        dt_dx[j] = 0.0;
                    }
                    else{
                        G[j] += dy_dt[j] * w * dt_dx[j];
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
                        double dllprior = dll_dx_prior[j](x_t[j], this->params_prior_double[j], 
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
            if (nmixcomp > 0 && has_prior_mixcomp){
                // Evaluate Dirichlet prior on mixture components
                // This fills dy_dt_dirichlet_prior
                dll_mixcomp_prior();
                for (int j = 0; j < nmixcomp; ++j){
                    int j2 = n_param-nmixcomp + j;
                    if (!x_skip[j2]){
                        G[j2] += dy_dt_mixcomp_prior[j] * dt_dx[j2];
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

    /**
     * Evaluate second derivative at current parameter values; store results in
     * Hessian matrix. This function might be unnecessary but can be used by
     * future child classes.
     */
    void multivar::eval_d2ll_dx2(int i){
        if (!has_2d){
            return;
        }
        if (i >= 0){
            for (int z = 0; z < n_param_extern; ++z){
                for (int zz = 0; zz < n_param_extern; ++zz){
                    this->d2y_dt2_extern[z][zz] = 0.0;
                }
            }
            d2ll_dx2(x_t_extern, this->param_double_cur, this->param_int_cur, this->d2y_dt2_extern);
            for (int j = 0; j < n_param_extern; ++j){
                for (int k = 0; k < n_param_extern; ++k){
                    if (isnan(d2y_dt2_extern[j][k]) || isinf(d2y_dt2_extern[j][k])){
                        fprintf(stderr, "ERROR: illegal value returned by 2nd derivative function\n");
                        fprintf(stderr, "parameters: %d %d\n", j, k);
                        print_function_error();
                        exit(1);
                    }
                    
                    bool j_is_p = nmixcomp > 0 && j == n_param_extern-1;
                    bool k_is_p = nmixcomp > 0 && k == n_param_extern-1;
                    
                    double w = 1.0;
                    if (this->weights.size() > 0){
                        w = this->weights[i];
                    }

                    if (j_is_p && k_is_p){
                        d2y_dp2 = d2y_dt2_extern[j][k] * w;
                    }    
                    else if (j_is_p && !x_skip[k]){
                        d2y_dpdt[k] = d2y_dt2_extern[j][k] * w;
                    }
                    else if (k_is_p && !x_skip[j]){
                        d2y_dtdp[k] = d2y_dt2_extern[j][k] * w;
                    }
                    else if (!x_skip[j] && !x_skip[k]){
                        H[j][k] += d2y_dt2_extern[j][k] * w * d2t_dx2[j][k];
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
                            
                            H[j_i][k_i] += d2y_dp2 * dt_dx[j_i] * dt_dx[k_i] + 
                                dy_dp * d2t_dx2[j_i][k_i];
                            
                        }
                    }
                }
            }
        }
        else{
            for (int j = 0; j < n_param-nmixcomp; ++j){
                if (this->has_prior[j] && !x_skip[j]){
                    double d2llprior = d2ll_dx2_prior[j](x_t[j], this->params_prior_double[j], 
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
            if (nmixcomp > 0 && has_prior_mixcomp){
                // Evaluate Dirichlet prior on mixture components
                // This fills d2y_dt2_mixcomp_prior (represents diagonal elements only)
                d2ll_mixcomp_prior();
                for (int j = 0; j < nmixcomp; ++j){
                    int j2 = n_param-nmixcomp + j;
                    if (!x_skip[j2]){
                        for (int k = 0; k < nmixcomp; ++k){
                            int k2 = n_param-nmixcomp + k;
                            if (!x_skip[k2]){
                                if (j == k){
                                    H[j2][j2] += dt_dx[j2]*dt_dx[j2] * d2y_dt2_mixcomp_prior[j] + 
                                        dy_dt_mixcomp_prior[j2] * d2t_dx2[j2][j2]; 
                                }
                                else{
                                    // Off diagonal second derivatives of prior are zero
                                    H[j2][k2] += dy_dt_mixcomp_prior[j2] * d2t_dx2[j2][k2];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * After a solution is found, fill external data structures that user 
     * can see
     */
    void multivar::fill_results(double llprev){
        map<int, double> grpsums;
        for (int i = 0; i < n_param_grp; ++i){
            grpsums.insert(make_pair(i, 0.0));
        }

        // Store un-transformed results
        for (int j = 0; j < n_param-nmixcomp; ++j){
            if (trans_log[j]){
                results[j] = exp(x[j]);
            }
            else if (trans_logit[j]){
                results[j] = expit(x[j]);
            }
            else if (param2grp.count(j) > 0){
                results[j] = expit(x[j]);
                grpsums[param2grp[j]] += results[j];
            }
            else{
                results[j] = x[j];
            } 
        }

        for (map<int, int>::iterator pg = param2grp.begin(); pg != param2grp.end(); ++pg){
            results[pg->first] /= grpsums[pg->second];
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
     * To be implemented by child classes
     */
    bool multivar::solve(){
        return false;
    }

}

