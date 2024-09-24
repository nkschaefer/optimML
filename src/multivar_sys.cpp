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
#include "solver.h"
#include "multivar_ml.h"
#include "multivar_sys.h"

using namespace std;

namespace optimML{
    
    void multivar_sys_solver::add_one_param(double param){
        this->dyi_dx.push_back(0.0);
        this->results.push_back(0.0);
        this->trans_log.push_back(false);
        this->trans_logit.push_back(false);
        this->params.push_back(param);
    }

    // Set up solver with initial guesses for parameter values
    multivar_sys_solver::multivar_sys_solver(vector<double> params_init){
        
        this->n_param_grp = 0;

        n_equations = 0;
        
        // Make room for temporary storage of equation gradients
        for (int i = 0; i < params_init.size(); ++i){
            this->add_one_param(params_init[i]);
        }

        this->rss = -1;
        this->nthread = 0;
        this->nthread_bfgs = 0;
    }
    
    multivar_sys_solver::multivar_sys_solver(){
        this->n_param_grp = 0;
        n_equations = 0;
        this->rss = -1;
        this->nthread = 0;
        this->nthread_bfgs = 0;
    }

    void multivar_sys_solver::add_param_grp(vector<double>& p){
        int grp_idx = this->n_param_grp;
        for (int i = 0; i < p.size(); ++i){
            int param_idx = this->params.size();
            this->param2grp.insert(make_pair(param_idx, grp_idx));
            this->params.push_back(p[i]);
            this->dyi_dx.push_back(0.0);
            this->results.push_back(0.0);
            this->trans_log.push_back(false);
            this->trans_logit.push_back(false);
        }
        this->n_param_grp++;
    }
    
    void multivar_sys_solver::set_threads(int nt){
        if (nt <= 1){
            nt = 0;
        }
        this->nthread = nt;
    }
    
    void multivar_sys_solver::set_threads_bfgs(int nt){
        if (nt <= 1){
            nt = 1;
        }
        this->nthread_bfgs = nt;
    }

    // Add an equation
    bool multivar_sys_solver::add_equation(multivar_sys_func eq,
        multivar_sys_func_d d_eq,
        double rhs,
        double weight){
        
        this->rhs.push_back(rhs);
        
        int eq_idx = n_equations;

        equations1.insert(make_pair(eq_idx, eq));
        equations1_deriv.insert(make_pair(eq_idx, d_eq));
        this->y.push_back(0.0);

        for (map<string, vector<double> >::iterator d = data_d.begin(); d != data_d.end();
            ++d){
            d->second.push_back(0.0);
        }
        for (map<string, vector<int> >::iterator d = data_i.begin(); d != data_i.end();
            ++d){
            d->second.push_back(0);
        }
        this->weights.push_back(weight);
        n_equations++;
        return true;
    }
    
    bool multivar_sys_solver::add_equation(multivar_sys_func eq,
        multivar_sys_func_d d_eq,
        double rhs){
        
        return add_equation(eq, d_eq, rhs, 1.0);
    }

    bool multivar_sys_solver::add_equation(multivar_func eq,
        multivar_func_d d_eq,
        double rhs,
        double weight){
        
        this->rhs.push_back(rhs);

        int eq_idx = n_equations;

        equations2.insert(make_pair(eq_idx, eq));
        equations2_deriv.insert(make_pair(eq_idx, d_eq));
        this->y.push_back(0.0);

        for (map<string, vector<double> >::iterator d = data_d.begin(); d != data_d.end();
            ++d){
            d->second.push_back(0.0);
        }
        for (map<string, vector<int> >::iterator d = data_i.begin(); d != data_i.end();
            ++d){
            d->second.push_back(0);
        }
        this->weights.push_back(weight);

        n_equations++;
        return true;
    }
    
    bool multivar_sys_solver::add_equation(multivar_func eq,
        multivar_func_d d_eq,
        double rhs){
        return this->add_equation(eq, d_eq, rhs, 1.0);
    }
    bool multivar_sys_solver::add_data(std::string name, double dat){
        if (name == "rhs"){
            fprintf(stderr, "ERROR: key \"rhs\" is reserved\n");
            return false;
        }
        if (data_d.count(name) == 0){
            vector<double> v;
            data_d.insert(make_pair(name, v));
            // Catch up to current position.
            for (int i = 0; i < n_equations-1; ++i){
                data_d[name].push_back(0.0);
            }
            data_d[name].push_back(dat);
        }
        else{
            data_d[name][data_d[name].size()-1] = dat;
        }
        return true;
    }

    bool multivar_sys_solver::add_data(std::string name, int dat){
        if (name == "eq_idx"){
            fprintf(stderr, "ERROR: name \"eq_idx\" is reserved\n");
            return false;
        }
        if (data_i.count(name) == 0){
            vector<int> v;
            data_i.insert(make_pair(name, v));
            // Catch up to current position.
            for (int i = 0; i < n_equations-1; ++i){
                data_i[name].push_back(-1);
            }
            data_i[name].push_back(dat);
        }
        else{
            data_i[name][data_i[name].size()-1] = dat;
        }
        return true;
    }
    
    bool multivar_sys_solver::add_data_fixed(std::string name, double dat){
        if (name == "rhs"){
            fprintf(stderr, "ERROR: key \"rhs\" is reserved\n");
            return false;
        }
        this->data_d_fixed.insert(make_pair(name, dat));
        return true;
    }
    
    bool multivar_sys_solver::add_data_fixed(std::string name, int dat){
        if (name == "eq_idx"){
            fprintf(stderr, "ERROR: name \"eq_idx\" is reserved\n");
            return false;
        }
        this->data_i_fixed.insert(make_pair(name, dat));
        return true;
    }
    
    bool multivar_sys_solver::add_prior(int idx, std::string name,
        std::vector<double> params){
        if (name != "normal" && name != "beta" && name != "poisson"){
            fprintf(stderr, "ERROR: name %s not supported.\n", name.c_str());
            fprintf(stderr, "Allowed: normal, beta, poisson\n");
            return false;
        }
        if (name == "normal" && params.size() != 2 && params.size() != 4){
            fprintf(stderr, "ERROR: expect 2 or 4 params for normal prior\n");
            return false;
        }
        if (name == "beta" && params.size() != 2){
            fprintf(stderr, "ERROR: expect 2 params for beta prior\n");
            return false;
        }
        if (name == "poisson" && params.size() != 1){
            fprintf(stderr, "ERROR: expect 1 param for poisson prior\n");
            return false;
        }
        this->priors.insert(make_pair(idx, name));
        this->prior_params.insert(make_pair(idx, params));
        return true;
    }

    bool multivar_sys_solver::set_data(int idx, std::string name, double dat){
        if (idx < 0 || idx > n_equations-1){
            fprintf(stderr, "ERROR: invalid index %d\n", idx);
            return false;
        }
        if (name == "rhs"){
            fprintf(stderr, "ERROR: name \"rhs\" is reserved\n");
            return false;
        }
        if (data_d.count(name) == 0){
            vector<double> v;
            data_d.insert(make_pair(name, v));
            // Catch up to current position.
            for (int i = 0; i < n_equations; ++i){
                data_d[name].push_back(0.0);
            }
            data_d[name][idx] = dat;
        }
        else{
            data_d[name][idx] = dat;
        }
        return true;
    }
    
    bool multivar_sys_solver::set_data(int idx, std::string name, int dat){
        if (idx < 0 || idx > n_equations-1){
            fprintf(stderr, "ERROR: invalid index %d\n", idx);
            return false;
        }
        if (name == "eq_idx"){
            fprintf(stderr, "ERROR: name \"eq_idx\" is reserved\n");
            return false;
        }
        if (data_i.count(name) == 0){
            vector<int> v;
            data_i.insert(make_pair(name, v));
            // Catch up to current position.
            for (int i = 0; i < n_equations; ++i){
                data_i[name].push_back(-1);
            }
            data_i[name][idx] = dat;
        }
        else{
            data_i[name][idx] = dat;
        }
        return true;
    }

    // Here, what would normally be an index into the data vector -- corresponding
    // to a single observation -- is an index into the set of equations.

    // We only need to return the squared residual for a single equation.

    double multivar_sys_solver::eval_sum_sq(const vector<double>& params,
        const map<string, double>& data_d,
        const map<string, int>& data_i){
        
        int eq_idx = data_i.at("eq_idx");
        
        // Evaluate equation
        double result;
        if (this->equations1.count(eq_idx) > 0){
            result = this->equations1[eq_idx](params);
        }
        else{
            result = this->equations2[eq_idx](params, data_d, data_i);
        }
        
        // Store result to access when computing derivative
        this->y[eq_idx] = result;

        // Return sum of squared difference to data point
        double rhs = data_d.at("rhs");
        
        // multivar_ml_solver wants to maximize (log likelihood), but we
        // want to minimize (RSS) -- so flip sign.
        return -pow(result-rhs, 2);
    }

    void multivar_sys_solver::eval_dsum_sq(const vector<double>& params,
        const map<string, double>& data_d,
        const map<string, int>& data_i,
        vector<double>& results){
        
        int eq_idx = data_i.at("eq_idx");
        
        double y_j = this->y[eq_idx];

        double rhs = data_d.at("rhs");
        
        // Evaluate derivative of RSS wrt this equation
        // Again, make negative since we want to minimize this and
        // multivar_ml_solver is set up to maximize
        double drss_dy_j = -(2*y_j - 2*rhs);
        
        for (int i = 0; i < this->dyi_dx.size(); ++i){
            this->dyi_dx[i] = 0.0;
        }

        // Evaluate current function's derivative wrt all params 
        if (this->equations1_deriv.count(eq_idx) > 0){
            this->equations1_deriv[eq_idx](params, this->dyi_dx);
        }
        else{
            this->equations2_deriv[eq_idx](params, data_d, data_i, this->dyi_dx);
        }

        for (int i = 0; i < params.size(); ++i){
            results[i] = drss_dy_j * dyi_dx[i];    
        }    
    }

    bool multivar_sys_solver::solve(){
        
        std::function<double(const vector<double>&, const map<string, double>&, 
            const map<string, int>&)> f = 
            [=](const vector<double>& params, const map<string, double>& data_d,
                const map<string, int>& data_i){
           return this->eval_sum_sq(params, data_d, data_i);
        };

        std::function<void(const vector<double>&, const map<string, double>&, 
            const map<string, int>&, vector<double>&)> fprime = 
            [=](const vector<double>& params, const map<string, double>& data_d,
                const map<string, int>& data_i, vector<double>& results){
           this->eval_dsum_sq(params, data_d, data_i, results);
        };
        
        this->solver.init(std::vector<double>{ }, f, fprime);
        if (this->nthread > 1){
            this->solver.set_threads(this->nthread);
        }
        if (this->nthread_bfgs > 1){
            this->solver.set_bfgs_threads(this->nthread_bfgs);
        }
        vector<double> curgroup;
        bool in_group = false;
        for (int i = 0; i < params.size(); ++i){
            if (this->param2grp.count(i) > 0){
                in_group = true;
                curgroup.push_back(this->params[i]);
            }
            else{
                if (in_group){
                    this->solver.add_param_grp(curgroup);
                    in_group = false;
                    curgroup.clear();
                }
                this->solver.add_one_param(params[i]);
            }
        }
        if (in_group){
            this->solver.add_param_grp(curgroup);
            in_group = false;
            curgroup.clear();
        }

        for (int i = 0; i < params.size(); ++i){
            if (this->trans_log[i]){
                solver.constrain_pos(i);
            }
            else if (this->trans_logit[i]){
                solver.constrain_01(i);
            }
        } 
        
        for (map<int, string>::iterator p = priors.begin(); p != priors.end(); ++p){
            if (p->second == "normal"){
                if (prior_params[p->first].size() == 2){
                    solver.add_normal_prior(p->first, prior_params[p->first][0], prior_params[p->first][1]);
                }
                else{
                    solver.add_normal_prior(p->first, prior_params[p->first][0], prior_params[p->first][1],
                        prior_params[p->first][2], prior_params[p->first][3]);
                }
            }
            else if (p->second == "beta"){
                solver.add_beta_prior(p->first, prior_params[p->first][0], prior_params[p->first][1]);
            }
            else if (p->second == "poisson"){
                solver.add_poisson_prior(p->first, prior_params[p->first][0]);
            }
        }

        this->solver.set_delta(1e-6);
        
        // Add data
        this->solver.add_weights(this->weights);

        this->solver.add_data("rhs", this->rhs);
        vector<int> indices;
        for (int i = 0; i < n_equations; ++i){
            indices.push_back(i);
        }
        this->solver.add_data("eq_idx", indices);
        
        for (map<string, vector<double> >::iterator d = 
            this->data_d.begin(); d != this->data_d.end(); ++d){
            this->solver.add_data(d->first, d->second);
        }
        for (map<string, vector<int> >::iterator d = 
            this->data_i.begin(); d != this->data_i.end(); ++d){
            this->solver.add_data(d->first, d->second);
        }
        for (map<string, double>::iterator d = 
            this->data_d_fixed.begin(); d != this->data_d_fixed.end(); ++d){
            this->solver.add_data_fixed(d->first, d->second);
        }
        for (map<string, int>::iterator d = 
            this->data_i_fixed.begin(); d != this->data_i_fixed.end(); ++d){
            this->solver.add_data_fixed(d->first, d->second);
        }
        bool success = this->solver.solve();
        if (success){
            // Copy log likelihood and store it as RSS
            this->rss = -solver.log_likelihood;
            // Copy results
            for (int i = 0; i < solver.results.size(); ++i){
                this->results[i] = solver.results[i];
            }
        }
        return success;
    } 
    
    bool multivar_sys_solver::constrain_pos(int idx){
        if (idx < 0 || idx > params.size()-1){
            fprintf(stderr, "ERROR: invalid idx %d\n", idx);
            return false;
        }
        if (param2grp.count(idx) > 0){
            fprintf(stderr, "ERROR: parameter %d is part of param group %d\n", idx, param2grp[idx]);
            return false;
        }
        this->trans_log[idx] = true;
        this->trans_logit[idx] = false;
        return true;
    }

    bool multivar_sys_solver::constrain_01(int idx){
        if (idx < 0 || idx > params.size()-1){
            fprintf(stderr, "ERROR: invalid idx %d\n", idx);
            return false;
        }
        if (param2grp.count(idx) > 0){
            fprintf(stderr, "ERROR: parameter %d is part of param group %d\n", idx, param2grp[idx]);
            return false;
        }
        this->trans_log[idx] = false;
        this->trans_logit[idx] = true;
        return true;
        
    }
        

}

