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
#include "em.h"

using namespace std;

namespace optimML{
    
    void em_solver::add_one_param(double param){
        if (!is_fit){
            this->params.push_back(param);
        }
        this->params_orig.push_back(param);
        n_params++;
    }
    
    void em_solver::set_param(int param_idx, double param){
        this->params[param_idx] = param;
        if (!is_fit){
            this->params_orig[param_idx] = param;
        }
        if (initialized){
            solver_global->set_param(param_idx, param);
            for (int i = 0; i < n_components; ++i){
                solvers[i]->set_param(param_idx, param);
            }
        }
    }

    // Set up solver with initial guesses for parameter values
    em_solver::em_solver(vector<double> params_init){
        n_components = 0;
        n_params = 0;

        // Make room for temporary storage of equation gradients
        for (int i = 0; i < params_init.size(); ++i){
            this->add_one_param(params_init[i]);
        }

        this->loglik = 0;
        this->responsibility_matrix = NULL;
        this->n_obs = -1;
        this->initialized = false;
        this->delta_thresh = 0.1;
        this->maxiter = 1000;
        this->loglik = 0.0;
        this->no_data_yet = true;
        this->solver_global = NULL;
        this->weightsum = 0.0;
        this->is_fit = false;
    }
    
    em_solver::em_solver(){
        n_components = 0;
        n_params = 0;
        this->loglik = 0;
        this->responsibility_matrix = NULL;
        this->n_obs = -1;
        this->initialized = false;
        this->delta_thresh = 0.1;
        this->maxiter = 1000;
        this->loglik = 0.0;
        this->no_data_yet = true;
        this->solver_global = NULL;
        this->weightsum = 0.0;
        this->is_fit = false;
    }
    
    em_solver::~em_solver(){
        if (this->initialized){
            for (int i = 0; i < solvers.size(); ++i){
                if (this->solvers[i] != NULL){
                    delete this->solvers[i];
                }
                this->solvers[i] = NULL;
            }
            this->solvers.clear();
            this->free_responsibility_matrix();
            if (this->solver_global != NULL){
                delete this->solver_global;
                this->solver_global = NULL;
            }
            lls_tmp.clear();
            lls_tmp_rowsum.clear();
        }
    }

    void em_solver::set_delta(double d){
        delta_thresh = d;
        if (initialized){
            for (int j = 0; j < n_components; ++j){
                solvers[j]->set_delta(d);
            }
            solver_global->set_delta(d);
        }
    }
    
    void em_solver::set_maxiter(int m){
        maxiter = m;
        if (initialized){
            for (int j = 0; j < n_components; ++j){
                solvers[j]->set_maxiter(m);
            }
            solver_global->set_maxiter(m);
        }
    }

    void em_solver::init_responsibility_matrix(int n_obs){
        if (this->n_obs != -1){
            // Responsibility matrix was formerly initialized for a different
            // data set
            this->free_responsibility_matrix();
        }
        if (this->n_components == 0){
            fprintf(stderr, "ERROR: cannot init responsibility matrix with 0 components.\n");
            exit(1);
        }
        this->n_obs = n_obs;

        this->responsibility_matrix = new double*[this->n_obs];
        for (int i = 0; i < this->n_obs; ++i){
            this->responsibility_matrix[i] = new double[this->n_components];
        }
    }

    void em_solver::free_responsibility_matrix(){
        if (this->n_obs != -1 && this->responsibility_matrix != NULL){
            for (int i = 0; i < this->n_obs; ++i){
                delete[] this->responsibility_matrix[i];
                this->responsibility_matrix[i] = NULL;
            }
            delete[] this->responsibility_matrix;
            this->responsibility_matrix = NULL;
        }
        this->n_obs = -1;
    }
    
    // Add a component
    bool em_solver::add_component(multivar_func func, multivar_func_d func_d){
        add_component(func, func_d, "");
        return true;
    }
    
    bool em_solver::add_component(multivar_func func, multivar_func_d func_d, std::string n){
        dist_funcs.push_back(func);
        dist_funcs_deriv.push_back(func_d);
        component_names.push_back(n);
        n_components++; 
        return true;
    }
    
    bool em_solver::rm_component(int idx){
        if (idx < 0 || idx > n_components-1){
            fprintf(stderr, "Invalid component idx %d\n", idx);
            return false;
        }
        int i = 0;
        vector<multivar_func>::iterator f = dist_funcs.begin();
        vector<multivar_func_d>::iterator df = dist_funcs_deriv.begin();
        vector<string>::iterator n = component_names.begin();
        vector<double>::iterator w = component_weights.begin(); 
        deque<multivar_ml_solver* >::iterator s = solvers.begin();
        while (i <= idx){
            if (i == idx){
                dist_funcs.erase(f);
                dist_funcs_deriv.erase(df);
                component_names.erase(n);
                if (initialized){
                    component_weights.erase(w);
                    delete *s;
                    s = solvers.erase(s);
                }
                break;
            }
            else{
                ++f;
                ++df;
                ++n;
                if (initialized){
                    ++w;
                    ++s;
                }
            }
            ++i;
        }
        
        n_components--;

        // Fix responsibility matrix
        if (n_obs > 0 && responsibility_matrix != NULL){
            for (int i = 0; i < n_obs; ++i){
                delete[] this->responsibility_matrix[i];
                this->responsibility_matrix[i] = NULL;
                this->responsibility_matrix[i] = new double[this->n_components]; 
            }
        }

        return true;
    }

    void em_solver::init(){
        if (initialized){
            fprintf(stderr, "ERROR: cannot re-initialize em_solver\n");
            return;
        }
        if (n_components < 1){
            fprintf(stderr, "ERROR: not enough components: %d\n", n_components);
            exit(1);
        }
        
        for (int i = 0; i < n_components; ++i){
            vector<double> paramscpy = params;
            multivar_ml_solver* solver = new multivar_ml_solver(paramscpy, dist_funcs[i], dist_funcs_deriv[i]);
            solvers.emplace_back(solver);
            solver->set_delta(delta_thresh);
            solver->set_maxiter(maxiter);
            solver->store_ll_data_points();
            component_weights.push_back(1.0/(double)n_components);
            compsums.push_back(0.0);
            params_extern.push_back(0.0);
        }
         
        // Create one solver for group (to update parameters)
        llfunc_wrapper = [=](const vector<double>& p, 
            const map<string, double>& dd,
            const map<string, int>& di){
            return this->llfunc(p, dd, di);
        };
        dllfunc_wrapper = [=](const vector<double>& p,
            const map<string, double>& dd,
            const map<string, int>& di,
            vector<double>& r){
            this->dllfunc(p, dd, di, r);
        };
        
        solver_global = new multivar_ml_solver(params, llfunc_wrapper, dllfunc_wrapper);
        solver_global->set_delta(delta_thresh);
        solver_global->set_maxiter(maxiter);
        
        initialized = true;
    }

    void em_solver::add_param_grp(vector<double>& p){
        if (!initialized){
            fprintf(stderr, "ERROR: not initialized\n");
            exit(1);
        }
        for (int i = 0; i < n_components; ++i){
            solvers[i]->add_param_grp(p);
        }
        solver_global->add_param_grp(p);
    }
    
    void em_solver::set_threads(int nt){
        if (!initialized){
            fprintf(stderr, "ERROR: not initialized\n");
            exit(1);
        }
        for (int i = 0; i < n_components; ++i){
            solvers[i]->set_threads(nt);
        }
        solver_global->set_threads(nt);
    }
    
    void em_solver::set_bfgs_threads(int nt){
        if (!initialized){
            fprintf(stderr, "ERROR: not initialized\n");
            exit(1);
        }
        for (int i = 0; i < n_components; ++i){
            solvers[i]->set_bfgs_threads(nt);
        }
        solver_global->set_bfgs_threads(nt);
    }

    bool em_solver::add_data(std::string name, vector<double>& dat){
        if (!initialized){
            fprintf(stderr, "ERROR: not initialized\n");
            exit(1);
        }
        if (name == "j"){
            fprintf(stderr, "ERROR: name %s is reserved\n", name.c_str());
            return false;
        }
        if (no_data_yet){
            for (int i = 0; i < dat.size(); ++i){
                data_idx.push_back(i);
            }
            solver_global->add_data("j", data_idx);
            
            for (int i = 0; i < dat.size(); ++i){
                vector<double> llrow(n_components, 0.0);
                lls_tmp.push_back(llrow);
                lls_tmp_rowsum.push_back(0.0);
            }
            no_data_yet = false;
        }
        for (int i = 0; i < n_components; ++i){
            solvers[i]->add_data(name, dat);
        }
        solver_global->add_data(name, dat);
        return true;
    }

    bool em_solver::add_data(std::string name, vector<int>& dat){
        if (!initialized){
            fprintf(stderr, "ERROR: not initialized\n");
            exit(1);
        }
        if (name == "j"){
            fprintf(stderr, "ERROR: name %s is reserved\n", name.c_str());
            return false;
        }
        if (no_data_yet){
            vector<int> idx;
            for (int i = 0; i < dat.size(); ++i){
                idx.push_back(i);
            }
            solver_global->add_data("j", idx);
            no_data_yet = false;
        }
        for (int i = 0; i < n_components; ++i){
            solvers[i]->add_data(name, dat);
        }
        solver_global->add_data(name, dat);
        return true;
    }
    
    bool em_solver::add_data_fixed(std::string name, double dat){
        if (!initialized){
            fprintf(stderr, "ERROR: not initialized\n");
            exit(1);
        }
        for (int i = 0; i < n_components; ++i){
            solvers[i]->add_data_fixed(name, dat);
        }
        solver_global->add_data_fixed(name, dat);
        return true;
    }
    
    bool em_solver::add_data_fixed(std::string name, int dat){
        if (!initialized){
            fprintf(stderr, "ERROR: not initialized\n");
            exit(1);
        }
        for (int i = 0; i < n_components; ++i){
            solvers[i]->add_data_fixed(name, dat);
        }
        solver_global->add_data_fixed(name, dat);
        return true;
    }
    
    bool em_solver::constrain_pos(int idx){
        if (!initialized){
            fprintf(stderr, "ERROR: not initialized\n");
            exit(1);
        }
        for (int i = 0; i < n_components; ++i){
            solvers[i]->constrain_pos(idx);
        }
        solver_global->constrain_pos(idx);
        return true;
    }

    bool em_solver::constrain_01(int idx){
        if (!initialized){
            fprintf(stderr, "ERROR: not initialized\n");
            exit(1);
        }
        for (int i = 0; i < n_components; ++i){
            solvers[i]->constrain_01(idx);
        }
        solver_global->constrain_01(idx);
        return true;
    }
    
    void em_solver::add_weights(vector<double>& w){
        if (!initialized){
            fprintf(stderr, "ERROR: not initialized\n");
            exit(1);
        }
        for (int i = 0; i < w.size(); ++i){
            if (isnan(w[i]) || isinf(w[i])){
                fprintf(stderr, "ERROR: nan or inf in weight vector\n");
                exit(1);
            }
            weightsum += w[i];
        }
        weights_global = w; 
        solver_global->add_weights(weights_global);
    }
    
    /**
     * Note: everything to do with prior funcs is only important for M step - 
     * prior will apply equally to all component dists and therefore will
     * not affect the E step.
     */
    bool em_solver::add_prior(int idx, prior_func ll, prior_func dll){
        if (!initialized){
            return false;      
        }
        return solver_global->add_prior(idx, ll, dll);
    }

    bool em_solver::add_normal_prior(int idx, double mu, double sigma){
        if (!initialized){
            return false;
        }
        return solver_global->add_normal_prior(idx, mu, sigma);
    }

    bool em_solver::add_normal_prior(int idx, double mu, double sigma, double a, double b){
        if (!initialized){
            return false;
        }
        return solver_global->add_normal_prior(idx, mu, sigma, a, b);
    }

    bool em_solver::add_beta_prior(int idx, double alpha, double beta){
        if (!initialized){
            return false;
        }
        return solver_global->add_beta_prior(idx, alpha, beta);
    }

    bool em_solver::add_poisson_prior(int idx, double lambda){
        if (!initialized){
            return false;
        }
        return solver_global->add_poisson_prior(idx, lambda);
    }

    void em_solver::E_step(){
        if (n_obs == -1 || responsibility_matrix == NULL){
            // Learn how many data points
            int nd = solver_global->get_n_data();
            init_responsibility_matrix(nd);
        }
        
        double cstot = 0.0;
        
        vector<double> log_component_weights;
        
        for (int j = 0; j < n_components; ++j){
            if (component_weights[j] > 0){
                // Compute log likelihood of each observation under this model
                solvers[j]->eval_ll_all();
                log_component_weights.push_back(log(component_weights[j]));
            }
            else{
                log_component_weights.push_back(0.0);
            }
            compsums[j] = 0.0;
        }
        
        // Fill in responsibility matrix & weight matrix
        for (int i = 0; i < n_obs; ++i){
            if (weights_global.size() > 0 && weights_global[i] == 0.0){
                continue;
            }
            double rowmax = 0;
            for (int j = 0; j < n_components; ++j){
                if (component_weights[j] > 0.0){
                    //double ll = solvers[j]->data_ll[i] + log_component_weights[j];
                    double ll = solvers[j]->data_ll[i];
                    if (ll > rowmax){
                        rowmax = ll;
                    }
                }
            }
            double rowsum = 0.0;
            for (int j = 0; j < n_components; ++j){
                if (component_weights[j] > 0.0){
                    //rowsum += exp(solvers[j]->data_ll[i] + log_component_weights[j] - rowmax);
                    rowsum += exp(solvers[j]->data_ll[i] - rowmax);
                }
            }
            if (rowsum == 0.0){
                rowsum = 1.0;
            }
            rowsum = log(rowsum) + rowmax;
            for (int j = 0; j < n_components; ++j){
                if (component_weights[j] == 0.0){
                    responsibility_matrix[i][j] = 0.0;
                }
                else{
                    //responsibility_matrix[i][j] = exp(solvers[j]->data_ll[i] + log_component_weights[j] - rowsum);
                    responsibility_matrix[i][j] = exp(solvers[j]->data_ll[i] - rowsum);
                    double w = 1.0;
                    if (weights_global.size() > 0){
                        w = weights_global[i];
                    }
                    compsums[j] += w * responsibility_matrix[i][j];
                    cstot += w * responsibility_matrix[i][j];
                }
            }
        }
        
        // Update weights of components
        for (int j = 0; j < n_components; ++j){
            component_weights[j] = compsums[j] / cstot;
        }
    }
    
    /**
     * Returns log likelihood.
     */
    double em_solver::M_step(){
        // We already did the E-step
        solver_global->solve();
        
        // Update parameters for each dist
        for (int x = 0; x < params.size(); ++x){
            for (int j = 0; j < n_components; ++j){
                if (component_weights[j] > 0.0){
                    bool success = solvers[j]->set_param(x, solver_global->results[x]);
                    if (!success){
                        fprintf(stderr, "ERROR: could not set param %d in solver %d\n", x, j);
                        exit(1);
                    }
                }
            }
        }
        return solver_global->log_likelihood;
    }

    /**
     * Note: ignore weights_global here, since already incorporated into global solver
     */
    double em_solver::llfunc(const vector<double>& params,
        const map<string, double>& data_d, 
        const map<string, int>& data_i){
        
        // Get index into observation vector
        int idx = data_i.at("j");
        if (weights_global.size() > 0 && weights_global[idx] == 0.0){
            return 0.0;
        } 
         
        double llmax = 0.0;
        
        for (int j = 0; j < n_components; ++j){
            if (component_weights[j] > 0.0){
                
                lls_tmp[idx][j] = log(component_weights[j]) + 
                    this->dist_funcs[j](params, data_d, data_i);
                if (llmax == 0.0 || lls_tmp[idx][j] > llmax){
                    llmax = lls_tmp[idx][j];
                }
            }
        }

        
        if (llmax == 0.0){
            // ?? should not have made it this far
            return 0.0;
        }
        
        double llsum = 0.0;
        for (int j = 0; j < n_components; ++j){
            if (component_weights[j] > 0.0){
                llsum += exp(lls_tmp[idx][j] - llmax);
            }
        }
        
        llsum = log(llsum) + llmax;
        lls_tmp_rowsum[idx] = llsum;
        
        return llsum;
    }
    
    /**
     * Note: ignore weights_global here, since already incorporated into global solver
     */
    void em_solver::dllfunc(const vector<double>& params,
        const map<string, double>& data_d,
        const map<string, int>& data_i,
        vector<double>& results){
        
        // Get index into observation vector
        int idx = data_i.at("j");
        if (weights_global.size() > 0 && weights_global[idx] == 0.0){
            return;
        }
        
        for (int j = 0; j < n_components; ++j){
            if (component_weights[j] > 0.0){
                vector<double> dtmp(results.size(), 0.0);
                this->dist_funcs_deriv[j](params, data_d, data_i, dtmp);
                for (int i = 0; i < n_params; ++i){
                    results[i] += exp(lls_tmp[idx][j] - lls_tmp_rowsum[idx]) * dtmp[i];
                    //results[i] += responsibility_matrix[idx][j] * dtmp[i];
                }
            }
        }
    }
   
    double em_solver::fit(){
        // Make sure all have even weight - in case this was run previously
        for (int i = 0; i < n_components; ++i){
            component_weights[i] = 1.0 / (double)n_components;
        }

        double llprev = 0.0;
        double delta = 999;
        int it = 0;
        while (delta > delta_thresh && it < maxiter){
            E_step();
            double ll = M_step();
            if (llprev != 0.0){
                delta = ll - llprev;
            }
            llprev = ll;
            it++;
            if (n_components == 1){
                // Can't update weights - nothing else to do
                break;
            }
        }
        
        // Compute BIC & AIC
        if (weightsum > 0.0){
            bic = (double)(n_components-1 + params.size())*log(weightsum) - 2.0*llprev;
        }
        else{
            bic = (double)(n_components-1 + params.size())*log((double)n_obs) - 2.0*llprev;
        }
        aic = 2.0 * (double)(n_components-1 + params.size()) - 2.0*llprev;
        loglik = llprev;
        
        // Copy results from solver
        results.clear();
        for (int i = 0; i < solver_global->results.size(); ++i){
            results.push_back(solver_global->results[i]);
        }
        is_fit = true;
        return llprev;
    }

    /**
     * There may be a case where a model was fit with too many components, and
     * some components do not reflect true factors that generated the data.
     *
     * One strategy for dealing with this is to consider the BIC -- this penalizes
     * model complexity and can be used to choose a high-likelihood but simple model.
     *
     * BIC may be overzealous, though, in the case where one or more model components
     * are a good fit to a tiny subset of the data.
     *
     * In this case, this algorithm considers the Pearson correlation of 
     * responsibility matrix weights between every pair of components. A positive
     * correlation suggests that the two components tend to describe the same
     * observations.
     *
     * As long as there are positively correlated components, it chooses a single
     * one to remove, by storing, for each component, the sum of all positive
     * correlation coefficients involving that component, where the component was
     * lower weight than the other component in the comparison.
     *
     * The chosen component is then removed and the model is re-fit, and this continues
     * until no positive correlations are detected between components.
     */
    bool em_solver::rm_correlated_components(){
        if (!initialized){
            return false;
        }
        if (!is_fit){
            return false;
        }
        while (n_components > 1){
            map<pair<int, int>, double> corr_num;
            map<int, double> corr_denom;
            for (int i = 0; i < component_weights.size(); ++i){
                corr_denom.insert(make_pair(i, 0.0));
                for (int j = i + 1; j < component_weights.size(); ++j){
                    corr_num.insert(make_pair(make_pair(i,j), 0.0));
                }
            }
            
            for (int i = 0; i < n_obs; ++i){
                if (weights_global.size() == 0 || weights_global[i] > 0.0){
                    double w = 1.0;
                    if (weights_global.size() > 0){
                        w = weights_global[i];
                    }
                    for (int j = 0; j < component_weights.size(); ++j){
                        double elt1 = pow(responsibility_matrix[i][j] - component_weights[j], 2);
                        corr_denom[j] += w * elt1;
                        for (int k = j + 1; k < component_weights.size(); ++k){
                            pair<int, int> key = make_pair(j,k);
                            double elt2 = (responsibility_matrix[i][j] - component_weights[j]) * 
                                (responsibility_matrix[i][k] - component_weights[k]);
                            corr_num[key] += w * elt2;
                        }
                    }
                }
            }
            if (component_weights.size() > 0){
                for (map<int, double>::iterator cd = corr_denom.begin(); cd != corr_denom.end(); ++cd){
                    cd->second /= weightsum;
                }
            }
            double wsum = 1.0;
            if (component_weights.size() > 0){
                wsum = weightsum;
            }
            // Track maximum sum of positive correlations per component in which
            // that component is the "loser" (lower weight of the two)
            double maxpos = 0.0;
            int maxposcomp = -1;
            map<int, double> r2sumpos;
            for (map<pair<int, int>, double>::iterator cn = corr_num.begin(); cn != corr_num.end(); ++cn){
                cn->second /= wsum;
                // Calculate Pearson correlation
                double r = cn->second / sqrt(corr_denom[cn->first.first]*corr_denom[cn->first.second]);
                if (r > 0){
                    // Add weight to the loser component (lower model weight)
                    if (component_weights[cn->first.first] < component_weights[cn->first.second]){
                        if (r2sumpos.count(cn->first.first) == 0){
                            r2sumpos.insert(make_pair(cn->first.first, 0.0));
                        }
                        r2sumpos[cn->first.first] += r;
                        if (r2sumpos[cn->first.first] > maxpos){
                            maxpos = r2sumpos[cn->first.first];
                            maxposcomp = cn->first.first;
                        }
                    }
                    else{
                        r2sumpos[cn->first.second] += r;
                        if (r2sumpos[cn->first.second] > maxpos){
                            maxpos = r2sumpos[cn->first.second];
                            maxposcomp = cn->first.second;
                        }
                    }
                }         
            }
            if (maxposcomp >= 0){
                // Eliminate ALL pos components?
                for (map<int, double>::reverse_iterator r2s = r2sumpos.rbegin(); r2s != 
                    r2sumpos.rend(); ++r2s){
                    rm_component(r2s->first);
                }
                // Eliminate the chosen component.
                //rm_component(maxposcomp);
                // Re-set parameters to their original values
                for (int i = 0; i < params_orig.size(); ++i){
                    set_param(i, params_orig[i]);
                }
                fit();
            }
            else{
                // Finished
                return true;
            }
        }
        // Ran out of components to eliminate
        return true;
    }
    
    void em_solver::print(){
        if (!initialized){
            return;
        }
        fprintf(stderr, "LL = %.3f BIC %.3f AIC %.3f\n", loglik, bic, aic);
        fprintf(stderr, "  data weight sum %f\n", weightsum);
        fprintf(stderr, "  Params:\n");
        for (int i = 0; i < results.size(); ++i){
            fprintf(stderr, "    %d) %f\n", i, results[i]);   
        }
        fprintf(stderr, "  Component weights:\n");
        for (int i = 0; i < n_components; ++i){
            fprintf(stderr, "    ");
            if (component_names[i] != ""){
                fprintf(stderr, "%s) ", component_names[i].c_str());
            }
            else{
                fprintf(stderr, "%d) ", i);
            }
            fprintf(stderr, "%f\n", component_weights[i]);
        }
    }    
}
