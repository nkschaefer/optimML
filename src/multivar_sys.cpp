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

    // Set up solver with initial guesses for parameter values
    multivar_sys_solver::multivar_sys_solver(vector<double> params_init){
        
        this->params = params_init;
        
        n_equations = 0;
        
        // Make room for temporary storage of equation gradients
        for (int i = 0; i < params_init.size(); ++i){
            this->dyi_dx.push_back(0.0);
            this->results.push_back(0.0);
        }

        this->rss = -1;
    }

    // Add an equation
    bool multivar_sys_solver::add_equation(multivar_sys_func eq,
        multivar_sys_func_d d_eq,
        double rhs){
        
        this->rhs.push_back(rhs);
            
        equations.push_back(eq);
        equations_deriv.push_back(d_eq);
        this->y.push_back(0.0);

        n_equations++;
        return true;
    }


    void multivar_sys_solver::constrain_pos(int idx){
        this->solver.constrain_pos(idx);
    }

    void multivar_sys_solver::constrain_01(int idx){
        this->solver.constrain_01(idx);
    }
        
    // Here, what would normally be an index into the data vector -- corresponding
    // to a single observation -- is an index into the set of equations.

    // We only need to return the squared residual for a single equation.

    double multivar_sys_solver::eval_sum_sq(const vector<double>& params,
        const map<string, double>& data_d,
        const map<string, int>& data_i){
        
        int eq_idx = data_i.at("eq_idx");
        
        // Evaluate equation
        double result = this->equations[eq_idx](params);
        
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
        
        // Evaluate current function's derivative wrt all params 
        this->equations_deriv[eq_idx](params, this->dyi_dx);
        
        for (int i = 0; i < params.size(); ++i){
            results[i] = drss_dy_j * dyi_dx[i];    
        }    
    }

    bool multivar_sys_solver::solve(){
        
        std::function<double(const vector<double>&, const map<string, double>&, 
            const map<string, int>&)> f = 
            [=](const vector<double> params, const map<string, double>& data_d,
                const map<string, int>& data_i){
           return this->eval_sum_sq(params, data_d, data_i);
        };

        std::function<void(const vector<double>&, const map<string, double>&, 
            const map<string, int>&, vector<double>&)> fprime = 
            [=](const vector<double> params, const map<string, double>& data_d,
                const map<string, int>& data_i, vector<double>& results){
           this->eval_dsum_sq(params, data_d, data_i, results);
        };

        this->solver.init(this->params, f, fprime);
        
        // Add data
        this->solver.add_data("rhs", this->rhs);
        vector<int> indices;
        for (int i = 0; i < n_equations; ++i){
            indices.push_back(i);
        }
        this->solver.add_data("eq_idx", indices);

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
    

}

