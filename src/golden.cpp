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
#include "golden.h"

using std::cout;
using std::endl;
using namespace std;

// ----- Golden section search to find a maximum/minimum of a function
// ----- in a given interval without access to derivative information
//
// This algorithm is a very slight modification of the one presented in
// Chapter 10.3 of Numerical Recipes in C++ (Third Edition) by Press, 
// Teukolsky, Vetterling, & Flannery https://numerical.recipes/

golden_solver::golden_solver(golden_func f){
    this->func = f;
    this->has_prior = false;
    delta_thresh = 1e-6;
    maxiter = 1000;
}

void golden_solver::add_prior(golden_func p){
    this->has_prior = true;
    this->prior = p;
}

void golden_solver::set_delta(double d){
    this->delta_thresh = d;    
}

void golden_solver::set_maxiter(int i){
    this->maxiter = i;
}

void golden_solver::print(double min, double max, double step){
    for (double x = min; x <= max; x += step){
        double y = eval(x);
        fprintf(stdout, "%f\t%f\n", x, y);
    }
}

double golden_solver::solve(double lower, double upper, bool max){

    double tol = delta_thresh;
    double edge = 1e-6;

    double x0 = lower;
    double x3 = upper;

    double R = 0.61803399;
    double C = 1.0 - R;
    
    // Choose third point as midway between x0 and x3
    double x1 = (x0 + x3)/2.0;
    double x2 = x1 + C*(x3-x1);

    double f1 = eval(x1);
    double f2 = eval(x2);
    
    double xmax;
    int nit = 0;
    while (nit < maxiter && abs(x3-x0) > tol*(abs(x1) + abs(x2))){
        ++nit;
        if ((max && f2 > f1) || (!max && f2 < f1)){
            x0 = x1;
            x1 = x2;
            x2 = R*x2 + C*x3;   
            
            f1 = f2;
            f2 = eval(x2);
        }
        else{
            x3 = x2;
            x2 = x1;
            x1 = R*x1 + C*x0;

            f2 = f1;
            f1 = eval(x1);
        }
    }
    if ((max && f1 > f2) || (!max && f1 < f2)){
        this->y = f1;
        xmax = x1;
    }
    else{
        this->y = f2;
        xmax = x2;
    }
    
    return xmax;
}

// Add in data for log likelihood function
bool golden_solver::add_data(string name, std::vector<double>& dat){
    // Make sure dimensions agree
    int nd = dat.size();
    if (this->n_data != 0 && this->n_data != nd){
        fprintf(stderr, "ERROR: data vectors do not have same dimensions\n");
        return false;
    }
    if (param_double_cur.count(name) > 0){
        //fprintf(stderr, "WARNING: %s already keyed to data. Overwriting\n", name.c_str());
        int idx = -1;
        for (int i = 0; i < params_double_names.size(); ++i){
            if (params_double_names[i] == name){
                idx = i;
                break;
            }
        }
        params_double_vals[idx] = dat.data();
        //return false;
        return true;
    }
    this->n_data = nd;
    this->params_double_names.push_back(name);
    this->params_double_vals.push_back(dat.data());
    this->param_double_cur.insert(make_pair(name, 0.0));
    this->param_double_ptr.push_back(&(this->param_double_cur.at(name)));
    return true;
}

bool golden_solver::add_data(string name, std::vector<int>& dat){
    // Make sure dimensions agree
    int nd = dat.size();
    if (this->n_data != 0 && this->n_data != nd){
        fprintf(stderr, "ERROR: data vectors do not have same dimensions\n");
        return false;
    }
    if (this->param_int_cur.count(name) > 0){
        fprintf(stderr, "ERROR: %s already keyed to data\n", name.c_str());
        return false;
    }
    this->n_data = nd;
    this->params_int_names.push_back(name);
    this->params_int_vals.push_back(dat.data());
    this->param_int_cur.insert(make_pair(name, 0.0));
    this->param_int_ptr.push_back(&(this->param_int_cur.at(name)));
    return true;
}

bool golden_solver::add_data_fixed(string name, double dat){
    if (param_double_cur.count(name) > 0){
        fprintf(stderr, "ERROR: %s already keyed to data\n", name.c_str());
        return false;
    }
    param_double_cur.insert(make_pair(name, dat));
    return true;
}

bool golden_solver::add_data_fixed(string name, int dat){
    if (param_int_cur.count(name) > 0){
        fprintf(stderr, "ERROR: %s already keyed to data\n", name.c_str());
        return false;
    }
    param_int_cur.insert(make_pair(name, dat));
    return true;
}

bool golden_solver::add_weights(vector<double>& weights){
    if (this->n_data != 0 && this->n_data != weights.size()){
        fprintf(stderr, "ERROR: dimension of weights does not equal dimension of data\n");
        return false;
    }
    this->weights = weights;
    return true;
}

// Add in data for prior
bool golden_solver::add_prior_param(string name, double dat){
    this->params_prior_double.insert(make_pair(name, dat));
    return true;
}
bool golden_solver::add_prior_param(string name, int dat){
    this->params_prior_int.insert(make_pair(name, dat));
    return true;
}

// Evaluate the function at a given point
double golden_solver::eval(double x){
    double f_x_all = 0.0;

    for (int i = 0; i < this->n_data; ++i){
        for (int j = 0; j < this->params_double_names.size(); ++j){
            *(this->param_double_ptr[j]) = (this->params_double_vals[j][i]);
        }
        for (int j = 0; j < this->params_int_names.size(); ++j){
            *(this->param_int_ptr[j]) = (this->params_int_vals[j][i]);
        }
        double f_x = this->func(x, this->param_double_cur, this->param_int_cur);
        if (isnan(f_x) || isinf(f_x)){
            fprintf(stderr, "ERROR: nan or inf from function\n");
            fprintf(stderr, "parameter: %f\n", x);
            fprintf(stderr, "data:\n");
            for (map<string, double>::iterator it = param_double_cur.begin(); it != 
                param_double_cur.end(); ++it){
                fprintf(stderr, "%s = %f\n", it->first.c_str(), it->second);
            }
            for (map<string, int>::iterator it = param_int_cur.begin(); it != 
                param_int_cur.end(); ++it){
                fprintf(stderr, "%s = %d\n", it->first.c_str(), it->second);
            }
            exit(1);
        }
        if (this->weights.size() > 0){
            f_x *= weights[i];
        }
        f_x_all += f_x;
    }
    if (this->has_prior){
        double f_prior = prior(x, this->params_prior_double, this->params_prior_int);
        if (isinf(f_prior) || isnan(f_prior)){
            fprintf(stderr, "ERROR: illegal value from prior function\n");
            fprintf(stderr, "parameter: %f\n", x);
            fprintf(stderr, "prior dist parameters:\n");
            for (map<string, double>::iterator it = params_prior_double.begin(); it != 
                params_prior_double.end(); ++it){
                fprintf(stderr, "%s = %f\n", it->first.c_str(), it->second);
            }
            for (map<string, int>::iterator it = params_prior_int.begin(); it != 
                params_prior_int.end(); ++it){
                fprintf(stderr, "%s = %d\n", it->first.c_str(), it->second);
            }
            exit(1);
        }
        f_x_all += f_prior;
    }
    return f_x_all;
}

