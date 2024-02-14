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
#include "functions.h"
#include "multivar_ml.h"
#include "mixcomp.h"

using std::cout;
using std::endl;
using namespace std;

// ===== mixcomp =====

// A wrapper class that makes it easy to solve problems that only solve
// for a set of mixture proportions.
//
// These problems are similar in form to this:
// 
// You measure a set of allele frequencies, each denoted f
// 
// You think the group you sampled from is a mixture of 3 populations P1
// P2, and P3, but you don't know what percent of the pool is composed of
// each population. Let these unknown proportions be p1, p2, and p3, where
// 0 < p < 1 for each, and where p1 + p2 + p3 = 1.
// 
// Each allele has an expected frequency e1 in P1, e2 in P2, and e3 in P3.
//
// You therefore want to learn p1, p2, and p3, keeping all between 0 and 1
// and requiring that they sum to 1. Each piece of data you have collected
// can be compared to your model like this:
//
// Observed allele freq: f | Expected allele freq: E = p1e1 + p2e2 + p3e3
//
// You can use several pre-sets to compare the expected allele frequencies
// E to whatever data (like f) you have. Each f can be a fixed number you 
// compare to E with least squares, or you can provide mu & sigma for a 
// Normal distribution, or alpha and beta for a beta distribution. You can
// also supply a custom function.

// ----- Find the most likely set of mixture proportions that produced -----
// ----- a set of observations                                         ----- 

namespace optimML{
    /**
     * Cost function least squares
     */
    double mixcomp_solver::y_ls(vector<double>& x, const map<string, double>& params_d, 
        const map<string, int>& params_i){
        // NOTE: this will seek to maximize the function, so we need to make this negative.
        return -pow(x[0] - params_d.at("y"), 2);
    }

    /**
     * dy_dx least squares
     */
    void mixcomp_solver::dy_dx_ls(vector<double>& x, const map<string, double>& params_d, 
        const map<string, int>& params_i, vector<double>& results){
        // Again, operating on negative sum of squares instead of positive, so we can
        // maximize
        results[0] = -2*x[0] + 2*params_d.at("y");
    }

    /**
     * d^2y/dx^2 least squares
     */
    void mixcomp_solver::d2y_dx2_ls(vector<double>& x, const map<string, double>& params_d,
        const map<string, int>& params_i, vector<vector<double> >& results){
        // Operating on negative sum of squares instead of positive, so we can maximize
        results[0][0] = -2.0;
    }

    /**
     * Log likelihood Normal
     *
     * Note that for truncated normal, derivatives don't depend on the
     * normalization, so optimizing a normal is the same as optimizing
     * a truncated normal
     */
    double mixcomp_solver::y_norm(vector<double>& x, const map<string, double>& params_d, 
        const map<string, int>& params_i){
        return -0.5*pow((x[0]-params_d.at("mu"))/params_d.at("sigma"), 2) - 
            log(params_d.at("sigma")) - 
            log(sqrt(2*M_PI));
    }

    /**
     * dy_dx Normal
     */
    void mixcomp_solver::dy_dx_norm(vector<double>& x, const map<string, double>& params_d,
        const map<string, int>& params_i, vector<double>& results){
        results[0] = (x[0]-params_d.at("mu"))/(params_d.at("sigma") * params_d.at("sigma"));
    }

    /**
     * d^2y/dx^2 Normal
     */
    void mixcomp_solver::d2y_dx2_norm(vector<double>& x, const map<string, double>& params_d,
        const map<string, int>& params_i, vector<vector<double> >& results){
        results[0][0] = 1.0/(params_d.at("sigma") * params_d.at("sigma"));
    }

    /**
     * Log likelihood Beta
     */
    double mixcomp_solver::y_beta(vector<double>& x, const map<string, double>& params_d,
        const map<string, int>& params_i){
        double alpha = params_d.at("alpha");
        double beta = params_d.at("beta");
        return (alpha-1.0)*log(x[0]) + (beta-1.0)*log(1.0-x[0]) + 
            lgamma(alpha+beta) - lgamma(alpha) - lgamma(beta);
    }

    /**
     * dy_dx Beta
     */
    void mixcomp_solver::dy_dx_beta(vector<double>& x, const map<string, double>& params_d,
        const map<string, int>& params_i, vector<double>& results){
        double alpha = params_d.at("alpha");
        double beta = params_d.at("beta");
        results[0] = (alpha- 1.0)/x[0] - (beta - 1.0)/(1.0-x[0]);
    }

    /**
     * d^2y_dx^2 Beta
     */
    void mixcomp_solver::d2y_dx2_beta(vector<double>& x, const map<string, double>& params_d,
        const map<string, int>& params_i, vector<vector<double> >& results){
        double alpha = params_d.at("alpha");
        double beta = params_d.at("beta");
        results[0][0] = (1.0-alpha)/(x[0]*x[0]) + (1.0-beta)/pow(1.0-x[0], 2);
    }

    /**
     * Log likelihood Binomial
     */
    double mixcomp_solver::y_binom(vector<double>& x, const map<string, double>& params_d,
        const map<string, int>& params_i){
        double n = params_d.at("n");
        double k = params_d.at("k");
        
        double p = x[0];
        double ll = k * log(p) + (n-k)*log(1.0-p);
        
        // Compute log binomial coefficient
        if (k < n && k != 0){
            // Use Stirling's approximation
            ll += n*log(n) - k*log(k) - (n-k)*log(n-k) + 0.5*(log(n) - log(k) - log(n-k) - log(2*M_PI));
        }
        return ll;
    }

    /**
     * dy_dx Binomial
     */
    void mixcomp_solver::dy_dx_binom(vector<double>& x, const map<string, double>& params_d,
        const map<string, int>& params_i, vector<double>& results){
        double n = params_d.at("n");
        double k = params_d.at("k");
        double p = x[0];
        results[0] = (k-n*p)/(p - p*p);
    }

    /**
     * d^2y_dx^2 Binomial
     */
    void mixcomp_solver::d2y_dx2_binom(vector<double>& x, const map<string, double>& params_d,
        const map<string, int>& params_i, vector<vector<double> >& results){
        double n = params_d.at("n");
        double k = params_d.at("k");
        double p = x[0];
        
        results[0][0] = (k*(2*p - 1) - n*p*p)/(pow(p-1, 2) * p*p);
    }

    mixcomp_solver::mixcomp_solver(vector<vector<double> >& mixfracs, 
        string preset, vector<double>& data){
        if (preset == "ls" || preset == "LS" || preset == "lstsq"){
            solver = multivar_ml_solver({}, y_ls, dy_dx_ls);
            solver.add_data("y", data);       
            solver.add_mixcomp(mixfracs);
        }
        else{
            solver = multivar_ml_solver({}, y_ls, dy_dx_ls);
            fprintf(stderr, "ERROR: preset %s not recognized or requires additional data\n", preset.c_str());
            exit(1);
        }
    }

    mixcomp_solver::mixcomp_solver(vector<vector<double> >& mixfracs,
        string preset, vector<double>& data1, vector<double>& data2){
        
        if (preset == "normal" || preset == "Normal" || preset == "gaussian" ||
            preset == "Gaussian" || preset == "norm" || preset == "gauss"){
            solver = multivar_ml_solver({}, y_norm, dy_dx_norm);
            solver.add_data("mu", data1);
            solver.add_data("sigma", data2);
            solver.add_mixcomp(mixfracs);
            return;
        }
        else if (preset == "beta" || preset == "Beta"){
            solver = multivar_ml_solver({}, y_beta, dy_dx_beta);
            solver.add_data("alpha", data1);
            solver.add_data("beta", data2);
            solver.add_mixcomp(mixfracs);
            return;
        }
        else if (preset == "binom" || preset == "binomial" || preset == "Binomial"){
            solver = multivar_ml_solver({}, y_binom, dy_dx_binom);
            solver.add_data("n", data1);
            solver.add_data("k", data2);
            solver.add_mixcomp(mixfracs);
            return;
        }
        // This line is only here because we're required to initialize
        solver = multivar_ml_solver({}, y_norm, dy_dx_norm);
        fprintf(stderr, "ERROR: preset string %s does not match a known preset, or you have\n", preset.c_str());
        fprintf(stderr, "provided the wrong number of input data variables\n");
        exit(1);
    }

    bool mixcomp_solver::add_mixcomp_fracs(vector<double>& fracs){
        return solver.add_mixcomp_fracs(fracs);
    }

    void mixcomp_solver::randomize_mixcomps(){
        solver.randomize_mixcomps();
    }

    bool mixcomp_solver::add_mixcomp_prior(vector<double>& alphas){
        return solver.add_mixcomp_prior(alphas);
    }

    void mixcomp_solver::set_delta(double d){
        solver.set_delta(d);
    }

    void mixcomp_solver::set_maxiter(int i){
        solver.set_maxiter(i);
    }

    bool mixcomp_solver::solve(){
        bool success = solver.solve();
        results.clear();
        for (int i = 0; i < solver.results_mixcomp.size(); ++i){
            results.push_back(solver.results_mixcomp[i]);
        }
        log_likelihood = solver.log_likelihood;
        return success;
    }
}
