#ifndef _OPTIMML_MIXCOMP_H
#define _OPTIMML_MIXCOMP_H
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
#include "multivar_ml.h"

// ===== Mixture component proportion solver =====
// 
// This is a wrapper around multivar_ml_solver, for a sub-class
// of problems that class is designed to solve. This allows users
// to easily compare results of a mixture of components to observed
// data, using a preset likelihood function.

namespace optimML{
    /**
     * This class uses Newton-Raphson (with optional constraints) to find maximum 
     * likelihood estimates of an array of input variables.
     */
    class mixcomp_solver{
        private: 
            
            // The object that will actually do all the work
            multivar_ml_solver* solver;
            
            // Preset functions
            
            // --- Least squares ---
            // Cost function / log likelihood like (negative sum of squares)
            static double y_ls(const std::vector<double>& params, 
                const std::map<std::string, double>& data_d,
                const std::map<std::string, int>& data_i);
            // First derivative
            static void dy_dx_ls(const std::vector<double>& params, 
                const std::map<std::string, double>& data_d,
                const std::map<std::string, int>& data_i, std::vector<double>& results);
            // Second derivative
            static void d2y_dx2_ls(const std::vector<double>& params, 
                const std::map<std::string, double>& data_d,
                const std::map<std::string, int>& data_i, std::vector<std::vector<double> >& results);
            
            // --- Normal ---
            // Log likelihood function
            static double y_norm(const std::vector<double>& params, 
                const std::map<std::string, double>& data_d,
                const std::map<std::string, int>& data_i);
            // First derivative
            static void dy_dx_norm(const std::vector<double>& params, 
                const std::map<std::string, double>& data_d,
                const std::map<std::string, int>& data_i, std::vector<double>& results);
            // Second derivative
            static void d2y_dx2_norm(const std::vector<double>& params, 
                const std::map<std::string, double>& data_d,
                const std::map<std::string, int>& data_i, std::vector<std::vector<double> >& results);

            // --- Beta ---
            // Log likelihood function
            static double y_beta(const std::vector<double>& params, 
                const std::map<std::string, double>& data_d,
                const std::map<std::string, int>& data_i);
            // First derivative
            static void dy_dx_beta(const std::vector<double>& params, 
                const std::map<std::string, double>& data_d,
                const std::map<std::string, int>& data_i, std::vector<double>& results);
            // Second derivative
            static void d2y_dx2_beta(const std::vector<double>& params, 
                const std::map<std::string, double>& data_d,
                const std::map<std::string, int>& data_i, std::vector<std::vector<double> >& results);

            // --- Binomial ---
            // Log likelihood function
            static double y_binom(const std::vector<double>& params, 
                const std::map<std::string, double>& data_d,
                const std::map<std::string, int>& data_i);
            // First derivative
            static void dy_dx_binom(const std::vector<double>& params, 
                const std::map<std::string, double>& data_d,
                const std::map<std::string, int>& data_i, std::vector<double>& results);
            // Second derivative
            static void d2y_dx2_binom(const std::vector<double>& params, 
                const std::map<std::string, double>& data_d,
                const std::map<std::string, int>& data_i, std::vector<std::vector<double> >& results);

        public:
            
            // Constructors
            mixcomp_solver(std::vector<std::vector<double> >& mixfracs,
                std::string preset, std::vector<double>& data);
            mixcomp_solver(std::vector<std::vector<double> >& mixfracs,
                std::string preset, std::vector<double>& data1, 
                std::vector<double>& data2);
            ~mixcomp_solver();

            bool add_mixcomp_fracs(std::vector<double>& fracs);
            void randomize_mixcomps();
            bool add_mixcomp_prior(std::vector<double>& alphas);
            void set_delta(double d);
            void set_maxiter(int i); 
            void set_threads(int nt);
            void set_threads_bfgs(int nt);
            bool solve();

            // Where to look after the routine completes
            double log_likelihood;
            std::vector<double> results;

    };
}
#endif
