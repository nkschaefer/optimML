#include <string>
#include <algorithm>
#include <vector>
#include <iterator>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <array>
#include <cmath>
#include <sstream>
#include <set>
#include <cstdlib>
#include <utility>
#include <math.h>
#include "stlbfgs/stlbfgs.h"
#include "functions.h"

using std::cout;
using std::endl;
using namespace std;

// ===== Functions used across multiple source files =====

/**
 * Logit transform (quantile function of logistic distribution)
 */
double logit(double x){
    return log(x) - log(1.0-x);
    //return log(x/(1.0-x));
}

/**
 * Inverse logit function / Expit function / Logistic function
 */
double expit(double num){
   double val;
    if (num >= 0) {
        double z = exp(-num);
        val = 1.0 / (1.0 + z);
    } else {
        double z = exp(num);
        val = z / (1.0 + z);
    }
    
    double bumper = 1e-8;
    if (val < bumper){
        val = bumper;
    }
    else if (val > 1.0-bumper){
        val = 1.0 - bumper;
    }
    return val;
    //return 1.0/(1.0 + exp(-x));
}

/**
 * Uses a moderately-accurate version of Stirling's approximation for
 * the factorial. Should not use this if the exact value is very
 * important.
 */
double poisll(double x, double l){
    if (l < 1){
        return -1;
    }
    double xfac = x*log(x) - x;
    return x * log(l) + -((float)l) - xfac;
}

/**
 * Lightweight wrapper for L-BFGS solver
 * ARGS: parameters (will be modified with results), function
 *      function params: parameters, log likelihood, gradient
 * REMEMBER: BFGS minimizes, so if maximizing (e.g. log likelihood),
 *   make gradient & LL negative
 */
void bfgs(vector<double>& params,
    function<void(const vector<double>&, double&, vector<double>&)> func){

    STLBFGS::Optimizer opt{func, 1, 10};
    opt.verbose = false;
    opt.ftol = 1e-6;
    opt.maxiter = 100;
    double res = opt.run(params);
}

/**
 * Univariate Nelder-Mead algorithm: finds value of independent variable
 * that maximizes the function value, given an initial guess.
 *
 * Returns: function evaluation at optimum
 *
 * Arguments:
 * f: function to optimize
 * guess: initial guess for independent variable
 * best: best guess
 * step: step size
 * tol: step tolerance to determine convergence
 * maxit: maximum number of iterations
 */
double nelder_mead_1d(const std::function<double(double)>& f,
    double guess, 
    double& best,
    double step,
    double tol, 
    int maxit){

    // --- The 1D "simplex": two points a (better) and b (worse). ---
    double a = guess;
    double b = guess + step;
    double fa = f(a);
    double fb = f(b);
    // Sort so a is the better (higher) point.
    if (fb > fa) { 
        swap(a, b); 
        swap(fa, fb); 
    }

    // Standard coefficients.
    const double alpha = 1.0;   // reflection
    const double gamma = 2.0;   // expansion
    const double rho   = 0.5;   // contraction

    for (int it = 0; it < maxit; ++it) {
        // Convergence: the two points are close, or their values are.
        if (fabs(b - a) < tol) break;

        // --- Reflection: reflect worst (b) through best (a). ---
        // In 1D the centroid of "all but worst" is just a.
        double r  = a + alpha * (a - b);
        double fr = f(r);

        if (fr > fa) {
            // --- Reflection beat the best -> try expanding further. ---
            double e  = a + gamma * (r - a);
            double fe = f(e);
            if (fe > fr){ 
                b = e; 
                fb = fe; 
            }   // expansion better
            else{ 
                b = r; 
                fb = fr; 
            }   // reflection better
        }
        else {
            // --- Reflection didn't beat the best -> contract inward. ---
            // Contract between best (a) and worst (b).
            double c  = a + rho * (b - a);
            double fc = f(c);
            if (fc > fb) { 
                b = c; 
                fb = fc; 
            }   // contraction helped
            else {
                // --- Shrink: pull worst halfway to best. ---
                b  = a + 0.5 * (b - a);
                fb = f(b);
            }
        }

        // Re-sort so a stays the better point.
        if (fb > fa) { 
            swap(a, b); 
            swap(fa, fb); 
        }
    }
    
    best = a;
    return fa;
}

/**
 * Bivariate Nelder-Mead algorithm: finds value of two independent variables
 * that together maximize the function value, given an initial guess.
 *
 * Returns: function evaluation at optimum
 *
 * Arguments:
 * f: function to optimize
 * gx: independent variable 1 initial guess
 * gy: independent variable 2 initial guess
 * outx: independent variable 1 at optimum
 * outy: independent variable 2 at optimum
 * stepx: step size for moving independent variable 1
 * stepy: step size for moving independent variable 2
 * tol: step tolerance to determine convergence
 * maxit: maximum number of iterations
 */
double nelder_mead_2d(const std::function<double(double,double)>& f,
    double gx, 
    double gy,
    double& outx, 
    double& outy,
    double stepx, 
    double stepy,
    double tol, 
    int maxit){

    // --- The 2D simplex: THREE vertices. ---
    array<array<double, 2>, 3> v = {
        array<double, 2>{gx,gy},
        array<double, 2>{gx + stepx, gy},
        array<double, 2>{ gx, gy + stepy}
    };
    array<double, 3> fv = { f(v[0][0], v[0][1]),
        f(v[1][0], v[1][1]),
    f(v[2][0], v[2][1]) };

    const double alpha = 1.0;   // reflection
    const double gamma = 2.0;   // expansion
    const double rho = 0.5;   // contraction
    const double sigma = 0.5;   // shrink

    auto eval = [&](const array<double, 2>& p){ return f(p[0], p[1]); };

    for (int it = 0; it < maxit; ++it) {
        // 1. Sort so fv[0] >= fv[1] >= fv[2]  (best = highest = index 0,
        //    worst = lowest = index 2).
        for (int i = 0; i < 3; ++i){
            for (int j = i + 1; j < 3; ++j){
                if (fv[j] > fv[i]) { 
                    swap(fv[i], fv[j]); 
                    swap(v[i], v[j]); 
                }
            }
        }

        // 2. Convergence: are all vertices close together?
        double d1 = hypot(v[1][0]-v[0][0], v[1][1]-v[0][1]);
        double d2 = hypot(v[2][0]-v[0][0], v[2][1]-v[0][1]);
        if (d1 < tol && d2 < tol){
            break;
        }

        // 3. Centroid of all vertices EXCEPT the worst (v[2]):
        //    midpoint of the two better vertices.
        array<double, 2> c = { 0.5*(v[0][0]+v[1][0]),
                               0.5*(v[0][1]+v[1][1]) };

        // 4. Reflection: reflect the worst through the centroid.
        array<double, 2> r = { c[0] + alpha*(c[0]-v[2][0]),
                               c[1] + alpha*(c[1]-v[2][1]) };
        double fr = eval(r);

        if (fr > fv[0]) {
            // 5. Reflection is a new best (higher) -> try expanding further.
            array<double, 2> e = { c[0] + gamma*(r[0]-c[0]),
                                   c[1] + gamma*(r[1]-c[1]) };
            double fe = eval(e);
            if (fe > fr) { 
                v[2] = e; 
                fv[2] = fe; 
            }   // expansion better
            else{ 
                v[2] = r; 
                fv[2] = fr; 
            }   // reflection better
        }
        else if (fr > fv[1]) {
            // 6. Reflection better than second-worst: accept it.
            v[2] = r; 
            fv[2] = fr;
        }
        else {
            // 7. Reflection didn't help enough -> contract.
            if (fr > fv[2]) {
                // Outside contraction: between centroid and reflection.
                array<double, 2> cc = { c[0] + rho*(r[0]-c[0]),
                                        c[1] + rho*(r[1]-c[1]) };
                double fcc = eval(cc);
                if (fcc >= fr) { 
                    v[2] = cc; 
                    fv[2] = fcc; 
                }
                else{
                    goto shrink;
                }
            } 
            else{
                // Inside contraction: between centroid and worst.
                array<double, 2> cc = { c[0] + rho*(v[2][0]-c[0]),
                                        c[1] + rho*(v[2][1]-c[1]) };
                double fcc = eval(cc);
                if (fcc > fv[2]) { 
                    v[2] = cc; 
                    fv[2] = fcc; 
                }
                else{
                    goto shrink;
                }
            }
            continue;

            shrink:
            // 8. Shrink: pull the two worse vertices halfway to the best.
            for (int i = 1; i < 3; ++i){
                v[i][0] = v[0][0] + sigma*(v[i][0]-v[0][0]);
                v[i][1] = v[0][1] + sigma*(v[i][1]-v[0][1]);
                fv[i] = eval(v[i]);
            }
        }
    }

    // Best (highest) vertex.
    int best = 0;
    for (int i = 1; i < 3; ++i){
        if (fv[i] > fv[best]){
            best = i;
        }
    }
    outx = v[best][0];
    outy = v[best][1];
    return fv[best];
}


