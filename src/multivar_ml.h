#ifndef _OPTIMML_MULTIVAR_ML_H
#define _OPTIMML_MULTIVAR_ML_H
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
#include "multivar.h"

namespace optimML{
    
    class multivar_ml_solver: public multivar{
       
        protected:
            
            const void eval_funcs_bfgs(const std::vector<double>& x_bfgs, 
                double& f_bfgs, std::vector<double>& g_bfgs);
            
            void explore_starting_mixcomps_aux(std::set<int>& elim, double& ll, 
                std::vector<double>& params, const std::vector<double>& params_orig);  


        public:
           
            multivar_ml_solver();

            multivar_ml_solver(std::vector<double> params_init, multivar_func ll,
                multivar_func_d dll);
            
            bool solve();

            bool explore_starting_mixcomps();
    };

}

#endif
