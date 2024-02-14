#ifndef _OPTIMML_GOLDEN_H
#define _OPTIMML_GOLDEN_H
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
#include <set>
#include <cstdlib>
#include <utility>
#include <math.h>
#include "functions.h"
#include "univar.h"

namespace optimML{
    class golden_solver: public univar{
        private:
            
            // Set to false if we're looking for minimum instead of maximum
            bool max;
        public:

            golden_solver(univar_func ll);
            void set_min();
            void set_max();
            double solve(double min, double max);
    };
}

#endif
