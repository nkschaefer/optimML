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
#include <set>
#include <cstdlib>
#include <utility>
#include <math.h>
#include "functions.h"

using std::cout;
using std::endl;
using namespace std;

// ===== Functions used across multiple source files =====

/**
 * Logit transform (quantile function of logistic distribution)
 */
double logit(double x){
    return log(x/(1.0-x));
}

/**
 * Inverse logit function / Expit function / Logistic function
 */
double expit(double x){
   return 1.0/(1.0 + exp(-x));
}

