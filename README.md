# optimML
A fast and flexible C++ library for numeric optimization of complex log likelihood functions, including mixture components that must sum to 1

## Use case
This library was designed to make it easy to find maximum likelihood estimates (MLE) or maximum a posteriori estimates (MAP), given arbitrary complex log likelihood functions to be evaluated on a data set consisting of many observations. It includes classes designed to:
* Maximize univariate log likelihood functions by finding a root of the derivative within a fixed interval (using Brent's method)
  * Can optionally estimate the standard error of the MLE/MAP estimate using the Fisher information, if a function for evaluating the second derivative is provided
* Maximize univariate log likelihood functions (or any other function) within a fixed interval without derivative information, using golden section search
* Maximize multivariate log likelihood functions, given initial parameter guesses, using BFGS
  
Some nice features it has are:
* The ability to add data points as named vectors, which can be accessed by outside functions and looked up by name
* The ability to add some pre-set prior distributions to calculations
* The ability to constrain variables to (0, infinity] by log transformation or (0,1) by logit transformation automatically
  * Automatically handles these transformations and makes the gradient depend on the un-transformed versions of the variables
* The ability to model mixture proportions (see below)
## Mixture proportions
There are some special cases where you have collected observations that are thought to result from a mixture (in unknown proportions) of components, each contributing a known expected value to the result. For example, suppose you have sequenced a pool of individuals thought to belong to three populations, and you want to know what percent of the pool is made up of individuals of each population. 

Your data consist of a series of $n$ allele frequencies measured at different alleles: $A = A_1, A_2, A_3 ... A_n$

You are interested in modeling your data as coming from a mixture of three populations $P_1$, $P_2$, and $P_3$

For any given allele, you know the expected allele frequency in population 1, 2, and 3: $f_{i1} = E[A_i | P_1], f_{i2} = E[A_i | P_2], f_{i3} = E[A_i | P_3]$

You want to solve for the mixture components $m_1$, $m_2$, and $m_3$, where each denotes the proportion of the pool made up of individuals from each population, and $\sum_{j=1}^{3}(m_j) = 1$

To handle the requirement that for each $m_j$, $0 < m_j < 1$, and that all must sum to 1, each variable is logit transformed, and each appears in the log likelihood function as follows: $t(m_j) = \frac{\frac{1}{e^{-m_j} + 1}}{\sum_{k=1}^n \frac{1}{e^{-m_k} + 1}}$
## Requirements
Only requires a C++11 compiler. Also depends on the [stlbfgs](https://github.com/nkschaefer/stlbfgs) library, which is included as a submodule. The original repository is [here](https://github.com/ultimaille/stlbfgs), and the forked version was modified to be compatible with older compilers. 
