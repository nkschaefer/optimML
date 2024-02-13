# optimML
A fast and flexible C++ library for numeric optimization of complex log likelihood functions, including mixture components that must sum to 1

## Use case
This library was designed to make it easy to find maximum likelihood estimates (MLE) or maximum a posteriori estimates (MAP), given arbitrary complex log likelihood functions to be evaluated on a data set consisting of many observations. It includes classes designed to:
* Maximize univariate log likelihood functions by finding a root of the derivative within a fixed interval (using Brent's method)
  * Can optionally estimate the standard error of the MLE/MAP estimate using the Fisher information, if a function for evaluating the second derivative is provided
  * Relevant class: `brentSolver`
* Maximize univariate log likelihood functions (or any other function) within a fixed interval without derivative information, using golden section search
   * Relevant class: `golden_solver`
* Maximize multivariate log likelihood functions, given initial parameter guesses, using BFGS
   * Relevant class: `multivar_ml_solver`
   * Helper class to simplify solving mixture proportion problems (see below): `mixcomp_solver`
     
Some features:
* The ability to add data points as named vectors, which can be accessed by outside functions and looked up by name
* The ability to add some pre-set prior distributions to calculations
* The ability to constrain variables to $(0, \infty]$ by log transformation or $(0,1)$ by logit transformation automatically
  * Automatically handles these transformations and makes the gradient depend on the un-transformed versions of the variables
* The ability to model mixture proportions (see below)

## Variable transformations

#### Constraining to $[0,\infty)$

Constraining a variable to $[0, \infty)$ can be accomplished through log transformation (and handled automatically by class functions). In this case, any given variable $x_j$ will be stored as $x_j = log(g_j)$, where $g_j$ is the initial guess for $x_j$. Then, the input to the log likelihood function and its gradient are the back-transformed version of each $x_j$ constrained this way: $t(x_j) = e^{x_j}$. 

#### Constraining to $[0,1]$

Constraining a variable to $[0,1]$ can be accomplished through logit transformation (and handled automatically by class functions). In this case, any given variable $x_j$ will be stored as $x_j = log(\frac{g_j}{1-g_j})$, where $g_j$ is the initial guess for $x_j$. Then, the input to the log likelihood function and its gradient are the back-transformed version of each $x_j$ constrained this way: $t(x_j) = \frac{1}{e^{-x_j} + 1}$.

#### Constraining a set of variables to $[0,1]$ and ensuring that they sum to 1

This case is designed to model mixture proportions. See next section.

## Mixture proportions
There are some special cases where you have collected observations that are thought to result from a mixture (in unknown proportions) of components, each contributing a known expected value to the result. For example, suppose you have sequenced a pool of individuals thought to belong to three populations, and you want to know what percent of the pool is made up of individuals of each population. 

Your data consist of a series of $n$ allele frequencies measured at different alleles: $A = A_1, A_2, A_3 ... A_n$

You are interested in modeling your data as coming from a mixture of three populations $P_1$, $P_2$, and $P_3$

For any given allele, you know the expected allele frequency in population 1, 2, and 3: $f_{i1} = E[A_i | P_1], f_{i2} = E[A_i | P_2], f_{i3} = E[A_i | P_3]$

You want to solve for the mixture components $m_1$, $m_2$, and $m_3$, where each denotes the proportion of the pool made up of individuals from each population. These are subject to these constraints: 

$$
\begin{aligned}
0 < m_1 < 1 \\
0 < m_2 < 1 \\
0 < m_3 < 1 \\
\sum\limits_{j=1}^{3}(m_j) = 1
\end{aligned}
$$

To accomodate these constraints, each variable is logit transformed and divided by the sum of all mixture component variables. In other words, if the initial guess for a given mixture component variable $m_j$ is $g_j$, then each $m_j$ is initialized to $m_j = log(\frac{g_j'}{1 - g_j'})$, where $g_j' = \frac{g_j}{\sum\limits_{k=1}^3 g_k}.$ The input to the log likelihood function and its gradient are the back-transformed version of each $m_j$ rather than the $m_j$ values themselves: $$t(m_j) = \frac{\frac{1}{e^{-m_j} + 1}}{\sum\limits_{k=1}^n \frac{1}{e^{-m_k} + 1}}$$

`multivar_ml_solver` handles all this behind the scenes and exposes a single variable $p_i = \sum\limits_{j=1}^3 f_{ij}t(m_j)$ to the functions the user provided to evaluate the log likelihood and its gradient. This means that the user does not need to worry about the transformation, but only needs to provide the number of mixture components in the model and their expected contributions to the result, the $f_i$ vector, for every observation/data point $A_i$. In this example, the user would need to compare the value of $p_i$ at each function evaluation to the measured frequency of allele $i$ $A_i$. If the user has collected a reference allele count $r_i$ and alt allele count $a_i$ for each allele $i$, for example, this could be done by computing the binomial log likelihood of $a_i$ successes in $r_i + a_i$ draws with the parameter $p_i$.

`multivar_ml_solver` allows users to set up mixture component problems with an arbitrary number of other data values, and to incorporate these however is desired in the supplied log likelihood and gradient functions (although only one set of mixture components is currently allowed).

If a simpler interface is desired, and the user only needs to solve for a set of mixture components given some data (as in this example), the class `mixcomp_solver` is provided. This class has some pre-set ways of relating $p_i$ to data: via least squares, the normal distribution, the beta distribution, or the binomial distribution (as above).

Both classes also allow the initial guesses of mixture components to start as an even pool of all possible individuals, a user-supplied vector of initial guesses of mixture proportions, or to randomly shuffle mixture components. The user can also provide a Dirichlet prior on mixture components, and if provided, random shuffling will use the Dirichlet concentration parameter for each mixture component.

## Requirements
Only requires a C++11 compiler. Also depends on the [stlbfgs](https://github.com/nkschaefer/stlbfgs) library, which is included as a submodule. The original repository is [here](https://github.com/ultimaille/stlbfgs), and the forked version was modified to be compatible with older compilers. 
