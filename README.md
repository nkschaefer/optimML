# optimML
A fast and flexible C++ library for numeric optimization of complex log likelihood functions, including mixture components that must sum to 1

## Use case
This library was designed to make it easy to find maximum likelihood estimates (MLE) or maximum a posteriori estimates (MAP), given arbitrary complex log likelihood functions to be evaluated on a data set consisting of many observations. It includes classes designed to:
* Maximize univariate log likelihood functions, with or without using the derivative, within a fixed interval (using Brent's method)
  * Can optionally estimate the standard error of the MLE/ Laplace approximation standard error around the MAP estimate using the observed Fisher information, if a function for evaluating the second derivative is provided
  * Can also find a root of a function within a fixed interval
  * Relevant class: `brent_solver`
* Maximize multivariate log likelihood functions, with the ability to compute the first derivative and given initial parameter guesses, using BFGS
   * Relevant class: `multivar_ml_solver`
   * Helper class to simplify solving mixture proportion problems (see below): `mixcomp_solver`
 * Solve a multivariate system of equations by minimizing the sum of squared residuals, given one function per equation, and another function per equation to evaluate its gradient
   * Relevant class: `multivar_sys_solver`
      *  This class uses `multivar_ml_solver` as a backend, but simplifies things for users. Users provide a set of functions, their gradients, and the observed values corresponding to the result/right hand side of each function. Users can supply functions that take only the parameter vector as input, or can provide observed data values and a function that accesses them, in the same way as the other classes.
     
## Some features

#### General organization
* Users create solver objects and point them to the necessary user-provided functions. These functions must evaluate the log likelihood given a variable (univariate solvers) or a vector of variables (multivariate solver) and one or more pieces of observed data. Some solvers also require functions to evaluate the first and/or second derivative of the log likelihood function.
* After instantiating an object with the necessary functions, the user can add data points as named vectors.
   * In other words, if the user has collected $n$ pieces of data, where each is a vector of multiple values, the user creates an $n$-value vector for each type of observation.
   * Solvers can handle/store both integer and double observations, and each observation is mapped to a name the user provides.
   * If there are 10 rows of data, each with an integer measurement called `count1`, another called `count2`, and a decimal measurement called `intensity`, the user would add a `std::vector<int>(10)` named `count1`, a `std::vector<int>(10)` named `count2`, and a `std::vector<double>(10)` named "intensity." Then, in the functions they provided, each value can be accessed by name.
   * Functions are evaluated once at each data point, so the provided functions do not need to consider vectors of data.

#### Prior distributions
* Convenient methods are provided to add a prior distribution on one or more independent variables. This will result in maximum a posteriori (MAP) estimates being computed for each independent variable, rather than maximum likelihood estimates (MLEs).
   * To specify an arbitrary prior distribution, the user must provide a function to compute its log likelihood and its derivative (and its second derivative, if using Brent's method and the standard error of the estimate is desired)
   * If a second derivative is supplied, the standard errors given are for the [Laplace approximation](https://en.wikipedia.org/wiki/Laplace%27s_approximation) of the posterior around its mode, which becomes accurate toward the limit of infinite data.
* Some pre-set prior distributions are provided for which the user does not need to provide any functions (currently included: Normal, truncated Normal, Beta, and Binomial)

#### Weighted observations
* Observations/data points can have weights (provided as a vector of doubles). The solver then computes weighted maximum likelihood instead of standard maximum likelihood.

#### Variable constraints
* The solver can constrain variables to $(0, \infty]$ by log transformation or $(0,1)$ by logit transformation automatically (see below).
* The solver can treat a set of unknown variables as mixture proportions that must sum to 1 (see below).

#### Multithreading
* The solvers (currently besides multivar_sys_solver) can be parallelized, letting each row of data/observations be evaluated by a function call in its own thread. To use this functionality, call `solver.set_threads(x)` to use `x` threads. This is mostly useful when there are many observations/data points. In the case of many parameters, parallelization of the BFGS solver will be more useful.

## Variable transformations

#### Constraining to $(0,\infty]$

Constraining a variable to $(0, \infty]$ can be accomplished through log transformation (and handled automatically by class functions). In this case, any given variable $x_j$ will be stored as $x_j = log(g_j)$, where $g_j$ is the initial guess for $x_j$. Then, the input to the log likelihood function and its gradient are the back-transformed version of each $x_j$ constrained this way: $t(x_j) = e^{x_j}$. The user only needs to deal with the back-transformed values $t(x)$ in the provided log likelihood and gradient functions; transformation is handled behind the scenes.

#### Constraining to $(0,1)$

Constraining a variable to $(0,1)$ can be accomplished through logit transformation (and handled automatically by class functions). In this case, any given variable $x_j$ will be stored as $x_j = log(\frac{g_j}{1-g_j})$, where $g_j$ is the initial guess for $x_j$. Then, the input to the log likelihood function and its gradient are the back-transformed version of each $x_j$ constrained this way: $t(x_j) = \frac{1}{e^{-x_j} + 1}$. The user only needs to deal with the back-transformed values $t(x)$ in the provided log likelihood and gradient functions; transformation is handled behind the scenes.

#### Constraining a set of variables to $(0,1)$ and ensuring that they sum to 1

There are two cases of this handled by this library:

* You can add a parameter group to `multivar_ml_solver` with `add_param_grp(x)`, where `x` is a `std::vector<double>` containing initial guesses of the parameters (the solver will ensure that they sum to 1). The solver will remember that these parameters belong to a group and will handle variable transformations behind the scenes - you can operate on them as usual in function calls, but if you will need to keep track of which parameter in the parameter vector is which. In the end, they will sum to 1.
* "Mixture proportions" are another case of this situation, where observed data points are a combination of all mixture components, which each contribute a known expected value to the total. See next section.

## Mixture proportions

#### Problem definition
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

To accomodate these constraints, each variable is logit transformed and divided by the sum of all mixture component variables. In other words, if the initial guess for a given mixture component variable $m_j$ is $g_j$, then each $m_j$ is initialized to 

$$
\begin{aligned}
m_j = log\left(\frac{g_j'}{1 - g_j'}\right) \\
\end{aligned}
$$

where 

$$
\begin{aligned}
g_j' = \frac{g_j}{\sum\limits_{k=1}^3 g_k} \\
\end{aligned}
$$

The input to the log likelihood function and its gradient are the back-transformed version of each $m_j$ rather than the $m_j$ values themselves: 

$$
\begin{aligned}
t(m_j) = \frac{\frac{1}{e^{-m_j} + 1}}{\sum\limits_{k=1}^n \frac{1}{e^{-m_k} + 1}} \\
\end{aligned}
$$

`multivar_ml_solver` handles all this behind the scenes and exposes a single variable:

$$
\begin{aligned}
p_i = \sum\limits_{j=1}^3 f_{ij}t(m_j) \\
\end{aligned}
$$

to the functions the user provided to evaluate the log likelihood and its gradient. This means that the user does not need to worry about the transformation, but only needs to provide the number of mixture components in the model and their expected contributions to the result, the $f_i$ vector, for every observation/data point $A_i$. In this example, the user would need to compare the value of $p_i$ at each function evaluation to the measured frequency of allele $i$ $A_i$. If the user has collected a reference allele count $r_i$ and alt allele count $a_i$ for each allele $i$, for example, this could be done by computing the binomial log likelihood of $a_i$ successes in $r_i + a_i$ draws with the parameter $p_i$.

#### Implementation details
`multivar_ml_solver` allows users to set up mixture component problems with an arbitrary number of other data values, and to incorporate these however is desired in the supplied log likelihood and gradient functions (although only one set of mixture components is currently allowed).

If a simpler interface is desired, and the user only needs to solve for a set of mixture components given some data (as in this example), the class `mixcomp_solver` is provided. This class has some pre-set ways of relating $p_i$ to data: via least squares, the normal distribution, the beta distribution, or the binomial distribution (as above).

#### Use of a prior
The only tenable prior distribution for mixture components is a Dirichlet distribution, which provides a concentration parameter for each mixture component (similar in concept to the $\alpha$ and $\beta$ parameters of the Beta distribution) Draws from this Dirichlet distribution are then vectors of mixture proportions. `multivar_ml_solver` and its wrapper class `mixcomp_solver` allow users to specify a Dirichlet prior on mixture components. 

#### Initial guesses
These are difficult situations to model, and the maximum likelihood estimate provided is likely to be influenced by the initial guess of mixture components. By default, mixture components are intialized to $1/n$, where $n$ is the number of components in the model. Users can also provide a vector of initial guesses, if they have pre-existing information about the probable values of the mixture components.

To better explore the parameter space, `multivar_ml_solver` and `mixcomp_solver` also provide methods to randomly shuffle mixture components. If a Dirichlet prior has been specified, mixture components are randomly sampled from the prior distribution. One recommended strategy for getting a reasonable idea of the global MLE of mixture parameters would be to start at an even pool of mixture proportions and solve, then store the result and its log likelihood. Then, for some number of random trials $n$, randomly shuffle the mixture components and try again, again storing the result and its log likelihood. When this is done, choose the set of parameters with the highest log likelihood across all random trials.

## Requirements
The only requirement is a compiler that can support the C++11 standard. This corresponds to gcc >= 4.8.1, or to clang >= 3.3. 

The heavy lifting for solving multivariate functions depends on the [stlbfgs](https://github.com/nkschaefer/stlbfgs) library, which is included as a submodule. The original repository is [here](https://github.com/ultimaille/stlbfgs), and the forked version was modified to be compatible with older compilers. 
