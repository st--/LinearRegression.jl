# LinearRegression

[![Build Status](https://github.com/st--/LinearRegression.jl/actions/workflows/CI.yml/badge.svg?branch=)](https://github.com/st--/LinearRegression.jl/actions/workflows/CI.yml?query=branch%3A)

This is a simple package. [See below](#what-this-package-does-not-do-aka-alternatives) for a list of more complex packages for linear regression in Julia.

## Why this package?

Because I keep finding myself thinking, "I need some simple linear regression
here...", and missing the level of abstraction halfway in between `X \ y` and
[GLM.jl](https://github.com/JuliaStats/GLM.jl), without lots of additional
dependencies.
I keep running into [this Discourse
thread](https://discourse.julialang.org/t/efficient-way-of-doing-linear-regression/31232)
and wishing I could just be `using LinearRegression`.
[Alistair.jl](https://github.com/giob1994/Alistair.jl) would fit the bill, but
hasn't been maintained and doesn't work with Julia 1+.

## What this package supports:

**Linear regression based on vector and matrix inputs and outputs:**
```julia
lr = linregress(X, y)
```
`X` can be a *vector* (1D inputs, each element is one observation) or a *matrix* (multivariate inputs, each row is one observation, columns represent features).

`y` can be a *vector* (1D outputs, each element is one observation) or a *matrix* (multivariate outputs, each row is one observation, columns represent targets).

**Weighted linear regression:**
```julia
lr = linregress(X, y, weights)
```
`weights` is the vector of each observation's weight.

**Intercept/bias term:**
By default, implicitly adds a column of ones to account for the intercept term.

You can disable this and force the linear regression to go through the origin by passing the `intercept=false` keyword argument.

**Choice of solver:**
By default, uses QR factorization (`X \ y`) to solve the linear system.
You can explicitly choose a solver by passing the `method` keyword argument.
Currently implemented choices are `method=SolveQR()` (using QR factorization, the default) and `method=SolveCholesky()` (using Cholesky factorization; can be faster, but numerically less accurate).

**Predicting:**
```julia
ytest = lr(Xtest)
```

**Extracting coefficients:**
```julia
β = coef(lr)
```
which includes the intercept/bias in the last position, if `intercept=true` (the default).

You can explicitly obtain slopes and intercept/bias by calling
```julia
LinearRegression.slope(lr)
LinearRegression.bias(lr)
```

*I'm happy to receive [issue reports](https://github.com/st--/LinearRegression.jl/issues/new/choose) and [pull requests](https://github.com/st--/LinearRegression.jl/compare), though I am likely to say no to proposals that would significantly increase the scope of this package (see below for other packages with more features).*

## What this package does not do (aka Alternatives):

* Be as comprehensive as SciML's [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl/) (on the other hand, less dependencies).

* Ridge regression (use [MultivariateStats.jl](https://github.com/JuliaStats/MultivariateStats.jl) instead, or convince me it really should be part of LinearRegression.jl as well).

* Handling of DataFrames (use [GLM.jl](https://github.com/JuliaStats/GLM.jl) instead).

* Lots of regression statistics (use [GLM.jl](https://github.com/JuliaStats/GLM.jl) instead).

* Different (non-Gaussian) observation models (use [GLM.jl](https://github.com/JuliaStats/GLM.jl) instead).

* Sparse regression (use [SparseRegression.jl](https://github.com/joshday/SparseRegression.jl/) instead).

* Bayesian linear regression (use [BayesianLinearRegressors.jl](https://github.com/JuliaGaussianProcesses/BayesianLinearRegressors.jl) instead).

* Online estimation (use [OnlineStats.jl](https://github.com/joshday/OnlineStats.jl) instead).

*Want to suggest another package to recommend here? Feel free to [open a pull request](https://github.com/st--/LinearRegression.jl/compare)! (:*
