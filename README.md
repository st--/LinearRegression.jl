# LinearRegression

[![Build Status](https://github.com/st--/LinearRegression.jl/actions/workflows/CI.yml/badge.svg?branch=)](https://github.com/st--/LinearRegression.jl/actions/workflows/CI.yml?query=branch%3A)


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

Linear regression based on vector and matrix inputs.

Weighted linear regression.

Allowing you to extract coefficients and predict.

*I'm happy to receive [issue reports](https://github.com/st--/LinearRegression.jl/issues/new/choose) and [pull requests](https://github.com/st--/LinearRegression.jl/compare), though I am likely to say no to proposals that would significantly increase the scope of this package (see below for other packages with more features).*

## What this package does not do:

Be as comprehensive as SciML's [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl/) (on the other hand, less dependencies).

Ridge regression (use [MultivariateStats.jl](https://github.com/JuliaStats/MultivariateStats.jl) instead, or convince me it really should be part of LinearRegression.jl as well).

Handling of DataFrames (use [GLM.jl](https://github.com/JuliaStats/GLM.jl) instead).

Lots of regression statistics (use [GLM.jl](https://github.com/JuliaStats/GLM.jl) instead).

Different (non-Gaussian) observation models (use [GLM.jl](https://github.com/JuliaStats/GLM.jl) instead).

Sparse regression (use [SparseRegression.jl](https://github.com/joshday/SparseRegression.jl/) instead).

Bayesian linear regression (use [BayesianLinearRegressors.jl](https://github.com/JuliaGaussianProcesses/BayesianLinearRegressors.jl) instead).

Online estimation (use [OnlineStats.jl](https://github.com/joshday/OnlineStats.jl) instead).

*Want to suggest another package to recommend here? Feel free to [open a pull request](https://github.com/st--/LinearRegression.jl/compare)! (:*
