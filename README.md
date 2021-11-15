# LinearRegression

[![Build Status](https://github.com/st--/LinearRegression.jl/actions/workflows/CI.yml/badge.svg?branch=)](https://github.com/st--/LinearRegression.jl/actions/workflows/CI.yml?query=branch%3A)


## Why this package?

Because I keep finding myself thinking, "I need some simple linear regression
here...", and missing the level of abstraction halfway in between `X \ y` and
[GLM.jl](https://github.com/JuliaStats/GLM.jl).
I keep running into [this Discourse
thread](https://discourse.julialang.org/t/efficient-way-of-doing-linear-regression/31232)
and wishing I could just be `using LinearRegression`.
[Alistair.jl](https://github.com/giob1994/Alistair.jl) would fit the bill, but
hasn't been maintained and doesn't work with Julia 1+.

## What this package does

Linear regression based on vector and matrix inputs.

Weighted linear regression.

Allowing you to extract coefficients and predict.

## What this package does not do

Handling of DataFrames (use [GLM.jl](https://github.com/JuliaStats/GLM.jl) instead).

Different observation models (use [GLM.jl](https://github.com/JuliaStats/GLM.jl) instead).

Bayesian linear regression (use [BayesianLinearRegressors.jl](https://github.com/JuliaGaussianProcesses/BayesianLinearRegressors.jl) instead).

Online estimation (use [OnlineStats.jl](https://github.com/joshday/OnlineStats.jl) instead).
