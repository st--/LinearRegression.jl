abstract type AbstractLinregSolver end

struct SolveQR <: AbstractLinregSolver end
struct SolveCholesky <: AbstractLinregSolver end

struct LinearRegressor{T<:Number}
    intercept::Bool
    coeffs::Vector{T}
end

function (lr::LinearRegressor)(X::AbstractMatrix)
    if lr.intercept
        X = add_bias_column(X)
    end
    return X * lr.coeffs
end

function (lr::LinearRegressor)(X::AbstractVector)
    res = dot(X, lr.coeffs[1:length(X)])
    if lr.intercept
        res += lr.coeffs[end]
    end
    return res
end

"""
    linreg(X, y; intercept=true, method=SolveQR())

Currently assumes that, if `X` is a Matrix, that `size(X) == (length(y),
num_coefs)` (i.e., each row describes the features for one observation).
"""
function linregress(X, y, weights=nothing; intercept=true, method=SolveQR())
    size(X, 1) == length(y) || throw(DimensionMismatch("size of X and y does not match"))
    if intercept
        X = add_bias_column(X)
    end
    coeffs = if weights === nothing
        _lin_solve(method, X, y)
    else
        W = Diagonal(weights)
        _lin_solve(method, X, y, W)
    end
    return LinearRegressor(intercept, coeffs)
end

coef(lr::LinearRegressor) = lr.coeffs

function _lin_solve(::SolveQR, X, y)
    return X \ y
end

function _lin_solve(solver::AbstractLinregSolver, X, y, W)
    Wsqrt = sqrt(W)
    return _lin_solve(solver, Wsqrt * X, Wsqrt * y)
end

function _lin_solve(::SolveCholesky, X, y)
    return Hermitian(X'X) \ (X'*y)
end

function _lin_solve(::SolveCholesky, X, y, W)
    return Hermitian(X'*W*X) \ X'*(W*y)
end

function add_bias_column(X)
    return [X ones(size(X, 1))]
end
