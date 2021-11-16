abstract type AbstractLinregSolver end

"""
    SolveQR()

Pass as `method` to [`linregress`](@ref) to use QR factorization to solve the
linear regression system. Slower but more accurate for ill-conditioned systems
than [`SolveCholesky`](@ref).
"""
struct SolveQR <: AbstractLinregSolver end

"""
    SolveCholesky()

Pass as `method` to [`linregress`](@ref) to use Cholesky factorization to solve
the linear regression system. Faster but less accurate for ill-conditioned
systems than [`SolveQR`](@ref).
"""
struct SolveCholesky <: AbstractLinregSolver end

struct LinearRegressor{T<:Number}
    intercept::Bool
    coeffs::Vector{T}
end

"""
    coef(lr::LinearRegressor)

Returns the linear regression coefficients of a [`linregress`](@ref) result. If
`lr.intercept` is `true`, this includes the bias (intercept) in the last
position.
"""
coef(lr::LinearRegressor) = lr.coeffs

function (lr::LinearRegressor)(X::AbstractMatrix)
    res = X * lr.coeffs[1:size(X, 2)]
    if lr.intercept
        res .+= lr.coeffs[end]
    end
    return res
end

function (lr::LinearRegressor)(X::AbstractVector)
    res = dot(X, lr.coeffs[1:length(X)])
    if lr.intercept
        res += lr.coeffs[end]
    end
    return res
end

@doc raw"""
    linregress(X, y; intercept=true, method=SolveQR())
    linregress(X, y, weights; intercept=true, method=SolveQR())

Do (possibly weighted) linear regression to obtain coefficients β such that `ŷ
= X * β` minimizes `‖ŷ - y‖²`. In the default case, this corresponds to solving
`X \ y`.

Returns a `LinearRegressor`, which can be passed to [`coef`](@ref) to extract
the coefficients β or called with a vector or matrix `X` to predict at a single
point or set of points.

Currently assumes that, if `X` is a Matrix, that `size(X) == (length(y),
num_coefs)` (i.e., each row describes the features for one observation).

## Keyword Arguments:

- If `intercept` is `true`, implicitly adds a column of ones to `X` to model a
  bias term.

- `method` determines how to solve the linear regression system (see
  [`SolveQR`](@ref) and [`SolveCholesky`](@ref)).
"""
function linregress(X, y, weights=nothing; intercept=true, method=SolveQR())
    size(X, 1) == length(y) || throw(DimensionMismatch("size of X and y does not match"))
    if intercept
        X = _append_bias_column(method, X)
    else
        X = _maybe_convert(method, X)
    end
    coeffs = if weights === nothing
        _lin_solve(method, X, y)
    else
        length(weights) == length(y) || throw(DimensionMismatch("size of y and weights does not match"))
        W = Diagonal(weights)
        _lin_solve(method, X, y, W)
    end
    return LinearRegressor(intercept, coeffs)
end

function _lin_solve(solver::AbstractLinregSolver, X, y, W)
    # √W X \ √W y
    Wsqrt = sqrt(W)
    return _lin_solve(solver, Wsqrt * X, Wsqrt * y)
end

function _lin_solve(::SolveQR, X, y)
    return X \ y
end

function _lin_solve(::SolveCholesky, X, y, W)
    Wy = W * y
    XtWy = lmul!(X', Wy)
    WX = W * X
    XtWX = lmul!(X', WX)
    # X' W X \ (X' W y)
    return ldiv!(cholesky!(Hermitian(XtWX)), XtWy)
end

function _lin_solve(::SolveCholesky, X, y)
    # X'X \ X'y
    return ldiv!(cholesky!(Hermitian(X'X)), X'y)
end

function _lin_solve(::SolveCholesky, X, y)
    return Hermitian(X'X) \ (X'*y)
end

function _append_bias_column(::AbstractLinregSolver, X)
    ones_column = ones(eltype(X), size(X, 1))
    return [X ones_column]
end

function _append_bias_column(::SolveCholesky, X)
    ones_column = ones(LinearAlgebra.choltype(X), size(X, 1))
    return [X ones_column]
end

_maybe_convert(::AbstractLinregSolver, X) = X
_maybe_convert(::SolveCholesky, X) = convert(Matrix{LinearAlgebra.choltype(X)}, X)
