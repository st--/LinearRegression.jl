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

struct LinearRegressor{T<:Union{Matrix{<:Number},Vector{<:Number}}}
    intercept::Bool
    coeffs::T
end

"""
    coef(lr::LinearRegressor)

Returns the linear regression coefficients of a [`linregress`](@ref) result. If
`lr.intercept` is `true`, this includes the bias (intercept) in the last
position.
"""
coef(lr::LinearRegressor) = lr.coeffs

slope(lr::LinearRegressor) = lr.intercept ? _slope(lr.coeffs) : lr.coeffs
bias(lr::LinearRegressor) = lr.intercept ? _bias(lr.coeffs) : _zero_bias(lr.coeffs)

_slope(coef::AbstractMatrix) = @view coef[1:end-1, :]
_slope(coef::AbstractVector) = @view coef[1:end-1]
_bias(coef::AbstractMatrix) = @view coef[end:end, :]
_bias(coef::AbstractVector) = coef[end]
_zero_bias(coef::AbstractMatrix) = zero(eltype(lr.coeffs), 1, size(lr.coeffs, 2))
_zero_bias(coef::AbstractVector) = zero(eltype(lr.coeffs))

function (lr::LinearRegressor)(X::AbstractMatrix)
    if lr.intercept
        return X * _slope(lr.coeffs) .+ _bias(lr.coeffs)
    else
        return X * lr.coeffs
    end
end

function (lr::LinearRegressor)(x::AbstractVector)
    if lr.intercept
        res = x'_slope(lr.coeffs) + _bias(lr.coeffs)
    else
        res = x'lr.coeffs
    end
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
    size(X, 1) == size(y, 1) || throw(DimensionMismatch("size of X and y does not match"))
    if intercept
        X = add_bias_column(X)
    end
    coeffs = if weights === nothing
        _lin_solve(method, X, y)
    else
        length(weights) == size(y, 1) || throw(DimensionMismatch("size of y and weights does not match"))
        W = Diagonal(weights)
        _lin_solve(method, X, y, W)
    end
    return LinearRegressor(intercept, coeffs)
end

function _lin_solve(solver::AbstractLinregSolver, X, y, W)
    Wsqrt = sqrt(W)
    return _lin_solve(solver, Wsqrt * X, Wsqrt * y)
end

function _lin_solve(::SolveQR, X, y)
    return X \ y
end

function _lin_solve(::SolveCholesky, X, y, W)
    return Hermitian(X'*W*X) \ X'*(W*y)
end

function _lin_solve(::SolveCholesky, X, y)
    return Hermitian(X'X) \ (X'*y)
end

function add_bias_column(X)
    return [X ones(size(X, 1))]
end
