module LinearRegression

using LinearAlgebra
import StatsAPI.coef
import StatsAPI.islinear

export linregress
export coef
export SolveQR, SolveCholesky

include("linreg.jl")

end
