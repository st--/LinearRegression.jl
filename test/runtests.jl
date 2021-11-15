using LinearRegression
using Test

@testset "LinearRegression.jl" begin
    @testset "scalar" begin
        X = [4 6]'
        y = [-2, 4]
        @testset "Solver $method" for method in (SolveQR(), SolveCholesky())
            regressor = linregress(X, y; intercept=true, method)
            @test regressor([5]) ≈ 1
            @test regressor([4 5 6]') ≈ [-2, 1, 4]
            @test coef(regressor) ≈ [3, -14]

            regressor = linregress(X, y; intercept=false, method)
            @test regressor([0]) == 0
        end
    end

    @testset "matrix" begin
        @testset "intercept $intercept" for intercept in (true, false)
            X = rand(100, 13)
            X0 = intercept ? [X ones(size(X, 1))] : X
            beta = rand(size(X0, 2))
            y = X0 * beta
            @testset "Solver $method" for method in (SolveQR(), SolveCholesky())
                regressor = linregress(X, y; intercept, method)
                @test coef(regressor) ≈ beta
                @test regressor(X) ≈ y
                @test regressor(X[17, :]) ≈ y[17]
            end
        end
    end
end
