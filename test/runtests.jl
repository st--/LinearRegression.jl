using LinearRegression
using Test

@testset "LinearRegression.jl" begin
    @testset "scalar" begin
        X = [4 6]'
        y = [-2, 4]

        @testset "weights $(repr(weights))" for weights in (nothing, ones(2))
            @testset "Solver $method" for method in (SolveQR(), SolveCholesky())
                regressor = linregress(X, y, weights; intercept=true, method=method)
                @test regressor([5]) ≈ 1
                @test regressor([4 5 6]') ≈ [-2, 1, 4]
                @test coef(regressor) ≈ [3, -14]

                regressor = linregress(X, y, weights; intercept=false, method=method)
                @test regressor([0]) == 0
            end
        end
    end

    @testset "matrix" begin
        @testset "intercept $intercept" for intercept in (true, false)
            X = rand(100, 13)
            X0 = intercept ? [X ones(size(X, 1))] : X
            beta = rand(size(X0, 2))
            y = X0 * beta

            @testset "weights $(repr(weights))" for weights in (nothing, ones(size(X, 1)))
                @testset "Solver $method" for method in (SolveQR(), SolveCholesky())
                    regressor = linregress(X, y, weights; intercept=intercept, method=method)
                    @test coef(regressor) ≈ beta
                    @test regressor(X) ≈ y
                    @test regressor(X[17, :]) ≈ y[17]
                end
            end
        end
    end
end
