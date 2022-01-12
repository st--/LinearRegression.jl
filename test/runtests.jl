using LinearRegression
using Test

function test_linreg_multivariate(X, y, beta; index=1)
    @testset "weights $(repr(weights))" for weights in (nothing, ones(size(X, 1)))
        @testset "Solver $method" for method in (SolveQR(), SolveCholesky())
            regressor = linregress(X, y, weights; intercept=intercept, method=method)
            @test coef(regressor) ≈ beta
            @test regressor(X) ≈ y
            @test regressor(X[index, :]) ≈ y[index]

            @testset "type-inference" begin
                @inferred coef(regressor)
                @inferred regressor(X)
                @inferred regressor(X[index, :])
            end
        end
    end
end

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

                @testset "type-inference" begin
                    @inferred coef(regressor)
                    @inferred regressor([4 5 6]')
                    @inferred regressor([0])
                end
            end
        end
    end

    @testset "intercept $intercept" for intercept in (true, false)
        X = rand(100, 13)
        X0 = intercept ? [X ones(size(X, 1))] : X

        @testset "matrix" begin
            beta = rand(size(X0, 2))
            y = X0 * beta

            test_linreg(X, y, beta; index=17)
        end

        @testset "multioutput" begin
            beta = rand(size(X0, 2), 5)
            y = X0 * beta

            test_linreg(X, y, beta; index=17)
        end
    end
end
