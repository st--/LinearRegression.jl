using LinearRegression
using Test

function check_type_stability(regressor, X)
    @testset "type-inference" begin
        @inferred coef(regressor)
        @inferred regressor(X)
        @inferred regressor(X[1, :])
    end
end

function test_linreg_multivariate(X, y, beta; index=1)
    @testset "weights $(repr(weights))" for weights in (nothing, ones(size(X, 1)))
        @testset "Solver $method" for method in (SolveQR(), SolveCholesky())
            regressor = linregress(X, y, weights; intercept=intercept, method=method)
            @test coef(regressor) ≈ beta
            @test regressor(X) ≈ y
            @test regressor(X[index, :]) ≈ y[index]
            check_type_stability(regressor, X)
        end
    end
end

@testset "LinearRegression.jl" begin
    @testset "univariate" begin
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
                check_type_stability(regressor, [4 5 6]')
            end
        end
    end

    @testset "multivariate" begin
        @testset "intercept $intercept" for intercept in (true, false)
            X = rand(100, 13)
            X0 = intercept ? [X ones(size(X, 1))] : X

            @testset "single-output" begin
                beta = rand(size(X0, 2))
                y = X0 * beta

                test_linreg(X, y, beta; index=17)
            end

            @testset "multi-output" begin
                beta = rand(size(X0, 2), 5)
                y = X0 * beta

                test_linreg(X, y, beta; index=17)
            end
        end
    end
end
