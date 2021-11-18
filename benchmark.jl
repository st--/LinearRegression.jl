using LinearRegression
using BenchmarkTools

results = Dict()

for N in (500, 5000, 50000)
    for D in (10, 100, 1000)
        X = rand(N, D)
        y = rand(N)
        weights = 0.9 .+ 0.2rand(N)

        for solver in (SolveQR(), SolveCholesky())
            tu = @benchmark linregress($X, $y; method=$solver)
            tw = @benchmark linregress($X, $y, $weights; method=$solver)
            results[N, D, solver] = (tu, tw)
        end
    end
end
