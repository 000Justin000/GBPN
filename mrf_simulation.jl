using DelimitedFiles;
using StatsBase;
using Printf;
using Plots;

function stepheatbath!(grid::Array{Int,2};   # class labels
                       h::Array{Float64,3},  # self potential
                       H::Array{Float64,2},  # coupling potential
                       temp::Float64,        # temperature
                       verbose::Bool = true) # verbose flag

    # Randomly generate a new position within the grid and flip it with the right probability
    i = rand(1:size(grid,1))
    j = rand(1:size(grid,2))

    log_h = log.(h[i,j,:])
    log_H = log.(H)

    Eij = -log_h
    for (i_,j_) in [(-1,0), (1,0), (0,-1), (0,1)]
        if (1 <= i+i_ <= size(grid,1)) && (1 <= j+j_ <= size(grid,2))
            Eij -= log_H[grid[i+i_,j+j_],:]
        end
    end

    probs = exp.(-Eij ./ temp) / sum(exp.(-Eij ./ temp))

    grid[i,j] = sample(1:length(probs), Weights(probs))

    if verbose println("Changed $i,$j vertice spin to $(grid[i,j])") end
    return grid
end

# Several steps of the heat bath algorithm on a Ising's spin grid
function heatbath!(grid::Array{Int,2},     # class label
                   h::Array{Float64,3},    # self potential
                   H::Array{Float64,2};    # coupling potential
                   temp::Float64 = 1.0,    # temperature
                   iters::Integer = 50000, # number of iterations
                   plot::Bool = true,      # plot flag
                   verbose::Bool = true)   # verbose flag

    @assert iters > 100 "ArgumentError: \"iters\" must be higher than 100"
    for _ in 1:iters
        stepheatbath!(grid, h=h, H=H, temp=temp, verbose=false)
    end
    # next heatmap the final class labels
    if plot
        heatmap(grid);
    end
end

n, c = 51, 3;
l = range(-1.0, 1.0, length=n);
f = [[l[i],l[j]] for j in 1:n for i in 1:n];
s = zeros(n,n,c);
s[:,:,1] = (l.^2 .+ l'.^2) .- 0.65;
s[:,:,2] = 0.65 .- (l.^2 .+ l'.^2);
h = 1.0 ./ (1.0 .+ exp.(-s * 0.2));
H = [0.9 0.1 0.6; 
     0.1 0.9 0.6;
     0.6 0.6 0.2]

labels = []
board = zeros(n, n);
for _ in 1:10
    board = rand(1:c, n, n);
    heatbath!(board, h, H; iters=1000000, temp=1.4);
    push!(labels, board[:].-1);
end
writedlm("labels", hcat(labels...));
