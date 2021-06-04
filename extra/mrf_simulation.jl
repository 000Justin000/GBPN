using DelimitedFiles;
using StatsBase;
using Printf;
using Plots;

function stepheatbath!(board::Array{Int,2};   # class labels
                       h::Array{Float64,3},  # self potential
                       H::Array{Float64,2},  # coupling potential
                       temp::Float64,        # temperature
                       verbose::Bool = true) # verbose flag

    # Randomly generate a new position within the board and flip it with the right probability
    i = rand(1:size(board,1))
    j = rand(1:size(board,2))

    log_h = log.(h[i,j,:])
    log_H = log.(H)

    Eij = -log_h
    for (i_,j_) in [(-1,0), (1,0), (0,-1), (0,1)]
        if (1 <= i+i_ <= size(board,1)) && (1 <= j+j_ <= size(board,2))
            Eij -= log_H[board[i+i_,j+j_],:]
        end
    end

    probs = exp.(-Eij ./ temp) / sum(exp.(-Eij ./ temp))

    board[i,j] = sample(1:length(probs), Weights(probs))

    if verbose println("Changed $i,$j vertice spin to $(board[i,j])") end
    return board
end

# Several steps of the heat bath algorithm on a Ising's spin board
function heatbath!(board::Array{Int,2},     # class label
                   h::Array{Float64,3},    # self potential
                   H::Array{Float64,2};    # coupling potential
                   temp::Float64 = 1.0,    # temperature
                   iters::Integer = 50000, # number of iterations
                   verbose::Bool = true)   # verbose flag

    @assert iters > 100 "ArgumentError: \"iters\" must be higher than 100"
    for _ in 1:iters
        stepheatbath!(board, h=h, H=H, temp=temp, verbose=false)
    end
end

function visualize(board, i=nothing)
    n = size(board, 1);
    l = range(-1.0, 1.0, length=n);
    x = [l[i] for j in 1:n for i in 1:n];
    y = [l[j] for j in 1:n for i in 1:n];
    c = board[:];
    h = plot(size=(500,500), framestyle=:none);
    scatter!(h, x[c .== 1], y[c .== 1], color=:black, label="");
    scatter!(h, x[c .== 2], y[c .== 2], color=:white, label="");
    scatter!(h, x[c .== 3], y[c .== 3], color=:red,   label="");
    fname = ((i == nothing) ? "board.svg" : ("board"*string(i)*".svg"));
    savefig(h, fname);
    display(h);
end

n, c = 51, 3;
l = range(-1.0, 1.0, length=n);
f = [[l[i],l[j]] for j in 1:n for i in 1:n];

# H = [0.9 0.1 0.6; 
#      0.1 0.9 0.6;
#      0.6 0.6 0.2]
# hconst = 0.65
# hscale = 0.20
# temp = 1.40

H = [0.3 0.1 0.6; 
     0.1 0.9 0.1;
     0.6 0.1 0.3]
hconst = 0.00
hscale = 0.60
temp = 1.50

s = zeros(n,n,c);
s[:,:,1] = (l.^2 .+ l'.^2) .- hconst;
s[:,:,2] = hconst .- (l.^2 .+ l'.^2);
h = 1.0 ./ (1.0 .+ exp.(-s * hscale));

labels = []
board = rand(1:c, n, n);
visualize(board);
for i in 0:9
    board[:,:] = rand(1:c, n, n);
    heatbath!(board, h, H; iters=1000000, temp=temp);
    visualize(board, i);
    push!(labels, board[:].-1);
end
writedlm("labels", hcat(labels...));
