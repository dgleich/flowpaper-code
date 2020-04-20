## run "../flow_embedding.py" first to generate data
using Plots
using SparseArrays
using LinearAlgebra
using TextParse
using DelimitedFiles
using PyCall
using ColorSchemes

function read_graph(filename)
    src, dst, val = csvread(filename;
        spacedelim=true, colparsers=[Int,Int,Float64], header_exists=false)[1]
    src .+= 1
    dst .+= 1
    n = maximum(src)
    n = max(maximum(dst),n)
    A = sparse(src, dst, val, n, n)
    A = max.(A,A')
    return findnz(triu(A,1))[1:2]
end

src, dst = read_graph("../../dataset/lawlor-spectra-k32.edgelist.gz")
# x, y = csvread("$../../dataset/lawlor-spectra-k32.edgelist";
#         spacedelim=true, colparsers=[Float64,Float64],
#         header_exists=false)[1]
V = readdlm("../../dataset/lawlor-spectra-k32.coords")
V[:,1] *= -10
V[:,2] *= -4


function plotperm(v)
  return invperm(sortperm(v))
end
# function myscatter(x,y,color;kwargs...)
#   p = sortperm(color)
#   scatter(x[p],y[p], marker_z = color[p],
#     label="", alpha=0.1, markerstrokewidth=0,
#     colorbar=false, framestyle=:none; kwargs...)
# end

##
pfile = "embeddings/spectral_3_bfs.p.gz"

pickle = pyimport("pickle")
gzip = pyimport("gzip")
f = gzip.open(pfile)
data= pickle.load(f)
f.close()
function _pycsr_to_sparse(rowptr, colinds, vals, m, n)
  rowinds = zeros(eltype(colinds),0)
  for i in 1:length(rowptr)-1
    for j in rowptr[i]:rowptr[i+1]-1
      push!(rowinds, i)
    end
  end
  return sparse(rowinds, colinds, vals, m, n)
end
function pycsr_to_sparse(pycsr)
  rowptr = pycsr.indptr
  colinds = pycsr.indices
  vals = pycsr.data
  colinds .+= 1
  m, n = data.shape
  # put in a fucntion so we can do type inference
  return _pycsr_to_sparse(rowptr, colinds, vals, m, n )
end

X = pycsr_to_sparse(data)

flipexp = findall(X[517173,:] .> 0) # this is at the opposite end
for f in flipexp
  X[:,f] = 1.0 .-X[:,f]
end

X = Matrix(X) # convert to Matrix
X ./= 10 # convert to 0, 1
subset = vec((sum(X,dims=2) .> 0))
subsetnodes = findall(subset)
subset_c = vec((sum(X,dims=2) .<= 0))
subsetnodes_c = findall(subset_c)
U,sig,Vt = svd(X.-1)

new_coords = zeros(size(X,1),2)
new_coords[subsetnodes,1] = plotperm(-1 * U[subsetnodes,1])
new_coords[subsetnodes,2] = plotperm(-1 * U[subsetnodes,2])
i = argmax(new_coords[subsetnodes,1])
new_coords[subsetnodes_c,1] .= new_coords[subsetnodes[i],1] + 10000
new_coords[subsetnodes_c,2] .= new_coords[subsetnodes[i],2] + 10000


using Clustering

Xs = transpose(new_coords[subsetnodes,:])

R = kmeans(Xs,100; maxiter=200, display=:iter)

clusters = Dict(i=>[] for i = 1:100)

for i = 1:length(subsetnodes)
    push!(clusters[R.assignments[i]],subsetnodes[i]-1)
end

pushfirst!(PyVector(pyimport("sys")."path"), "/homes/liu1740/Research/LocalGraphClustering/")
lgc = pyimport("localgraphclustering")

G = lgc.GraphLocal("../../dataset/lawlor-spectra-k32.edgelist","edgelist")
conds = []
for i = 1:100
    push!(conds,G.compute_conductance(clusters[i]))
end
