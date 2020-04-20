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

pushfirst!(PyVector(pyimport("sys")."path"), "/homes/liu1740/Research/LocalGraphClustering/")
lgc = pyimport("localgraphclustering")

G = lgc.GraphLocal("../../dataset/lawlor-spectra-k32.edgelist","edgelist")

using Clustering

# function myscatter(x,y,color;kwargs...)
#   p = sortperm(color)
#   scatter(x[p],y[p], marker_z = color[p],
#     label="", alpha=0.1, markerstrokewidth=0,
#     colorbar=false, framestyle=:none; kwargs...)
# end

##

ncenters = 50

pfile = "embeddings/flow_3_bfs_delta_0.1.p.gz"

pickle = pyimport("pickle")
gzip = pyimport("gzip")
f = gzip.open(pfile)
data= pickle.load(f)
f.close()

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

Xs = transpose(new_coords)

R = kmeans(Xs,ncenters; maxiter=200, display=:iter)

clusters = Dict(i=>[] for i = 1:ncenters)

for i = 1:size(new_coords,1)
    push!(clusters[R.assignments[i]],i-1)
end

conds_flow = []
for i = 1:ncenters
    push!(conds_flow,G.compute_conductance(clusters[i]))
end

pfile = "embeddings/spectral_3_bfs.p.gz"

pickle = pyimport("pickle")
gzip = pyimport("gzip")
f = gzip.open(pfile)
data= pickle.load(f)
f.close()

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

Xs = transpose(new_coords)

R = kmeans(Xs,ncenters; maxiter=200, display=:iter)

clusters = Dict(i=>[] for i = 1:ncenters)

for i = 1:size(new_coords,1)
    push!(clusters[R.assignments[i]],i-1)
end

conds_spectral = []
for i = 1:ncenters
    push!(conds_spectral,G.compute_conductance(clusters[i]))
end

using Distributions

mpl = pyimport("matplotlib")

pyplot()
fig = mpl.pyplot.subplots(1,1,figsize=(6,4))
sns = pyimport("seaborn")
sns.distplot(conds_flow,hist=true,kde=true,ax=fig[2],
    kde_kws=Dict("kernel"=>"triw","clip"=>(0,0.99),"bw"=>0.1,"linewidth"=>3),
    hist_kws=Dict("alpha"=>0.2))
sns.distplot(conds_spectral,hist=true,kde=true,ax=fig[2],
    kde_kws=Dict("kernel"=>"triw","clip"=>(0,1),"bw"=>0.08,"linewidth"=>3),
    hist_kws=Dict("alpha"=>0.2))

fig[2].set_xlim(0.3,1)
fig[2].set_ylim(0,8)
fig[2].set_yticks([0,2,4,6])
fig[2].spines["top"].set_visible(false)
fig[2].spines["right"].set_visible(false)
fig[2].spines["left"].set_visible(false)
fig[2].spines["bottom"].set_visible(false)
fig[2].tick_params(axis="y", which="both", length=5,width=2)
fig[2].tick_params(axis="x", which="both", length=5,width=2)
for tick in fig[2].xaxis.get_major_ticks()
    tick.label.set_fontsize(15)
end
for tick in fig[2].yaxis.get_major_ticks()
    tick.label.set_fontsize(15)
end
fig[2].set_xlabel("ϕ(kmeans clusters)",fontsize=15)
fig[2].set_ylabel("Probability density",fontsize=15)
fig[1].legend(fig[2].get_lines(),["Local Flow Embedding","Local Spectral Embedding"],fancybox=true, shadow=true,
    fontsize=15,bbox_to_anchor=(0.73,0.85))
fig[1].savefig("cond_dist_50.pdf",format="pdf",bbox_inches="tight")
fig[1]

#################################################################################

ncenters = 100

pfile = "embeddings/flow_3_bfs_delta_0.1.p.gz"

pickle = pyimport("pickle")
gzip = pyimport("gzip")
f = gzip.open(pfile)
data= pickle.load(f)
f.close()

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

Xs = transpose(new_coords)

R = kmeans(Xs,ncenters; maxiter=200, display=:iter)

clusters = Dict(i=>[] for i = 1:ncenters)

for i = 1:size(new_coords,1)
    push!(clusters[R.assignments[i]],i-1)
end

conds_flow = []
for i = 1:ncenters
    push!(conds_flow,G.compute_conductance(clusters[i]))
end

pfile = "embeddings/spectral_3_bfs.p.gz"

pickle = pyimport("pickle")
gzip = pyimport("gzip")
f = gzip.open(pfile)
data= pickle.load(f)
f.close()

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

Xs = transpose(new_coords)

R = kmeans(Xs,ncenters; maxiter=200, display=:iter)

clusters = Dict(i=>[] for i = 1:ncenters)

for i = 1:size(new_coords,1)
    push!(clusters[R.assignments[i]],i-1)
end

conds_spectral = []
for i = 1:ncenters
    push!(conds_spectral,G.compute_conductance(clusters[i]))
end

using Distributions

mpl = pyimport("matplotlib")

pyplot()
fig = mpl.pyplot.subplots(1,1,figsize=(6,4))
sns = pyimport("seaborn")
sns.distplot(conds_flow,hist=true,kde=true,ax=fig[2],
    kde_kws=Dict("kernel"=>"triw","clip"=>(0,0.99),"bw"=>0.06,"linewidth"=>3),
    hist_kws=Dict("alpha"=>0.2))
sns.distplot(conds_spectral,hist=true,kde=true,ax=fig[2],
    kde_kws=Dict("kernel"=>"triw","clip"=>(0,1),"bw"=>0.06,"linewidth"=>3),
    hist_kws=Dict("alpha"=>0.2))

fig[2].set_xlim(0.3,1)
fig[2].set_ylim(0,10)
fig[2].set_yticks([0,2,4,6,8])
fig[2].spines["top"].set_visible(false)
fig[2].spines["right"].set_visible(false)
fig[2].spines["left"].set_visible(false)
fig[2].spines["bottom"].set_visible(false)
fig[2].tick_params(axis="y", which="both", length=5,width=2)
fig[2].tick_params(axis="x", which="both", length=5,width=2)
for tick in fig[2].xaxis.get_major_ticks()
    tick.label.set_fontsize(15)
end
for tick in fig[2].yaxis.get_major_ticks()
    tick.label.set_fontsize(15)
end
fig[2].set_xlabel("ϕ(kmeans clusters)",fontsize=15)
fig[2].set_ylabel("Probability density",fontsize=15)
fig[1].legend(fig[2].get_lines(),["Local Flow Embedding","Local Spectral Embedding"],fancybox=true, shadow=true,
    fontsize=15,bbox_to_anchor=(0.73,0.85))
fig[1].savefig("cond_dist_100.pdf",format="pdf",bbox_inches="tight")
fig[1]


###############################################################################


ncenters = 200

pfile = "embeddings/flow_3_bfs_delta_0.1.p.gz"

pickle = pyimport("pickle")
gzip = pyimport("gzip")
f = gzip.open(pfile)
data= pickle.load(f)
f.close()

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

Xs = transpose(new_coords)

R = kmeans(Xs,ncenters; maxiter=200, display=:iter)

clusters = Dict(i=>[] for i = 1:ncenters)

for i = 1:size(new_coords,1)
    push!(clusters[R.assignments[i]],i-1)
end

conds_flow = []
for i = 1:ncenters
    push!(conds_flow,G.compute_conductance(clusters[i]))
end

pfile = "embeddings/spectral_3_bfs.p.gz"

pickle = pyimport("pickle")
gzip = pyimport("gzip")
f = gzip.open(pfile)
data= pickle.load(f)
f.close()

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

Xs = transpose(new_coords)

R = kmeans(Xs,ncenters; maxiter=200, display=:iter)

clusters = Dict(i=>[] for i = 1:ncenters)

for i = 1:size(new_coords,1)
    push!(clusters[R.assignments[i]],i-1)
end

conds_spectral = []
for i = 1:ncenters
    push!(conds_spectral,G.compute_conductance(clusters[i]))
end

using Distributions

mpl = pyimport("matplotlib")

pyplot()
fig = mpl.pyplot.subplots(1,1,figsize=(6,4))
sns = pyimport("seaborn")
sns.distplot(conds_flow,hist=true,kde=true,ax=fig[2],
    kde_kws=Dict("kernel"=>"triw","clip"=>(0,0.99),"bw"=>0.06,"linewidth"=>3),
    hist_kws=Dict("alpha"=>0.2))
sns.distplot(conds_spectral,hist=true,kde=true,ax=fig[2],
    kde_kws=Dict("kernel"=>"triw","clip"=>(0,1),"bw"=>0.06,"linewidth"=>3),
    hist_kws=Dict("alpha"=>0.2))

fig[2].set_xlim(0.3,1)
fig[2].set_ylim(0,12)
fig[2].set_yticks([0,2,4,6,8,10])
fig[2].spines["top"].set_visible(false)
fig[2].spines["right"].set_visible(false)
fig[2].spines["left"].set_visible(false)
fig[2].spines["bottom"].set_visible(false)
fig[2].tick_params(axis="y", which="both", length=5,width=2)
fig[2].tick_params(axis="x", which="both", length=5,width=2)
for tick in fig[2].xaxis.get_major_ticks()
    tick.label.set_fontsize(15)
end
for tick in fig[2].yaxis.get_major_ticks()
    tick.label.set_fontsize(15)
end
fig[2].set_xlabel("ϕ(kmeans clusters)",fontsize=15)
fig[2].set_ylabel("Probability density",fontsize=15)
fig[1].legend(fig[2].get_lines(),["Local Flow Embedding","Local Spectral Embedding"],fancybox=true, shadow=true,
    fontsize=15,bbox_to_anchor=(0.73,0.85))
fig[1].savefig("cond_dist_200.pdf",format="pdf",bbox_inches="tight")
fig[1]

#################################################################################
#################################################################################

ncenters = 50

pfile = "embeddings/flow_3_bfs_delta_0.1.p.gz"

pickle = pyimport("pickle")
gzip = pyimport("gzip")
f = gzip.open(pfile)
data= pickle.load(f)
f.close()

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

new_coords = U[:,[1,2]]

Xs = transpose(new_coords)

R = kmeans(Xs,ncenters; maxiter=200, display=:iter)

clusters = Dict(i=>[] for i = 1:ncenters)

for i = 1:size(new_coords,1)
    push!(clusters[R.assignments[i]],i-1)
end

conds_flow = []
for i = 1:ncenters
    push!(conds_flow,G.compute_conductance(clusters[i]))
end

pfile = "embeddings/spectral_3_bfs.p.gz"

pickle = pyimport("pickle")
gzip = pyimport("gzip")
f = gzip.open(pfile)
data= pickle.load(f)
f.close()

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

new_coords = U[:,[1,2]]

Xs = transpose(new_coords)

R = kmeans(Xs,ncenters; maxiter=200, display=:iter)

clusters = Dict(i=>[] for i = 1:ncenters)

for i = 1:size(new_coords,1)
    push!(clusters[R.assignments[i]],i-1)
end

conds_spectral = []
for i = 1:ncenters
    push!(conds_spectral,G.compute_conductance(clusters[i]))
end

using Distributions

mpl = pyimport("matplotlib")

pyplot()
fig = mpl.pyplot.subplots(1,1,figsize=(6,4))
sns = pyimport("seaborn")
sns.distplot(conds_flow,hist=true,kde=true,ax=fig[2],
    kde_kws=Dict("kernel"=>"triw","clip"=>(0,0.99),"bw"=>0.1,"linewidth"=>3),
    hist_kws=Dict("alpha"=>0.2))
sns.distplot(conds_spectral,hist=true,kde=true,ax=fig[2],
    kde_kws=Dict("kernel"=>"triw","clip"=>(0,1),"bw"=>0.08,"linewidth"=>3),
    hist_kws=Dict("alpha"=>0.2))

fig[2].set_xlim(0.3,1)
fig[2].set_ylim(0,8)
fig[2].set_yticks([0,2,4,6])
fig[2].spines["top"].set_visible(false)
fig[2].spines["right"].set_visible(false)
fig[2].spines["left"].set_visible(false)
fig[2].spines["bottom"].set_visible(false)
fig[2].tick_params(axis="y", which="both", length=5,width=2)
fig[2].tick_params(axis="x", which="both", length=5,width=2)
for tick in fig[2].xaxis.get_major_ticks()
    tick.label.set_fontsize(15)
end
for tick in fig[2].yaxis.get_major_ticks()
    tick.label.set_fontsize(15)
end
fig[2].set_xlabel("ϕ(kmeans clusters)",fontsize=15)
fig[2].set_ylabel("Probability density",fontsize=15)
fig[1].legend(fig[2].get_lines(),["Local Flow Embedding","Local Spectral Embedding"],fancybox=true, shadow=true,
    fontsize=15,bbox_to_anchor=(0.73,0.85))
fig[1].savefig("cond_dist_50_no_perm.pdf",format="pdf",bbox_inches="tight")
fig[1]

#################################################################################

ncenters = 100

pfile = "embeddings/flow_3_bfs_delta_0.1.p.gz"

pickle = pyimport("pickle")
gzip = pyimport("gzip")
f = gzip.open(pfile)
data= pickle.load(f)
f.close()

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

new_coords = U[:,[1,2]]

Xs = transpose(new_coords)

R = kmeans(Xs,ncenters; maxiter=200, display=:iter)

clusters = Dict(i=>[] for i = 1:ncenters)

for i = 1:size(new_coords,1)
    push!(clusters[R.assignments[i]],i-1)
end

conds_flow = []
for i = 1:ncenters
    push!(conds_flow,G.compute_conductance(clusters[i]))
end

pfile = "embeddings/spectral_3_bfs.p.gz"

pickle = pyimport("pickle")
gzip = pyimport("gzip")
f = gzip.open(pfile)
data= pickle.load(f)
f.close()

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

new_coords = U[:,[1,2]]

Xs = transpose(new_coords)

R = kmeans(Xs,ncenters; maxiter=200, display=:iter)

clusters = Dict(i=>[] for i = 1:ncenters)

for i = 1:size(new_coords,1)
    push!(clusters[R.assignments[i]],i-1)
end

conds_spectral = []
for i = 1:ncenters
    push!(conds_spectral,G.compute_conductance(clusters[i]))
end

using Distributions

mpl = pyimport("matplotlib")

pyplot()
fig = mpl.pyplot.subplots(1,1,figsize=(6,4))
sns = pyimport("seaborn")
sns.distplot(conds_flow,hist=true,kde=true,ax=fig[2],
    kde_kws=Dict("kernel"=>"triw","clip"=>(0,0.99),"bw"=>0.06,"linewidth"=>3),
    hist_kws=Dict("alpha"=>0.2))
sns.distplot(conds_spectral,hist=true,kde=true,ax=fig[2],
    kde_kws=Dict("kernel"=>"triw","clip"=>(0,1),"bw"=>0.06,"linewidth"=>3),
    hist_kws=Dict("alpha"=>0.2))

fig[2].set_xlim(0.3,1)
fig[2].set_ylim(0,10)
fig[2].set_yticks([0,2,4,6,8])
fig[2].spines["top"].set_visible(false)
fig[2].spines["right"].set_visible(false)
fig[2].spines["left"].set_visible(false)
fig[2].spines["bottom"].set_visible(false)
fig[2].tick_params(axis="y", which="both", length=5,width=2)
fig[2].tick_params(axis="x", which="both", length=5,width=2)
for tick in fig[2].xaxis.get_major_ticks()
    tick.label.set_fontsize(15)
end
for tick in fig[2].yaxis.get_major_ticks()
    tick.label.set_fontsize(15)
end
fig[2].set_xlabel("ϕ(kmeans clusters)",fontsize=15)
fig[2].set_ylabel("Probability density",fontsize=15)
fig[1].legend(fig[2].get_lines(),["Local Flow Embedding","Local Spectral Embedding"],fancybox=true, shadow=true,
    fontsize=15,bbox_to_anchor=(0.73,0.85))
fig[1].savefig("cond_dist_100_no_perm.pdf",format="pdf",bbox_inches="tight")
fig[1]


###############################################################################


ncenters = 200

pfile = "embeddings/flow_3_bfs_delta_0.1.p.gz"

pickle = pyimport("pickle")
gzip = pyimport("gzip")
f = gzip.open(pfile)
data= pickle.load(f)
f.close()

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

new_coords = U[:,[1,2]]

Xs = transpose(new_coords)

R = kmeans(Xs,ncenters; maxiter=200, display=:iter)

clusters = Dict(i=>[] for i = 1:ncenters)

for i = 1:size(new_coords,1)
    push!(clusters[R.assignments[i]],i-1)
end

conds_flow = []
for i = 1:ncenters
    push!(conds_flow,G.compute_conductance(clusters[i]))
end

pfile = "embeddings/spectral_3_bfs.p.gz"

pickle = pyimport("pickle")
gzip = pyimport("gzip")
f = gzip.open(pfile)
data= pickle.load(f)
f.close()

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

new_coords = U[:,[1,2]]

Xs = transpose(new_coords)

R = kmeans(Xs,ncenters; maxiter=200, display=:iter)

clusters = Dict(i=>[] for i = 1:ncenters)

for i = 1:size(new_coords,1)
    push!(clusters[R.assignments[i]],i-1)
end

conds_spectral = []
for i = 1:ncenters
    push!(conds_spectral,G.compute_conductance(clusters[i]))
end

using Distributions

mpl = pyimport("matplotlib")

pyplot()
fig = mpl.pyplot.subplots(1,1,figsize=(6,4))
sns = pyimport("seaborn")
sns.distplot(conds_flow,hist=true,kde=true,ax=fig[2],
    kde_kws=Dict("kernel"=>"triw","clip"=>(0,0.99),"bw"=>0.04,"linewidth"=>3),
    hist_kws=Dict("alpha"=>0.2))
sns.distplot(conds_spectral,hist=true,kde=true,ax=fig[2],
    kde_kws=Dict("kernel"=>"triw","clip"=>(0,1),"bw"=>0.04,"linewidth"=>3),
    hist_kws=Dict("alpha"=>0.2))

fig[2].set_xlim(0.3,1)
fig[2].set_ylim(0,16)
fig[2].set_yticks([0,2,4,6,8,10,12,14])
fig[2].spines["top"].set_visible(false)
fig[2].spines["right"].set_visible(false)
fig[2].spines["left"].set_visible(false)
fig[2].spines["bottom"].set_visible(false)
fig[2].tick_params(axis="y", which="both", length=5,width=2)
fig[2].tick_params(axis="x", which="both", length=5,width=2)
for tick in fig[2].xaxis.get_major_ticks()
    tick.label.set_fontsize(15)
end
for tick in fig[2].yaxis.get_major_ticks()
    tick.label.set_fontsize(15)
end
fig[2].set_xlabel("ϕ(kmeans clusters)",fontsize=15)
fig[2].set_ylabel("Probability density",fontsize=15)
fig[1].legend(fig[2].get_lines(),["Local Flow Embedding","Local Spectral Embedding"],fancybox=true, shadow=true,
    fontsize=15,bbox_to_anchor=(0.73,0.85))
fig[1].savefig("cond_dist_200_no_perm.pdf",format="pdf",bbox_inches="tight")
fig[1]
