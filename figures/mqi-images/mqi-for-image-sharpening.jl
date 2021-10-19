## Goals:
# For the flow-clustering paper, a referee asked about how these ideas could
# be used on images.
# Here, we provide an example of how they can be used on images.
# This is based on the mqi-animation figures I made for the Argonne seminar.
# I copied "code" from that directory
# cp -r  ~/Dropbox/publications/flow-based-argonne/code .

# Originally based on dgleich/Dropbox/research/2021/09-22-mqi-figs-for-image

# Use Nate's flowseed code. (Push Relabel with customizable seeds... )
include("FlowSeed.jl")

## Setup code to work on a grid graph...
using SparseArrays

function grid_graph(m::Int, n::Int; distance::Int=1)
  N = m*n
  imap = reshape(1:N, m, n)
  ei = zeros(Int,0)
  ej = zeros(Int,0)
  for i=1:m
    for j=1:n
      # left-right neighbors
      for di=-distance:distance
        dj = 0
        if i+di >= 1 && i+di <= m && j+dj >= 1 && j+dj <= n
          src = imap[i,j]
          dst = imap[i+di,j+dj]
          if (src != dst)
            push!(ei,src)
            push!(ej,dst)
          end
        end
      end
      # up-down neighbors
      for dj=-distance:distance
        di = 0
        if i+di >= 1 && i+di <= m && j+dj >= 1 && j+dj <= n
          src = imap[i,j]
          dst = imap[i+di,j+dj]
          if (src != dst)
            push!(ei,src)
            push!(ej,dst)
          end
        end
      end
    end
  end
  return sparse(ei,ej,1.0,N,N)
end

A = grid_graph(100,100)
## Now generate a blurry image on a grid graph.
using Plots
function make_T(m::Integer,n::Integer)
  IType = typeof(m)

  # start of the left-right branch
  lstart = floor(IType,m/8)
  rend = m-lstart
  # end of the left-right branch
  tstart = floor(IType, n/8)
  bend = n-tstart

  halflstart = div(lstart,2)

  mstart = floor(IType,m/2)-lstart
  mend = floor(IType,m/2)+lstart

  X = zeros(m,n)
  X[lstart:rend,tstart:(tstart+lstart)] .= 1
  X[mstart:mend,tstart:bend] .= 1
  return X
end
X = make_T(100,100)
heatmap((make_T(100,100)))
## blur this picture.
using Images, Random
Random.seed!(0)
T = imfilter(X, Kernel.moffat(1.5, 1.2, 5))
Tblur = copy(T)
T .+= 0.1*(2*rand(size(T)...).-1)
T = max.(min.(T/0.3,1.0), 0.0)
heatmap(T)
## Solve one step of MQI-like problem on T
# This weights the problem with a volume dependent term
# That biases the results towards the T solution.
using PerceptualColourMaps
function MQITest(A,x,delta)
  volA = sum(A)
  d = vec(sum(A,dims=1))

  R = findall(x .> 0)
  #source = zeros(size(A,1))
  source = delta*d.*x # build edge weights from source to R

  sink = volA*ones(size(A,1)) # build edge weights from Rbar to sink
  sink[R] .= 0 # no edges from R
  #sstart = set_stats(A,R,volA) # get conductance
  #println("starting cond=", sstart[4], " = ", sstart[1], "/", sstart[2])
  S = NonLocalPushRelabel(A,R,source,sink)
  if length(S) > 0
    send = set_stats(A,S,volA)
    println("  ending cond=", send[4], " = ", send[1], "/", send[2])
  else
    send = sstart
    S = R
  end
  return S, send
end
S = MQITest(A,reshape(T,100*100),0.11)[1]
SMat = begin M =zeros(100,100); M[S] .= 1; M end
SMqi = begin S = MQITest(A,reshape(T,100*100),0.04)[1]; M =zeros(100,100); M[S] .= 1; M end
plot(heatmap(1 .-X, title="original",color=cmap("L3")),
  heatmap(1 .-T,title="noisy/blury",color=cmap("L3")),
  heatmap(1 .-(0.5*SMat+0.5*X),color=cmap("L3"),title="MQI like\\n subprob (\\delta=0.11)"),
  heatmap(1 .-(0.5*SMqi+0.5*X),color=cmap("L3"),title="MQI like\\n opt (\\delta=0.04)"), xticks=[],yticks=[],
    colorbar=false, aspect_ratio=:equal, framestyle=:none, size=(600,200),
    layout=(1,4),titlefontsize=12,color=cmap("L3"))

## Save final figures... for manuscript
using PerceptualColourMaps
colorview(RGB,permutedims(applycolourmap(1 .- X, cmap("L3")), (3,1,2)))
##
save("image-1-original.png",
  map(clamp01nan, colorview(RGB,permutedims(applycolourmap(1 .- X, cmap("L3")), (3,1,2))) ))
##
save("image-1-blur.png",
  map(clamp01nan, colorview(RGB,permutedims(applycolourmap(1 .- Tblur, cmap("L3")), (3,1,2))) ))

##
save("image-1-noisy.png",
  map(clamp01nan, colorview(RGB,permutedims(applycolourmap(1 .- T, cmap("L3")), (3,1,2))) ))
##
save("image-1-colormap.png",repeat(reverse(cmap("L3")), 1, 30))
save("image-1-colormap-horiz.png",repeat(reverse(cmap("L3")), 1, 30)')

##
save("image-1-MQI-like-1.png",
  map(clamp01nan, colorview(RGB,permutedims(applycolourmap(1 .- SMat, cmap("L3")), (3,1,2))) ))

##
save("image-1-MQI-opt-1.png",
  map(clamp01nan, colorview(RGB,permutedims(applycolourmap(1 .- SMqi, cmap("L3")), (3,1,2))) ))

##
save("image-1-MQI-like-1-errors.png",
  map(clamp01nan, colorview(RGB,permutedims(applycolourmap(
    1 .- 0.5*xor.(Int.(SMat),Int.(X)), cmap("L3"), [0,1]), (3,1,2))) ))
##
save("image-1-MQI-like-1-errors-and-soln.png",
  map(clamp01nan, colorview(RGB,permutedims(applycolourmap(
    1 .- 0.5.*SMat .- 0.5.*X , cmap("L3"), [0,1]), (3,1,2))) ))

##
save("image-1-MQI-opt-1-errors.png",
  map(clamp01nan, colorview(RGB,permutedims(applycolourmap(
    1 .- 0.5*xor.(Int.(SMqi),Int.(X)), cmap("L3"), [0,1]), (3,1,2))) ))
