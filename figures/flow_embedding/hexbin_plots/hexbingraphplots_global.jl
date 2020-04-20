## Load data for testing
using Plots
using SparseArrays
using LinearAlgebra
using TextParse
using DelimitedFiles
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
# x, y = csvread("$../../dataset/lawlor-spectra-k32.edgelist.xy";
#         spacedelim=true, colparsers=[Float64,Float64],
#         header_exists=false)[1]
V = readdlm("../../dataset/lawlor-spectra-k32.coords")
V[:,1] *= -10
V[:,2] *= -4

function plotperm(v)
  return invperm(sortperm(v))
end

x = Array{Float64}(plotperm(V[:,1]))
y = Array{Float64}(plotperm(V[:,2]))

## Form a random subset of edges
using Random
Random.seed!(0)
p = randperm(length(src))[1:10^5] # random set of 1M edges
srcsub = Int.(src[p])
dstsub = Int.(dst[p])

## Hexbin graph plots
include("hexbinplots_orig.jl")
function shapecoords(hx::HexBinPlots.HexHistogram)
  h,vh = HexBinPlots.make_shapes(hx)
  vmax = maximum(vh)
  xsh = Vector{Float64}()
  ysh = Vector{Float64}()
  for k in eachindex(h)
      append!(xsh,h[k].x)
      push!(xsh,h[k].x[1])
      push!(xsh,NaN)
      append!(ysh,h[k].y)
      push!(ysh,h[k].y[1])
      push!(ysh,NaN)
  end
  return (xsh, ysh, vh)
end
using Interpolations
function control_point(xi, xj, yi, yj, dist_from_mid)
    xmid = 0.5 * (xi+xj)
    ymid = 0.5 * (yi+yj)
    # get the angle of y relative to x
    theta = atan((yj-yi) / (xj-xi)) + 0.5pi
    # dist = sqrt((xj-xi)^2 + (yj-yi)^2)
    # dist_from_mid = curvature_scalar * 0.5dist
    # now we have polar coords, we can compute the position, adding to the midpoint
    (xmid + dist_from_mid * cos(theta),
     ymid + dist_from_mid * sin(theta))
end
function _add_edge_points!(xlines,ylines,xsi,ysi,xdi,ydi,curve_amount,interplen)
  xpt, ypt = control_point(xsi,xdi,ysi,ydi,curve_amount)
  xpts = [xsi, xpt, xdi]
  ypts = [ysi, ypt, ydi]
  t = range(0, stop=1, length=3)
  A = hcat(xpts, ypts)
  itp = scale(interpolate(A, BSpline(Cubic(Natural(OnGrid())))), t, 1:2)
  tfine = range(0, 1, length=interplen)
  for t in tfine
    push!(xlines, itp(t, 1))
  end
  for t in tfine
    push!(ylines, itp(t, 2))
  end
end

# Mutating version from HexBinPlots
function _xy2counts!(counts::Dict{(Tuple{Int, Int}), Int},
                    x::AbstractArray,y::AbstractArray,
                    xsize::Real,ysize::Real,x0,y0)
  @inbounds for i in eachindex(x)
      h = convert(HexagonOffsetOddR,
                  cube_round(x[i] - x0,y[i] - y0,xsize, ysize))
      idx = (h.q, h.r)
      counts[idx] = 1 + get(counts,idx,0)
  end
  counts
end
using Statistics
using Hexagons
# Generate counts for a graph edge plot histogram
# hhparam = (xsize,ysize,x0,y0) are the hex-histogrma params
function graphedgehist(src,dst,x,y,ptscale,hhparams,curvature = 0.05)
  xlines = similar(x,0)
  ylines = similar(y,0)
  sizehint!(xlines,max(length(src)+1000,10000000)) # we'll process them in batches this big
  sizehint!(ylines,max(length(src)+1000,10000000))

  # this is the counts of the edge vectors...
  edgecounts = Dict{(Tuple{Int, Int}), Int}()

  # now, run through each edges...
  for ei in eachindex(src)
    si = src[ei]
    di = dst[ei]
    xsi = x[si]
    ysi = y[si]
    xdi = x[di]
    ydi = y[di]
    curve_amount = curvature*sign((di-si))

    edgelen = sqrt((xdi-xsi)^2 + (ydi-ysi)^2)
    interplen = max(ceil(Int,2*edgelen/ptscale),2)

    _add_edge_points!(xlines, ylines, xsi, ysi, xdi, ydi, curve_amount, interplen)

    if length(xlines) >= length(src)
      # compress all to edgecounts
      println("Compressing counts... $(ei) / $(length(src))")
      _xy2counts!(edgecounts, xlines, ylines, hhparams...)
      resize!(xlines,0)
      resize!(ylines,0)
    end
  end
  # handle final update
  return _xy2counts!(edgecounts, xlines, ylines, hhparams...)
end

function hexbingraphplot(src,dst,x,y;nbins=500)
  xbins = nbins
  ybins = nbins

  # compute params for hex histogram bins
  xmin,xmax = extrema(x)
  ymin,ymax = extrema(y)
  xspan, yspan = xmax - xmin, ymax - ymin
  xsize, ysize = xspan / xbins, yspan / ybins
  x0, y0 = xmin - xspan / 2,ymin - yspan / 2
  hhparams = (xsize, ysize, x0, y0) # save them as a tuple

  edgescale = 6
  hsize = min((xmax-xmin)/(xbins*edgescale),(ymax-ymin)/(ybins*edgescale))

  # need to compute hhparams

  # get coords for edges
  counts = graphedgehist(src,dst,x,y,hsize,hhparams)
  # do this for the nodes...
  #counts = HexBinPlots.xy2counts_(x,y,hhparams...)

  xh,yh,vh = HexBinPlots.counts2xy_(counts,hhparams...)
  hedges = HexBinPlots.HexHistogram(xh,yh,vh,xsize,ysize,false)


  #hx = fit(HexBinPlots.HexHistogram,x,y,nbins)
  xsh,ysh,vh = shapecoords(hedges)

  #plot(xsh, ysh, seriestype=:shape, alpha=log10.(vh)/maximum(log10.(vh)), fill_z=vh, legend=false,linealpha=0)
  return hedges
end
@time hedges = hexbingraphplot(src,dst,x,y,nbins=1000)
##
xsh,ysh,vh = shapecoords(hedges)

val_range = (minimum(V[:,1]),0.008)
colors = [ColorSchemes.get(ColorSchemes.CMRmap,i,val_range) for i in V[:,1]]

axform = (log10.(vh).^3)/maximum(log10.(vh).^3)
avals = axform/maximum(axform)
plot(xsh, ysh, seriestype=:shape, alpha=avals, fillcolor=:darkblue, legend=false,linealpha=0, framestyle=:none,background=nothing)
scatter!(x,y,markercolor=colors,alpha=0.1,markersize=1,legend=false,colorbar=false,markerstrokewidth=0,background=nothing)
##
plot!(dpi=400,size=(1200,1200)) # size is inches times 100, so this should be about 3600x3600
png("hexbin_plot_global_high_res.png")


axform = (log10.(vh).^3)/maximum(log10.(vh).^3)
avals = axform/maximum(axform)
plot(xsh, ysh, seriestype=:shape, alpha=avals, fillcolor=:darkblue, legend=false,linealpha=0, framestyle=:none,background=nothing)
scatter!(x,y,markercolor=colors,alpha=0.1,markersize=1,legend=false,colorbar=false,markerstrokewidth=0,background=nothing)
plot!(dpi=150,size=(1200,1200)) # size is inches times 100, so this should be about 3600x3600
png("hexbin_plot_global_low_res.png")
