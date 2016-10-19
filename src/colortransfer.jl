# Color transfer using unbalanced optimal transport
# @ Lénaïc Chizat 2016

module ColorTransfer

export  convUOT, transportmap,
        im2color, color2im, myhist

using Images, Colors

"Special x/y with convention x/0=0 if x≧0"
function div0(x::Number,y::Number)
  if x>=0.0 && y> 0.0
    return x/y
  elseif (x >= 0.0 && y==0.0)
    return zero(x)
  else
    return oftype(x,Inf)
  end
end

function A_mul_B_perm!(dest, M::AbstractMatrix, src, dim::Integer)
    order = [dim; setdiff(1:ndims(src), dim)]
    srcp = permutedims(src, order)
    tmp = Array(eltype(dest), size(dest, dim), div(length(dest), size(dest, dim)))
    A_mul_B!(tmp, M, reshape(srcp, (size(src,dim), div(length(srcp), size(src,dim)))))
    iorder = [2:dim; 1; dim+1:ndims(src)]
    permutedims!(dest, reshape(tmp, size(dest)[order]), iorder)
    dest
end

"Convolute src along each dimension k by vs[k]"
function convol!(dest,vs,src)
  A_mul_B_perm!(dest,vs[1],src,1)
  for k=2:length(vs)
    A_mul_B_perm!(dest,vs[k],dest,k)
  end
end

function convol(vs,src)
  dest = similar(src)
  A_mul_B_perm!(dest,vs[1],src,1)
  for k=2:length(vs)
    A_mul_B_perm!(dest,vs[k],dest,k)
  end
  return dest
end


function proxdiv!(dest,φ,p,s,ϵ)
  if φ[:type] == "OT" || (in(φ[:type],["KL","TV"]) && φ[:param] == Inf)
    dest .= div0.(p,s)

  elseif φ[:type] == "KL"
    λ = φ[:param]
    dest .= div0.(p,s).^(λ/(λ+ϵ))

  elseif φ[:type] == "TV"
    λ = φ[:param]
    dest .= min( exp(λ/ϵ), max(exp(-λ/ϵ), div0.(p,s)) )

  elseif φ[:type] == "RG"
    β1,β2 = φ[:param]
    @assert 0 <= β1 <= β2 < Inf
    dest .= min(β2*div0.(p,s), max(β1*div0.(p,s),ones(p)))
  else
    error("Type of φ not recognized")
  end
end


"""
```
convUOT(marg1,marg2,prox1,prox2, [; lengths=ones, niter=1000, nsink=20]) -> kernel, a, b, gap
```
Computes fast unbalanceed optimal transport for the quadratic cost Only expected inputs
are the marginals and the divergences. Uses the separability of the Gibbs kernel for speed.
"""
function convUOT(p,q,F1,F2,ϵ; ll= ones(size(p)), niter=1e3)
  @assert all(p.>=0.) && all(q.>=0.)
  M = ntuple(k->Float64[exp(-((i-j)*ll[k]).^2/((size(p,k)-1)^2*ϵ))
                          for i=1:size(p,k), j=1:size(q,k)],ndims(p))
  #kernel  = gaussianconv(ϵ,size(p)...; l=ll)
  kernel(a) = convol(M,a)
  a, b, Ka, Kb = ones(p), ones(q), ones(p), ones(q)
  for i = 1:niter
    convol!(Kb,M,b)
    proxdiv!(a,F1,p,Kb,ϵ)
    convol!(Ka,M,a)
    proxdiv!(b,F2,q,Ka,ϵ)
    !all(isfinite([a[:];b[:]]))? error("Scalings exploded : increase ϵ(iter $(i)") : nothing
  end
  return kernel, a, b
end # convUOT


"""
Compute a transport map T : X ⊂ R^n -> Y ⊂ R^m using barycentric projection.
Expressed in indices coordinates. Identity is returned when T is not defined.
"""
# T[n][i,j,..] gives the n-th coord. of the image of point [i,j,...]
function transportmap(a, b, K)
  isa(K,Function)? KK = K : KK = x->K*x
  n, m = ndims(a), ndims(b)
  coordsX, coordsY = ntuple(i->1:size(a,i),n), ntuple(i->1:size(b,i),m)
  X, Y =  ndgrid(coordsX...), ndgrid(coordsY...)
  μ = a.* KK(b)
  mask = (μ .> (maximum(μ)*1e-20)) # for stability

  transport(i)    = min(size(b,i), max(1, div0.(a,μ).*KK(b.*Y[i]) ))
  identitymap(i)  = 1 + (X[i]-1)*size(b,i)/size(a,i)
  masktransport(i)= round(Int64,transport(i).*mask + identitymap(i).*(!mask))
  T = ntuple(i->masktransport(i),n)
end


"""
  f = im2color(img, colorspace)

`img` is obtained with `load("filename")`
"""
function im2color(img,colorspace)
  if colorspace == "LAB"
    xyz = convert(Image{Lab}, float32(img))
    X = Float32[xyz[i,j].l for i = 1:size(xyz,1), j=1:size(xyz,2)]
    Y = Float32[xyz[i,j].a for i = 1:size(xyz,1), j=1:size(xyz,2)]
    Z = Float32[xyz[i,j].b for i = 1:size(xyz,1), j=1:size(xyz,2)]
  elseif colorspace == "LUV"
    xyz = convert(Image{Luv}, float32(img))
    X = Float32[xyz[i,j].l for i = 1:size(xyz,1), j=1:size(xyz,2)]
    Y = Float32[xyz[i,j].u for i = 1:size(xyz,1), j=1:size(xyz,2)]
    Z = Float32[xyz[i,j].v for i = 1:size(xyz,1), j=1:size(xyz,2)]
  elseif colorspace == "RGB"
    xyz = convert(Image{RGB}, float32(img))
    X = Float32[xyz[i,j].r for i = 1:size(xyz,1), j=1:size(xyz,2)]
    Y = Float32[xyz[i,j].g for i = 1:size(xyz,1), j=1:size(xyz,2)]
    Z = Float32[xyz[i,j].b for i = 1:size(xyz,1), j=1:size(xyz,2)]
  else
    error("Colorspace not supported")
  end
  return [X[:] Y[:] Z[:]]
end

"""
  new_im = Lab2im(L,A,B,im)

`im` is an image of same type (only used for its properties)
"""
function color2im(f, colorspace, im; s =size(im.data))
  if colorspace == "LAB"
    R = [Lab{Float32}(f[i,:]...) for i=1:size(f,1)]
  elseif colorspace == "LUV"
    R = [Luv{Float32}(f[i,:]...) for i=1:size(f,1)]
  elseif colorspace == "RGB"
    R = [RGB{Float32}(f[i,:]...) for i=1:size(f,1)]
  else
    error("Colorspace not supported")
  end
  return copyproperties(im, reshape(R,s...))
end

"""
```
myhist(v, nbins[, maxs, mins]) -> bincenters, counts, coords
```
Returns the histogram of `v` for points in arbitrary dimension.
Each row of `v` is treated as a `d`-dimensional point and `nbins`
is a vector of length `d` giving the number of bins for each dimension.
"""
function myhist(v, nbins; mins = minimum(v,1), maxs = maximum(v,1))

    @assert all(mins.<=minimum(v,1))&&all(maxs.>=maximum(v,1))
    maxs[mins .== maxs] += 1.0
    bincenters  = [linspace(mins[i]+.5/nbins[i], maxs[i]-.5/nbins[i], nbins[i])
         for i = 1:length(nbins)]
    counts = zeros(nbins...)
    coords = Array{Int}(size(v,1),length(nbins))

    for i = 1:size(v,1)
        for j = 1:length(nbins)
            coords[i,j] = 1 + floor(Int,
                (nbins[j]-1)*(v[i,j]-mins[j])/(maxs[j]-mins[j]))
        end
        counts[coords[i,:]...] += 1
    end

    return bincenters, counts, coords
end # myhist

function ndgrid{T}(vs::AbstractVector{T}...) # taken from Julia examples files
    n = length(vs)
    sz = map(length, vs)
    out = ntuple(i->Array(T, sz), n)
    s = 1
    for i=1:n
        a = out[i]::Array
        v = vs[i]
        snext = s*size(a,i)
        ndgrid_fill(a, v, s, snext)
        s = snext
    end
    out
end

function ndgrid_fill(a, v, s, snext) # taken from Julia examples files
    for j = 1:length(a)
        a[j] = v[div(rem(j-1, snext), s)+1]
    end
end





end # end of module
