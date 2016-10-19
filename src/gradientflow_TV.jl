# A Hele Shaw tumor model as a gradient flow
# @ Lénaïc Chizat 2016

module gradientflowTV

export onestepflow

################################################################################
## I. Definitions of some numeric functions ####################################
################################################################################

"Special x/y with convention x/0=0 if x≧0"
function div0(x,y)
  if x>=0.0 && y> 0.0
    return x/y
  elseif (x >= 0.0 && y==0)
    return zero(x)
  else
    return oftype(x,Inf)
  end
end

@vectorize_2arg Number div0

function proxdiv1(a,u,ϵ,p)  return div0(p,a) end

function proxdivdiff(a,b,u,v,ϵ,τ) # only two args, on place (for speed)
  la, lb= ϵ*log(a), ϵ*log(b)
  return if (a.<=0.0) & (b.<=0.0)
    zero(a), zero(b)
  elseif  (la-lb+v-u .>  2τ) | (b.==0.0) | isinf(v) # I
    !isinf(u).*exp(-(u+τ)/ϵ),  !isinf(v).*exp(-(v-τ)/ϵ)
  elseif (la-lb+v-u .< -2τ) | (a.==0.0) | isinf(u) # J
    !isinf(u).*exp(-(u-τ)/ϵ),  !isinf(v).*exp(-(v+τ)/ϵ)
  else
    exp((lb-la-u-v)/(2ϵ)), exp((la-lb-u-v)/(2ϵ))
  end
end

function proxdiv2(b,v,ϵ,τ)
  a = zeros(b)
  for i=1:div(length(b),2)
    a[2i-1], a[2i] = proxdivdiff(b[2i-1], b[2i],v[2i-1],v[2i],ϵ,τ)
  end
  return a
end

function proxdiv3(b,v,ϵ,τ)
  a = zeros(b)
  for i=1:div(length(b),2)-1
    a[2i], a[2i+1] = proxdivdiff(b[2i], b[2i+1],v[2i],v[2i+1],ϵ,τ)
  end
  return a
end


"Compute the kernel given the potentials"
function computekernel!(K::SparseMatrixCSC, X::AbstractArray, u, v, ϵ, tol)
  nzvalK = K.nzval
  rows = K.rowval
  for col = 1:K.n # efficient iterations
    for i in nzrange(K, col)
      row = rows[i]
      xx = exp((u[row]+v[col]-sum((X[row,:]-X[col,:]).^2))/ϵ)
      nzvalK[i] = (xx <tol? zero(xx) : xx)
    end
  end
  dropzeros!(K)
  return Void
end

"Initialize the sparse kernel support"
function kernelinit(X,τ,dim)
  if dim==1
    return sparse(map(Float64,broadcast((x,y)->(y-x)^2, X, X').<(5τ)^2))
  elseif dim==2
    return sparse(map(Float64,broadcast((x1,x2,y1,y2)->(y1-x1)^2+(y2-x2)^2,
              X[:,1],X[:,2],X[:,1]',X[:,2]').<(5τ)^2))
  else     error("unknown dim in kernelinit")  end
 end

"Return the vector of coordinates"
function coordvect(N,dim)
  if dim==1
    return linspace(.5/N,1-.5/N,N)
  elseif dim==2
    z = linspace(.5/N,1-.5/N,N)
    return [repmat(z',N,1)[:] repmat(z,1,N)[:]]
  else    error("unknown dim in coordvect")  end
 end

################################################################################
## II. Multiresolution #########################################################
################################################################################

function coarser(p,n,dim) return (dim==1? coarser1d(p,n) : coarser2d(p,n)) end
function finer(p,n,dim)   return (dim==1? finer1d(p,n)   : finer2d(p,n))   end
function finerplan(K,dim) return (dim==1? finerplan1d(K) : finerplan2d(K)) end

function coarser1d(p0,n)
  p = zeros(div(length(p0),2^n))
  for i=1:length(p)
    p[i] = mean(p0[1+(i-1)*2^n:i*2^n])
  end
  return p
end

function coarser2d(p0,n)
  N0 = map(Int,sqrt(length(p0)))
  N  = div(N0,2^n)
  P0 = reshape(p0,N0,N0)
  P  = zeros(N,N)
  for i = 1:N
    for j = 1:N
      P[i,j] = mean(P0[1+(i-1)*2^n:i*2^n,1+(j-1)*2^n:j*2^n])
    end
  end
  return P[:]
end

function finer1d(p0,n)
  p = zeros(length(p0)*2^n)
  for i=1:length(p)
    p[i] = p0[div(i-1,2^n)+1]
  end
  return p
end

function finer2d(p0,n)
  N0 = map(Int,sqrt(length(p0)))
  N  = N0*2^n
  P0 = reshape(p0,N0,N0)
  P  = zeros(N,N)
  for i = 1:N
    for j = 1:N
      P[i,j] = P0[div(i-1,2^n)+1,div(j-1,2^n)+1]
    end
  end
  return P[:]
end

# return a new plan twice finer with same sparsity scheme
function finerplan1d(K)
  (I0,J0) = findn(K)
  n       = length(I0)
  I,J     = zeros(Int,4n), zeros(Int,4n)
  for k = 1:n
    ind, i, j = 4*(k-1), 2*(I0[k]-1)+1, 2*(J0[k]-1)+1
    I[1+ind], J[1+ind] = i,   j
    I[2+ind], J[2+ind] = i+1, j
    I[3+ind], J[3+ind] = i  , j+1
    I[4+ind], J[4+ind] = i+1, j+1
  end
  N = size(K,1)*2
  return sparse(I,J,1.,N,N)
end

function finerplan2d(K)
  (I0,J0) = findn(K)
  N0      = map(Int,sqrt(size(K,1)))
  nz      = length(I0)
  I,J     = zeros(Int,16nz), zeros(Int,16nz)
  f(i,ii) = i + (ii-1)*2N0
  for k = 1:nz
    ind = 16*(k-1)
    i, ii = 2*mod(I0[k]-1,N0)+1, 2*div(I0[k]-1,N0) + 1
    j, jj = 2*mod(J0[k]-1,N0)+1, 2*div(J0[k]-1,N0) + 1
    I[1+ind], J[1+ind]   = f(i,  ii),   f(j,  jj)
    I[2+ind], J[2+ind]   = f(i+1,ii),   f(j,  jj)
    I[3+ind], J[3+ind]   = f(i,  ii+1), f(j,  jj)
    I[4+ind], J[4+ind]   = f(i+1,ii+1), f(j,  jj)
    I[5+ind], J[5+ind]   = f(i,  ii),   f(j+1,jj)
    I[6+ind], J[6+ind]   = f(i+1,ii),   f(j+1,jj)
    I[7+ind], J[7+ind]   = f(i,  ii+1), f(j+1,jj)
    I[8+ind], J[8+ind]   = f(i+1,ii+1), f(j+1,jj)
    I[9+ind], J[9+ind]   = f(i,  ii),   f(j,  jj+1)
    I[10+ind], J[10+ind] = f(i+1,ii),   f(j,  jj+1)
    I[11+ind], J[11+ind] = f(i,  ii+1), f(j,  jj+1)
    I[12+ind], J[12+ind] = f(i+1,ii+1), f(j,  jj+1)
    I[13+ind], J[13+ind] = f(i,  ii),   f(j+1,jj+1)
    I[14+ind], J[14+ind] = f(i+1,ii),   f(j+1,jj+1)
    I[15+ind], J[15+ind] = f(i,  ii+1), f(j+1,jj+1)
    I[16+ind], J[16+ind] = f(i+1,ii+1), f(j+1,jj+1)
  end
  return sparse(I,J,1.,4N0^2,4N0^2)
end

################################################################################
## II. Scaling algorithm and multiresolution ###################################
################################################################################

"Scaling iterations"
function scalingiter(p, τ, X, K, u, v, w, ϵ, niter;tol=0,walls=falses(p))
  dx = 1/size(X,1)
  a,b,c = ones(size(X,1)),ones(size(X,1)),ones(size(X,1))
  computekernel!(K,X,u,v+w,ϵ,tol)
  while maximum(K)==Inf  # to make sure there is no +∞ in the kernel
    maximum([u v w])==Inf ? error(): println("ϵ too small thus doubled")
    ϵ = 2*ϵ
    computekernel!(K,X,u,v+w,ϵ,tol)
  end
  for i = 1:niter
      a = proxdiv1(K*(b.*c*dx),u,ϵ,p)
      Kax = K'*a*dx
      for l=1:div(size(X,1),1)
        b = proxdiv2(c.*Kax,v,ϵ,τ)
        c = proxdiv3(b.*Kax,w,ϵ,τ)
        if maximum(abs([a;b;c]))>1e5
          u, v, w = u + ϵ*map(log,a), v + ϵ*map(log,b), w + ϵ*map(log,c) # absorb
          computekernel!(K,X,u,v+w,ϵ,tol)
          a,b,c = ones(size(X,1)),ones(size(X,1)),ones(size(X,1))
          Kax = K'*a*dx
        end
      end
  end
  @show maximum(abs([a;b;c])) # numerical overflow?
  u, v, w = u + ϵ*map(log,a), v + ϵ*map(log,b), w + ϵ*map(log,c) # absorb
  computekernel!(K,X,u,v+w,ϵ,tol)
  return K,u,v,w
end # scalingiter

"""
Performs one step of the flow, including ϵ scaling and multiresolution
the values of ϵ start from 1 and is divided by epsstep each step until ϵ
the multiresolution starts for problems bigger than 4096
"""
function onestepflow(p0,τ,dim,ϵ;niter=300, walls=falses(p0), epsstep=6., tol=0.)
    N0 = map(Int,length(p0)^(1/dim))
    neps = map(Int,max(2,ceil(-log(ϵ)/log(epsstep)))) # number of changes of ϵ
    epsvec = logspace(0.,log10(ϵ),neps)
    nscales = clamp(map(Int, ceil(log2(length(p0)/4096)/dim)),0,neps-1)
    @assert mod(N0,2^nscales) == 0.
    scales = nscales-[zeros(Int,neps-nscales) ; 1:nscales]

    p = coarser(p0,scales[1],dim)
    W = coarser(walls,scales[1],dim).>=1
    u,v,w= zeros(p), zeros(p), zeros(p)
    X = coordvect(div(N0,2^scales[1]),dim)
    K = kernelinit(X,τ,dim)
    for k=1:neps
      @show k
      if k>1 && scales[k]<scales[k-1]
        u,v,w = finer(u,1,dim), finer(v,1,dim), finer(w,1,dim)
        K     = finerplan(K,dim)               # RENAME refine_plan_support
        p     = coarser(p0,scales[k],dim)
        W     = coarser(walls,scales[k],dim).>=1
        X     = coordvect(div(N0,2^scales[k]),dim)
      end
      (K,u,v,w) = scalingiter(p,τ,X,K,u,v,w, epsvec[k], niter;tol=tol,walls=W)
    end
    q  = sum(K,1)[:]/length(p0)
    #pr = (2τ-1+exp(-v))/2
  return q, u,v,w
end


end #module
