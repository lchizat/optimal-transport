# A Hele Shaw tumor model as a gradient flow
# @ Lénaïc Chizat 2016

module gradientflow

export onestepflow

################################################################################
## I. Definitions of some numeric functions ####################################
################################################################################

"Special x/y with convention x/0=0 if x≧0"
function div0(x,y)
  if x>=0. && y>0.
    return x/y
  elseif (x >= 0.&& y==0)
    return zero(x)
  else
    return oftype(x,Inf)
  end
end

@vectorize_2arg Number div0

function proxdiv1(a,u,ϵ,p)  return div0(p,exp(u).*a).^(1/(1+ϵ)) end

function proxdiv2(b,v,ϵ,τ,walls)
  κ     = 1-2*τ
  I     = (b .<= (exp(v)*κ^(1+ϵ)).^(1/ϵ))
  b[!I] = div0( 1, ( b[!I] .* exp(v[!I]) ).^(1/(1+ϵ)) )
  b[I]  = div0( 1, ( κ * exp(v[I]) ).^(1/ϵ) )
  b[walls]  = 0.
  return b
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
  SparseMatrix.dropzeros!(K)
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
function scalingiter(p, τ, X, K, u, v, ϵ, niter;tol=0,walls=falses(p))
  dx = 1/size(X,1)
  a,b = ones(size(X,1)),ones(size(X,1))
  computekernel!(K,X,u,v,ϵ,tol)
  while maximum(K)==Inf  # to make sure there is no +∞ in the kernel
    maximum([u v])==Inf ? error(): println("ϵ too small thus doubled")
    ϵ = 2*ϵ
    computekernel!(K,X,u,v,ϵ,tol)
  end
  for i = 1:niter
      a = proxdiv1(K*b*dx,u,ϵ,p)
      b = proxdiv2(K'*a*dx,v,ϵ,τ,walls)
  end
  @assert maximum(abs([a;b])) < Inf # numerical overflow?
  u, v = u + ϵ*map(log,a), v + ϵ*map(log,b) # absorb
  computekernel!(K,X,u,v,ϵ,tol)
  return K,u,v
end # scalingiter

"""
Performs one step of the flow, including ϵ scaling and multiresolution
the values of ϵ start from 1 and is divided by epsstep each step until ϵ
the multiresolution starts for problems bigger than 4096
"""
function onestepflow(p0,τ,dim,ϵ;niter=300, walls=falses(p), epsstep=6., tol=0.)

    N0 = map(Int,length(p0)^(1/dim))
    neps = map(Int,max(2,ceil(-log(ϵ)/log(epsstep)))) # number of changes of ϵ
    epsvec = logspace(0.,log10(ϵ),neps)
    nscales = clamp(map(Int, ceil(log2(length(p0)/4096)/dim)),0,neps-1)
    @assert mod(N0,2^nscales) == 0.
    scales = nscales-[zeros(Int,neps-nscales) ; 1:nscales]

    p = coarser(p0,scales[1],dim)
    w = coarser(walls,scales[1],dim).>=1
    u,v= zeros(p), zeros(p)
    X = coordvect(div(N0,2^scales[1]),dim)
    K = kernelinit(X,τ,dim)
    for k=1:neps
      if k>1 && scales[k]<scales[k-1]
        u,v   = finer(u,1,dim), finer(v,1,dim) # RENAME refine
        K     = finerplan(K,dim)               # RENAME refine_plan_support
        p     = coarser(p0,scales[k],dim)
        w     = coarser(walls,scales[k],dim).>=1
        X     = coordvect(div(N0,2^scales[k]),dim)
      end
      (K,u,v) = scalingiter(p, τ, X, K, u, v, epsvec[k], niter;tol=tol,walls=w)
    end
    q  = min(sum(K,1)[:]/length(p0)/(1-2τ),1)
    pr = (2τ-1+exp(-v))/2τ
  return q, pr
end

function flow(p0,τ,dim,ϵ, nstep;niter=300, walls=falses(p), epsstep=6., tol=0.)
  flow, pr = fill(p0, nstep+1), fill(p0, nstep+1)
  pr[1]=zeros(p0)

  for k = 1:nstep
    (flow[k+1],pr[k+1])  = onestepflow(flow[k],τ,dim,ϵ;niter=niter,
                                      walls=walls, epsstep=epsstep, tol=tol)
    print("k = ",k,"/$(nstep)")
  end
  return flow, pr
end

end #module
