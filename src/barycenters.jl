# Barycenters for (unbalanced) optimal tranport
# @ Lénaïc Chizat 2016

#module gradientflow

#export onestepflow

################################################################################
## I. Definitions of some numeric functions ####################################
################################################################################

"Special x/y with convention x/0=0 if x≧0"
function div0(x,y)
  if x>=0.0 && y>0.0
    return x/y
  elseif (x >= 0.0 && y==0)
    return zero(x)
  else
    return oftype(x,Inf)
  end
end

proxdivOT!(p,s,u,ϵ)   = (s .= div0.(p,s))
proxdivKL!(p,s,u,ϵ,λ) = (s .= !isinf(u).*div0.(p,s.*exp(u/λ)).^(λ/(λ+ϵ)))

baryOT!(h,s,u,ϵ,α)   = (h .= exp((log(s)-!isinf(u).*u/ϵ)*α))
baryKL!(h,s,u,ϵ,α,λ) = (h .= (((s.^(ϵ/(ϵ+λ)).*exp(-(!isinf(u).*u)/(ϵ+λ))))*α).^((ϵ+λ)/ϵ))

function proxdivbaryOT!(h,s,u,ϵ,α)
  baryOT!(h,s,u,ϵ,α)
  proxdivOT!(h,s,u,ϵ)
end

function proxdivbaryKL!(h,s,u,ϵ,α,λ)
  baryKL!(h,s,u,ϵ,α,λ)
  proxdivKL!(h,s,u,ϵ,λ)
end

"Compute the kernel given the potentials"
function computekernel!(K::SparseMatrixCSC, X::AbstractArray, u, v, ϵ)
  nzvalK = K.nzval
  rows = K.rowval
  for col = 1:K.n # efficient iterations
    for i in nzrange(K, col)
      row = rows[i]
      nzvalK[i] = exp((u[row]+v[col]-sum((X[row,:]-X[col,:]).^2))/ϵ)
    end
  end
  SparseMatrix.dropzeros!(K)
end

function truncatekernel!(K::SparseMatrixCSC,tol)
  nzvalK = K.nzval
  rows = K.rowval
  for col = 1:K.n # efficient iterations
    for i in nzrange(K, col)
      if nzvalK[i] < tol
        nzvalK[i] = zero(nzvalK[i])
      end
    end
  end
  SparseMatrix.dropzeros!(K)
end

"Initialize the sparse kernel support"
function kernelinit(X,maxcost,dim)
  if dim==1
    sparse(map(Float64,broadcast((x,y)->(y-x)^2, X, X').<maxcost))
  elseif dim==2
    sparse(map(Float64,broadcast((x1,x2,y1,y2)->(y1-x1)^2+(y2-x2)^2,
              X[:,1],X[:,2],X[:,1]',X[:,2]').<maxcost))
  else
    error("unknown dim in kernelinit")
  end
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

coarser(p,n,dim) = (dim==1? coarser1d(p,n) : coarser2d(p,n))
finer(p,n,dim)   = (dim==1? finer1d(p,n)   : finer2d(p,n))
finerplan(K,dim) = (dim==1? finerplan1d(K) : finerplan2d(K))

function coarser1d(p0,n)
  p = zeros(div(size(p0,1),2^n),size(p0,2))
  for k=1:size(p0,2)
    for i=1:size(p,1)
      p[i,k] = mean(p0[1+(i-1)*2^n:i*2^n,k])
    end
  end
  return p
end

function coarser2d(p0,n)
  N0 = map(Int,sqrt(size(p0,1)))
  N  = div(N0,2^n)
  P0 = reshape(p0,N0,N0,size(p0,2))
  P  = zeros(N,N,size(p0,2))
  for k = 1:size(p0,2)
    for j = 1:N
      for i = 1:N
        P[i,j,k] = mean(P0[1+(i-1)*2^n:i*2^n,1+(j-1)*2^n:j*2^n,k])
      end
    end
  end
  return reshape(P,N^2,size(p0,2))
end

function finer1d(p0,n)
  p = zeros(size(p0,1)*2^n,size(p0,2))
  for k=1:size(p0,2)
    for i=1:size(p0,1)
      p[i,k] = p0[div(i-1,2^n)+1,k]
    end
  end
  return p
end

function finer2d(p0,n)
  N0 = map(Int,sqrt(size(p0,1)))
  N  = N0*2^n
  P0 = reshape(p0,N0,N0,size(p0,2))
  P  = zeros(N,N,size(p0,2))
  for k=1:size(p0,2)
    for i = 1:N
      for j = 1:N
        P[i,j,k] = P0[div(i-1,2^n)+1,div(j-1,2^n)+1,k]
      end
    end
  end
  return reshape(P,N^2,size(p0,2))
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
  return sparse(I,J,1.0,4N0^2,4N0^2)
end

################################################################################
## II. Scaling algorithm and multiresolution ###################################
################################################################################

"Scaling iterations"
function scalingiter(p, λ, α, X, K, u, v, ϵ, niter,tol)
  dx = 1/size(X,1)
  n = size(p,2)
  a,b = ones(p),ones(p)
  h = ones(size(p,1))
  for k=1:n
    computekernel!(K[k],X,u[:,k],v[:,k],ϵ)
  end
  while maximum(maximum(K[k]) for k=1:n)==Inf  # to make sure there is no +∞ in the kernel
    maximum([u v])==Inf ? error(): println("ϵ too small thus doubled")
    ϵ = 2*ϵ
    for k=1:n
      computekernel!(K[k],X,u[:,k],v[:,k],ϵ)
    end
  end
  for i = 1:niter
    #@show i
      for k=1:n
        a[:,k] = K[k]*(b[:,k]*dx)
      end
      λ == Inf ? proxdivOT!(p,a,u,ϵ) : proxdivKL!(p,a,u,ϵ,λ)
      for k=1:n
        b[:,k] = K[k]'*(a[:,k]*dx)
      end
      λ == Inf ? proxdivbaryOT!(h,b,v,ϵ,α) : proxdivbaryKL!(h,b,v,ϵ,α,λ)
      if maximum(abs([a;b])) >1e10 # numerical overflow?
        u, v = u + ϵ*log(a), v + ϵ*log(b)
        for k=1:n
          computekernel!(K[k],X,u[:,k],v[:,k],ϵ)
        end
      end
  end
  @assert maximum(abs([a;b])) < Inf # numerical overflow?
  u, v = u + ϵ*log(a), v + ϵ*log(b)
  m = maximum(maximum(Kk) for Kk in K)
  for k=1:n
    computekernel!(K[k],X,u[:,k],v[:,k],ϵ)
    truncatekernel!(K[k],m*tol)
  end
  return K,u,v,h
end # scalingiter


#########
function barycenter(p0,λ,α, dim,ϵ; niter=300,epsstep=6., tol=0.)
  n = size(p0,2)
  N0 = map(Int,size(p0,1)^(1/dim))

  X = coordvect(div(N0,2^4),dim)
  K = Array{SparseMatrixCSC,1}(n)
  p = coarser(p0,4,dim)
  for k=1:n
    K[k]   = kernelinit(X,2.,dim) # parameter!
  end
  u,v = zeros(p), zeros(p)
  h = zeros(size(p,1))                                                                                              #1
  #scales =    [4;    4;     4;  4;        3;     3;  3    ; 2    ; 2   ; 2    ; 2    ;  1  ;     1 ;     0;     0;      0]
  #epsvec =    [1.0; 0.1; 1e-2;  5e-3;  5e-3;  1e-3; 5e-4  ; 1e-3 ; 5e-4; 1e-4 ; 5e-5 ; 1e-4;   3e-5;  3e-5;  9e-6;   5e-6]
  #nitervec =  [500; 500;  500;  500 ;   500;   500; 500   ; 500  ; 500 ; 500  ; 500  ;  500;  500;   500;   500;    500]
  #tolvec  =   [0.0; 0.0;   0.0;  0.0;   0.0;   0.0; 0.0   ; 1e-15;1e-15;1e-15; 1e-15 ;1e-10 ; 1e-10; 1e-10; 1e-10; 1e-10] #; 1e-10; 1e-10; 1e-10]*1e-10
  scales =    [4;    4;     4;     4;     4;    4;    4;    3;     3;    3;     3;    3;     3;   3;     2;    2;    2;    2;    2;    2;     2;    1;    1;   1;   1;
     1;    1;    1;    1;    0;     0;    0;    0]#;    0]#;
  epsvec =    [1.0; 0.1; 1e-2;  5e-3;  1e-3; 5e-4; 1e-4; 5e-3;  1e-3; 5e-4;  1e-4; 7e-5;  3e-5; 1e-5; 1e-3; 5e-4; 1e-4; 5e-5; 1e-5; 5e-6;  2e-6; 1e-3; 5e-4; 1e-4; 5e-5; 1e-5; 6e-6; 3e-6; 9e-7;  1e-4; 5e-5; 1e-5; 5e-6]#; 1e-6]#;
  nitervec =  50*[5;  5;    5;      5;     5;    5;    5;    5;     5;    5;     5;    5;     5;    5;    5;    5;    5;    5;    5;     5;    5;    5;    5;    5; 5;      5;   5 ;     5;    5;    5;     5;    5;    5;    5]#;
  tolvec  =   zeros(length(scales))
  for k = 1:length(epsvec)
    if k>1 && scales[k]<scales[k-1]
      u,v   = finer(u,1,dim),finer(v,1,dim)
      p     = coarser(p0,scales[k],dim)
      X     = coordvect(div(N0,2^scales[k]),dim)
      for l = 1:n
        K[l]     = finerplan(K[l],dim)
      end
    end
    @show scales[k], epsvec[k], nitervec[k],tolvec[k]
    @time (K,u,v,h) = scalingiter(p, λ, α, X, K, u, v, epsvec[k], nitervec[k],tolvec[k])
  end
  h,K
end


#end #module
