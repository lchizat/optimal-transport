# Scaling algorithm and and application to optimal transport and generalizations
# @ Lénaïc Chizat 2015
module sa

export  define_prox,
        scalingAlgo, simple_unbalancedOT,
        refine_1d,  example1d

# Content:
# I.  Definitions of some numeric functions and other useful functions
# II. Scaling algorithm
# III.Classical and unbalanced optimal transpotr
# IV. Barycenters
# V.  Gradient flows

#########################################################################################
## I. Definitions of some numeric functions #############################################
#########################################################################################

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

"Return a*b with the convention 0*Inf = 0"
function axb0(a,b) # non commutative
  if a == 0 && isinf(b)
    return zero(a)
  else
    return a*b
  end
end

@vectorize_2arg Number div0
@vectorize_2arg Number axb0

function fdiv(φ,x,p,dx)
  if φ[:type] == "OT" || (in(φ[:type],["KL","TV"]) && φ[:param] == Inf)
    return sum( axb0(dx,exp(abs(x-p))-1) ) # barrier approx
  elseif φ[:type] == "KL"
    λ = φ[:param]
    return  λ*sum( axb0(dx,axb0(x,log(div0(x,p))) - x + p ))
  elseif φ[:type] == "TV"
    λ = φ[:param]
    return λ*sum( dx.*abs(x-p) )
  elseif φ[:type] == "RG"
    β1,β2 = φ[:param]
    @assert 0 <= β1 <= β2 < Inf
    return sum( axb0(dx, exp(max(0,β1*p-x))+ exp(max(0,x-β2*p))-2)) #barrier
  else
    error("Type of φ not recognized")
  end
end

function fdivstar(φ,u,p,dx)
  if φ[:type] == "OT" || (in(φ[:type],["KL","TV"]) && φ[:param] == Inf)
    return sum( axb0(p.*dx,u) )
  elseif φ[:type] == "KL"
    λ = φ[:param]
    return λ*sum( axb0(p.*dx,exp(u/λ)-1) )
  elseif φ[:type] == "TV"
    λ = φ[:param]
    return λ*sum( dx.*min(p, max(-p,axb0(p,u/λ))) )
  elseif φ[:type] == "RG"
    β1,β2 = φ[:param]
    @assert 0 <= β1 <= β2 < Inf
    return sum( axb0(dx, max(β1*axb0(p,u), β2*axb0(p,u)) ))
  else
    error("Type of φ not recognized")
  end
end


function proxdiv(φ,p,s,u,ϵ)
  if φ[:type] == "OT" || (in(φ[:type],["KL","TV"]) && φ[:param] == Inf)
    return div0(p,s)
  elseif φ[:type] == "KL"
    λ = φ[:param]
    return div0(p,s.*exp(u/λ)).^(λ/(λ+ϵ))
  elseif φ[:type] == "TV"
    λ = φ[:param]
    return !isinf(u).*min( exp((λ-u)/ϵ), max(exp((-λ-u)/ϵ), div0(p,s)) )
  elseif φ[:type] == "RG"
    β1,β2 = φ[:param]
    @assert 0 <= β1 <= β2 < Inf
    return !isinf(u).*min(β2*div0(p,s), max(β1*div0(p,s),exp(-u/ϵ)))
  else
    error("Type of φ not recognized")
  end
end
"""
Computes the proxdiv operator, with stabilization
INPUTS:
- F ∈  {"OT" "KL" "TV" "RG"
        "BaryOT" "BaryKL" "BaryTV" "BaryRG"
        "GFheleshaw" "GFcrowd"}
- s,u,ϵ : arguments of proxdiv
- p ref measure for some divergences
- α = table of weights (for barycenters)
- args : parameters of the divergence and more...

OUTPUT:
- updates p as a side effect (barycenter, next gradient flow...)
- returns the proxdiv (same shape as s/u)
"""
function proxdiv(F,s,u,ϵ,p,α,args...)

  if F == "OT"
    return div0(p,s)

  elseif F == "BaryOT"
    p[:,1] = Float64[baryproxOT(s[i,:],u[i,:],ϵ,α) for i=1:size(s,1)][:]
    for k = 2:size(p,2);  p[:,k] = p[:,1] end # not necessary
    return div0(p,s)

  elseif F == "KL"
    λ= args[1]
    ss = similar(p)
    for k=1:size(p,2)
      ss[:,k] = div0(p[:,k],s[:,k].*exp(u[:,k]/(λ*α[k]))).^(λ*α[k]/(λ*α[k]+ϵ)) #alpha adapted
    end
    return ss #

  elseif F=="BaryKL"
    λ= args[1]
    p[:,1] =  Float64[baryproxKL(s[i,:],u[i,:],ϵ,α,λ) for i = 1:size(s,1)][:]
    for k = 2:size(p,2);  p[:,k] = p[:,1] end # not necessary
    return div0(p,s.*exp(u./λ)).^(λ./(λ+ϵ)) #alpha adapted

  elseif F == "TV"
    λ = args[1]
    return !isinf(u).*min( exp((λ-u)/ϵ), max(exp((-λ-u)/ϵ), div0(p,s)) )
  elseif F=="BaryTV"
    λ = args[1]
    p[:,1] = Float64[baryproxTV(s[i,:],u[i,:],ϵ,α,λ) for i=1:size(s,1)][:]
    for k = 2:size(p,2);  p[:,k] = p[:,1] end # not necessary
    return !isinf(u).*min( exp((λ-u)/ϵ), max(exp((-λ-u)/ϵ), div0(p,s)) )

  elseif F == "RG"
    β1, β2 = args[1], args[2]
    @assert 0 <= β1 <= β2 < Inf
    return !isinf(u).*min(β2*div0(p,s), max(β1*div0(p,s),exp(-u/ϵ)))

  elseif F=="BaryRG"
    β1, β2 = args[1], args[2]
    @assert 0 <= β1 <= β2 < Inf
    p[:,1] = Float64[baryproxRG(s[i,:],u[i,:],ϵ,α,β1,β2) for i=1:size(s,1)][:]
    for k = 2:size(p,2);  p[:,k] = p[:,1] end # not necessary
    return !isinf(u).*min(β2*div0(p,s), max(β1*div0(p,s),exp(-u/ϵ)))

  elseif F=="GFheleshaw"
    τ, V = args[1], args[2]
    κ = 1-2*τ*V
    I = (s .<= (exp(u)*κ^(1+ϵ)).^(1/ϵ))
    ss = div0(ones(s),(s.*exp(u)).^(1/(1+ϵ)))
    ss[I]=div0(ones(s),(exp(u)*κ).^(1/ϵ))[I]
    return ss

  elseif F=="GFcrowd"
    τ, V = args[1], args[2]
    z = (s.!=0).*exp(-(u+2*τ*V)/ϵ)
    ss = axb0(s,z)
    I = ss.>1.
    z[I]=div0(z[I],ss[I])
    return z
  end
end

# functions for the barycenter esptimate (we assume s=0 => u finite)
baryproxOT(s,u,ϵ,α) = (exp(sum(α[:].*(log(s[:])-u[:]/ϵ))/sum(α)))

#baryproxKL(s,u,ϵ,α,λ) = (sum(α[:].*exp(-u[:]/(ϵ+λ)).*(s[:].^(ϵ/(λ+ϵ))))/sum(α))^((ϵ+λ)/ϵ)
baryproxKL(s,u,ϵ,α,λ) = (sum(α[:].*div0(s[:].^ϵ,exp(u[:])).^(1/(ϵ+λ)))/sum(α))^((ϵ+λ)/ϵ)

implicitbaryTV(logh,s,u,α,ϵ,λ,I) = sum(α[!I]) + sum(α[I].*max(-1,min(1,(logh+u[I]/ϵ-log(s[I]))*ϵ/λ  )))

implicitbaryRG(logh,s,u,α,ϵ,β1,β2) = sum(α.*(β2*min(0,logh+u/ϵ-log(s/β2))+β1*max(0,logh+u/ϵ-log(s/β1))))

function baryproxTV(s,u,ϵ,α,λ)
    I = s[:].>0
    α0 = sum(α[!I])
    if 2*α0 < sum(α)
        nodes = sort(union( log(s[I])-u[I]/ϵ+λ/ϵ,log(s[I]) - u[I]/ϵ - λ/ϵ ))
        ind = 2
        while  implicitbaryTV(nodes[ind],s[:],u[:],α[:],ϵ,λ,I) < 0.
          ind+=1
        end
        v1, v2 = implicitbaryTV(nodes[ind-1],s[:],u[:],α[:],ϵ,λ,I), implicitbaryTV(nodes[ind],s[:],u[:],α[:],ϵ,λ,I)
        return exp(nodes[ind]-v2*(nodes[ind-1]-nodes[ind])/(v1-v2))
    else
      return 0.
    end
end

function baryproxRG(s,u,ϵ,α,β1,β2)
    if all(s.>0)
        nodes = sort(union( log(s[:])-u[:]/ϵ-log(β1), log(s[:]) - u[:]/ϵ - log(β2) ))
        ind = 2
        while  implicitbaryRG(nodes[ind],s[:],u[:],α[:],ϵ,β1,β2) < 0.
          ind+=1
        end
        v1, v2 = implicitbaryRG(nodes[ind-1],s[:],u[:],α[:],ϵ,β1,β2), implicitbaryRG(nodes[ind],s[:],u[:],α[:],ϵ,β1,β2)
        return exp(nodes[ind]-v2*(nodes[ind-1]-nodes[ind])/(v1-v2))
    else
      return 0.
    end
end

function extract_fonction(Function,n,N)
  F = Function[:type]
  p = (F in ["OT"; "KL"; "TV"; "RG"] ? Function[:ref] : zeros(N,n) )
  α = (haskey(Function,:weights) ? Function[:weights][:] : ones(n) )
  args = Function[:args]
  return F::String, p::Array{Float64}, α::Array{Float64,1}, args
end
#########################################################################################
## II. Scaling Algorithm ################################################################
#########################################################################################

"""
Scaling Algorithm (so far : same cost function for all couplings)

INPUTS :
proxdiv(a,u,ϵ) -> s    for size(s) = size(a) = size(u)
X, Y                   tuple of array with the coordinates of their elements, dimension
        ex : X = ([0:.1:1],[0:.1:1]') or X = ([0:.1:1],)
c(x,y): X×Y ⇾ R ∪ {∞}  function with input x and y (tuples)
the number of couplings n is determined from the number of function in c

OUTPUT :
K       optimal (array of) plan
u,v     optimal dual variables

OPTIONAL :
dx, dy  ref measure vectors or scalars
niter   nb of iterations
epsvec  decreasing vector of values of ϵ
K0      guessed sparcity of the solution (only the structure of K0 is considered)
"""

function scalingAlgo(n::Int, Function1, Function2, c, X, dx, Y, dy, niter, epsvec;
                          tol = 0.,  # threshold on the plan
                          u = zeros(length(X[1]),n), v = zeros(length(Y[1]),n),
                          spind = falses(n), # input matrices of sparsity
                          spK::Array{SparseMatrixCSC} = Array{SparseMatrixCSC}(n))
# Preliminaries
  @assert length(X)==length(Y)<3 # only dim=1 or dim=2 supported
  carray, dK = Array{Float64}, Array{Array{Float64}}(n) #dense kernel
  !all(spind) ? carray = broadcast(c,X...,Y...) : nothing # pre-compute the cost matrix

# initializations
  Nx,Ny = length(X[1]), length(Y[1])
  F1, p, α1, args1 = extract_fonction(Function1,n,Nx)
  F2, q, α2, args2 = extract_fonction(Function2,n,Ny)
  a, b = ones(Nx,n), ones(Ny,n)
  i,epsind = 1, 1
  ϵ = epsvec[epsind]
  updateK!(spK,dK,spind,u,v,X,Y,ϵ,c,carray,tol)

  while i < niter
    # scaling iterations
    for k = 1:n;      a[:,k] = (spind[k]? spK[k]:dK[k])*(b[:,k].*dy);     end
    a = proxdiv(F1,a,u,ϵ,p,α1,args1...)
    for k = 1:n;      b[:,k] = (spind[k]? spK[k]:dK[k])'*(a[:,k].*dx);    end
    b = proxdiv(F2,b,v,ϵ,q,α2,args2...)

    # stabilizations
    if maximum(abs([a;b]))>1e100 || (i/niter) > (epsind+1)/length(epsvec) || i ==niter-1
      u, v = u + ϵ*map(log,a), v + ϵ*map(log,b) # absorb
      (i/niter) > (epsind+1)/length(epsvec)? (epsind += 1;print(epsind) ): nothing
      ϵ = epsvec[epsind]
      updateK!(spK,dK,spind,u,v,X,Y,ϵ,c,carray,tol)
      a, b = ones(Nx,n), ones(Ny,n)
    end
    i += 1
  end

  K = Array{Any}(n)
  for k=1:n; K[k] = (spind[k]? spK[k] : dK[k] ); end
  return K, p, q
end # scalingAlgo


function updatespK!(K::SparseMatrixCSC, X::Tuple{AbstractArray}, Y, u, v, ϵ, c,tol)
  nzvalK = K.nzval
  rows = K.rowval
  for col = 1:K.n # efficient iterations
    for i in nzrange(K, col)
      row = rows[i]
      nzvalK[i] = exp((u[row]+v[col]-c(X[1][row],Y[1][col]))/ϵ)
      nzvalK[i] .<tol? nzvalK[i]=0: nothing
    end
  end
  SparseMatrix.dropzeros!(K)
  return Void
end

function updatespK!(K::SparseMatrixCSC, X::Tuple{AbstractArray,AbstractArray}, Y, u, v, ϵ, c,tol)
  nzvalK = K.nzval
  rows = K.rowval
  for col = 1:K.n # efficient iterations
    for i in nzrange(K, col)
      row = rows[i]
      nzvalK[i] = exp((u[row]+v[col]-c(X[1][row],X[2][row],Y[1][col],Y[2][col]))/ϵ)
      nzvalK[i] .<tol? nzvalK[i]=0: nothing
    end
  end
  SparseMatrix.dropzeros!(K)
  return Void
end

"Update the active kernel (dense or sparse depending on spind)"
function updateK!(spK,dK,spind,u,v,X,Y,ϵ,c,carray,tol)
  for k = 1:length(spind)
    if spind[k]
      updatespK!(spK[k], X, Y, u[:,k], v[:,k], ϵ, c,tol)
    else
      dK[k] = exp(broadcast(+,-carray, u[:,k], v[:,k]')/ϵ)
      dK[k][dK[k].<tol]=0.
      if countnz(dK[k])/length(dK[k]) < .25  # convert to sparse
        spind[k] = true
        spK[k]   = sparse(dK[k])
        dK[k]    = [0.]
      end
    end
  end
  return Void
end

#########################################################################################
## III. Classical and unbalanced optimal transport ######################################
#########################################################################################
"""
Simple function for displaying the convergence
Not optimized, very slow;
"""
function simple_unbalancedOT( p, q, φ1, φ2, c, ϵ, niter)
  F1(x,dx) = fdiv(φ1,x,p,dx)
  F1c(u,dx)= fdivstar(φ1,u,p,dx)
  F2(x,dx) = fdiv(φ2,x,q,dx)
  F2c(u,dx)= fdivstar(φ2,u,q,dx)

  I,J = size(c)
  dx,dy = ones(I)/I, ones(J)/J
  a, b = ones(I), ones(J)
  K = exp(-c/ϵ)
  pdgap = zeros(niter)

  for i = 1:niter
    Kb = K*(b.*dy)
    a  = proxdiv(φ1,p,Kb,0.,ϵ)
    Ka = K'*(a.*dx)
    b  = proxdiv(φ2,q,Ka,0.,ϵ)#

    R = broadcast(.*,broadcast(.*,K,a),b')
    primal = F1(R*dy,dx) + F2(R'*dx,dy) + ϵ*sum( axb0(R,log(div0(R,K))) - R + K )/(I*J)
    dual   = - F1c(-ϵ*log(a),dx) - F2c(-ϵ*log(b),dy) - ϵ*sum(R-K)/(I*J)
    pdgap[i] = primal-dual

  end
  R = broadcast(.*,broadcast(.*,K,a),b')
  return R, pdgap
end # simple_unbalancedOT

###############
function tumor_step(p, τ, λ, c, X, Y,n;
                          dx = 1/length(X[1]), dy = 1/length(Y[1]),
                          niter = 1e3,
                          epsvec= logspace(0,-5,20),
                          tol = 0.,
                          u = zeros(length(X[1]),n), v = zeros(length(Y[1]),n),
                          spind = falses(n), # input matrices of sparsity
                          spK = Array{SparseMatrixCSC}(n))

  @assert length(X)==length(Y)<3 # only dim=1 or dim=2 supported
  carray = Array{Float64}
  !all(spind) ? carray = broadcast(c,X...,Y...) : nothing # pre-compute the cost matrix
  dK = Array{Array{Float64}}(n) #dense kernel
  Nx,Ny = length(X[1]), length(Y[1])
  a, b = ones(Nx,n), ones(Ny,n)
  i,epsind = 1, 1
  κ = 1-2*τ*λ
  ϵ = epsvec[epsind]
  updateK!(spK,dK,spind,u,v,X,Y,ϵ,c,carray,tol)

  while i < niter
    Kb = spK[1]*b
    a = div0(p,exp(u).*Kb).^(1/(1+ϵ))
    Ka = spK[1]'*a
    b = proxdiv_tumor(Ka,v,τ,λ,ϵ)

    if maximum(abs([a;b]))>1e40 || (i/niter) > (epsind+1)/length(epsvec) || i ==niter-1
      u, v = u + ϵ*map(log,a), v + ϵ*map(log,b) # absorb
      (i/niter) > (epsind+1)/length(epsvec)? epsind += 1 : nothing
      ϵ = epsvec[epsind]
      updateK!(spK,dK,spind,u,v,X,Y,ϵ,c,carray,tol)
      a, b = ones(Nx,n), ones(Ny,n)
      #perc = (i/niter)*100; print("\rProgress: $(perc)%");
    end
    i += 1
  end

  return min(1, sum(spK[1],1)/κ)[:], v[:]
end

function proxdiv_tumor(b,v,τ,λ,ϵ)
      κ = 1-2*τ*λ
      I = (b .<= (exp(v)*κ^(1+ϵ)).^(1/ϵ))
      b = div0(ones(b),(b.*exp(v)).^(1/(1+ϵ)))
      b[I]=div0(ones(b),(exp(v)*κ).^(1/ϵ))[I]
  return b
end


function tumor_flow(p0, τ, λ, c, X,Y,nstep, K0;
                  niter = 1e3, epsvec = logspace(0,-5,20))
  flow = fill(p0, nstep+1)
  pot  = fill(p0, nstep+1)
  for t = 1:nstep
    (flow[t+1], pot[t+1]) = tumor_step(flow[t], τ, λ, c, X, Y,1;
                          niter = niter,
                          epsvec= epsvec, spind = [true], spK = fill(copy(K0),1))
  @show t
  end
  return flow, pot
end



###########################################################################
function tumor2_step(p, τ, α, c, X, Y,n;
                          dx = 1/length(X[1]), dy = 1/length(Y[1]),
                          niter = 1e3,
                          epsvec= logspace(0,-5,20),
                          tol = 0.,
                          u = zeros(length(X[1]),n), v = zeros(length(Y[1]),n),
                          spind = falses(n), # input matrices of sparsity
                          spK = Array{SparseMatrixCSC}(n))

  @assert length(X)==length(Y)<3 # only dim=1 or dim=2 supported
  carray = Array{Float64}
  !all(spind) ? carray = broadcast(c,X...,Y...) : nothing # pre-compute the cost matrix
  dK = Array{Array{Float64}}(n) #dense kernel
  Nx,Ny = length(X[1]), length(Y[1])
  a, b = ones(Nx,n), ones(Ny,n)
  i,epsind = 1, 1
  ϵ = epsvec[epsind]
  updateK!(spK,dK,spind,u,v,X,Y,ϵ,c,carray,tol)
  q = copy(p)

  while i < niter
   for k = 1:n
      spind[k] ? a[:,k] = spK[k]*(sub(b,:,k).*dy): a[:,k] = dK[k]*(sub(b,:,k).*dy)
   end
    a = div0(p,exp(u).*a).^(1/(1+ϵ))

    for k = 1:n
      spind[k] ? b[:,k] = spK[k]'*(sub(a,:,k).*dx): b[:,k] = dK[k]'*(sub(a,:,k).*dx)
    end
    b = proxdiv_tumor2(b,v,q,τ,α,ϵ)

    if maximum(abs([a;b]))>1e40 || (i/niter) > (epsind+1)/length(epsvec) || i ==niter-1
      u, v = u + ϵ*map(log,a), v + ϵ*map(log,b) # absorb
      (i/niter) > (epsind+1)/length(epsvec)? epsind += 1 : nothing
      ϵ = epsvec[epsind]
      updateK!(spK,dK,spind,u,v,X,Y,ϵ,c,carray,tol)
      a, b = ones(Nx,n), ones(Ny,n)
      #perc = (i/niter)*100; print("\rProgress: $(perc)%");
    end
    i += 1
  end

  return q, v
end

function proxdiv_tumor2(b,v,q,τ,α,ϵ)
      I    = (v[:,1].< v[:,2]) & (b[:,1].>0)
      l    = -v[:,2]/(1+ϵ) + log(b[:,2] + axb0(b[:,1],exp((v[:,2]-v[:,1])/ϵ)))*ϵ/(1+ϵ)
      l[I] = -v[I,1]/(1+ϵ) + log(b[I,1] + axb0(b[I,2],exp((v[I,1]-v[I,2])/ϵ)))*ϵ/(1+ϵ)
     β = max(l,log(1-2*α*τ))
      q[:] =  axb0(b,exp(-broadcast(+,v,(1+ϵ)*β)/ϵ))[:]
  return (b.!=0.).*exp(-broadcast(+,v,β)/ϵ)
end


function tumor2_flow(p0, τ, α, c, X,Y,nstep, K0;
                  niter = 1e3, epsvec = logspace(0,-5,20))
  flow = fill(p0, nstep+1)
  pot  = fill(p0, nstep+1)
  spK = cell(2)
  for t = 1:nstep
    spK[1]=copy(K0)
    spK[2]=copy(K0)
    (flow[t+1], pot[t+1]) = tumor2_step(flow[t], τ, α, c, X, Y,2;
                          niter = niter,
                          epsvec= epsvec, spind = [true true], spK = spK)

  @show t
  end
  return flow, pot
end


###########################################################################
########### EXAMPLES OF DENSITIES
###########################################################################

inseg(x,s) = ( s[1]<= x <= s[2] ? 1.0 : 0.0 )

function example1d(X,Y)
    fp = x -> 2*inseg(x,[.0 .2]) +
        40*(x-.9)*inseg(x,[.9 .95])+
        40*(1.-x)*inseg(x,[.95 1.])
    fq = x -> 10*(x-.2)*inseg(x,[.2 .4])+
        1.3*sqrt(1-(x-.7).^2/.04+0*im)*inseg(x,[.5 .9])
    p = Float64[fp(x) for x in X]
  p[1], p[end] = 0, 0
    q = Float64[fq(y) for y in Y]
    return p,q*sum(p)/sum(q)
end

function example2d(X1,X2)
    fp = (x,y) -> 1*(x>=.7)*(y<=.3)*(x-.7<=y) +
                1*(norm([x;y])<=.45) +
                1*(norm([x;y-1.])<=.25)
    fq = (x,y) -> 1*(x+y>=1.6) +
                1*(norm([x-.57,y-.15])<=.118) +
                1*(x+y>=.8)*(x+y<=1.1)*(y<=x+.5)*(y>=x+.2)
    p = Float64[fp(x1,x2) for x1 in X1, x2 in X2]
    q = Float64[fq(x1,x2) for x1 in X1, x2 in X2]
    return p,q*sum(p)/sum(q)
end

function example2dtumor(X1,X2)
    fp = (x,y) -> x^2*(x>=.7)*(y<=.3)*(x-.7<=y) +
                y^2*(norm([x;y])<=.45) +
                1*(norm([x;y-1.])<=.25)
    p = Float64[fp(x1,x2) for x1 in X1, x2 in X2]
    return p
end


function example2dcrowd(X1,X2)
    a = .05
    #f1 = (x,y) -> augmenter la longueur des portes
    fp = (x,y) -> 1.*(x.>=0.8)*(y.>=0.8)
    fq = (x,y) -> 1.*(x.>=.6)*(y.>=.1)*(y.<=.2)
    fV1 = (x,y) -> (x.<=0.48)*sqrt(x^2+y^2) + (x.>0.48)*(x.<=0.52)*sqrt(x^2+y^2) +(x.>0.52)*(sqrt((x-.5)^2+(y-.5)^2)+sqrt(2)/2)+ Inf*((.48<=x<=0.52)&&((y>.5+a)||(y<.5-a)))
    fV2 = (x,y) -> (x.<=0.48)*sqrt(x^2+(1-y)^2) + (x.>0.48)*(x.<=0.52)*sqrt(x^2+(1-y)^2) +(x.>0.52)*(sqrt((x-.5)^2+(y-.5)^2)+sqrt(2)/2)+ Inf*((.48<=x<=0.52)&&(y>.5+a||y<.5-a))
    p = Float64[fp(x1,x2) for x1 in X1, x2 in X2]
    q = Float64[fq(x1,x2) for x1 in X1, x2 in X2]
    V1 = Float64[fV1(x1,x2) for x1 in X1, x2 in X2]
    V2 = Float64[fV2(x1,x2) for x1 in X1, x2 in X2]
    return p, q, V1, V2
end

function example2dcrowd2(X1,X2)
    fp = (x,y) -> 1.*(rem(9.*x,2).>1)*(rem(9.*y,2).>1)
    fV1 = (x,y)-> 3*sqrt((x-0.5)^2+y^2)+exp(-((x-0.5)^2+(y-.5)^2)/.01)
    p = Float64[fp(x1,x2) for x1 in X1, x2 in X2]
    V1 = Float64[fV1(x1,x2) for x1 in X1, x2 in X2]
    return p, V1
end


end # module
