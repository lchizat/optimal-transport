abstract Var{N} # (D,Z) on a N dimensional grid

immutable Cvar{N} <: Var{N}    # centered variable
  cs::NTuple{N,Integer}        # centered grid sizes
  ll::Array{Float64,1}         # size of time × space domain
  D::Array{Array{Float64,N},1} # density ρ and momentum ω
  Z::Array{Float64,N}          # source (nullable?)

  function Cvar{N}(cs::NTuple{N,Integer},ll,D,Z)
    @assert length(ll) == N
    @assert all(size(Dk)==cs for Dk in D)
    @assert size(Z) == cs
    new(cs,copy(ll),D,Z)
  end
end

immutable Svar{N} <: Var{N} # staggered variable
  cs::NTuple{N,Integer}
  ll::Array{Float64,1}
  D::Array{Array{Float64,N},1}
  Z::Array{Float64,N}

  function Svar{N}(cs::NTuple{N,Integer},ll,D,Z)
    @assert length(ll) == N
    @assert all(size(D[k])==tuple([cs...] + (1:N .== k)...) for k=1:N)
    @assert size(Z) == cs
    return new(cs,copy(ll),D,Z)
  end
end

immutable CSvar{N} # pair of centered/staggered variables
  cs::NTuple{N,Integer} # size of the centered grid
  ll::Array{Float64,1}  # length of the box
  U::Svar{N}
  V::Cvar{N}

  function CSvar(cs,ll,U,V)
    @assert V.cs == U.cs == cs
    @assert V.ll == V.ll == ll
    return new(cs,copy(ll),U,V)
  end
end

function CSvar(ρ_0, ρ_1, T, ll) # MAIN CONSTRUCTOR
  U  = Svar(ρ_0,ρ_1,T,ll)
  V  = interp(U)
  N  = length(U.cs)
  return CSvar{N}(U.cs,U.ll,U,V)
end

#########################################################
# Constructor using linear interpolation
##########################################################

function Svar(ρ_0::AbstractArray, ρ_1, T, ll)
  N = ndims(ρ_0)+1
  @assert length(ll) == N
  cs = tuple(T,size(ρ_0)...)
  f(t,r0,r1)= t*r1 + (1-t)*r0 # linear interpolation
  D = [zeros(tuple([cs...] + (1:N .== k)...)) for k=1:N]
  D[1] = broadcast(f,linspace(0,1,T+1),
        reshape(ρ_0,1,size(ρ_0)...),reshape(ρ_1,1,size(ρ_1)...))
  Z = zeros(cs)
  Svar{N}(cs,ll,D,Z)
end

function interp{N}(U::Svar{N})
    V = Cvar{N}(U.cs,copy(U.ll),[zeros(U.cs) for i=1:N],zeros(U.Z))
    interp!(V,U)
end

function interp!{N}(V::Cvar{N},U::Svar{N})
  for k = 1:N
    interp!(V.D[k],U.D[k],k)
  end
  V.Z .= U.Z
  V
end

function interp!{N}(V::Svar{N},U::Cvar{N}) # expand first
  for k = 1:N
    dk = [U.cs...]
    dk[k] = 1
    interp!(V.D[k],cat(k,zeros(dk...),U.D[k],zeros(dk...)),k)
  end
  V.Z .= U.Z
  V
end

function interp!(dest::AbstractArray,M,dim) # GENERAL TOOLS
  indk = ntuple(n->1*(n==dim),ndims(M))
  @assert size(M)==tuple([size(dest)...]+[indk...]...)
  Ik = CartesianIndex(indk)
  for I in CartesianRange(size(dest))
    dest[I] = (M[I]+M[I+Ik])/2
  end
  dest
end

function diff!(dest::AbstractArray,M,dim) # GENERAL TOOLS
  indk = ntuple(n->1*(n==dim),ndims(M))
  @assert size(M)==tuple([size(dest)...]+[indk...]...)
  Ik = CartesianIndex(indk)
  for I in CartesianRange(size(dest))
    dest[I] = M[I+Ik]-M[I]
  end
  dest
end

function Base.diff(M,dim)
  indk = ntuple(n->1*(n==dim),ndims(M))
  sdest = tuple([size(M)...]-[indk...]...)
  diff!(zeros(sdest),M,dim)
end

#########################################################
# Basic methods on Grids
##########################################################
"Project on the positivity constraint for the density"
function projpositive!(x::CSvar)
  projpositive!(x.U)
  projpositive!(x.V)
  x
end
projpositive!(A::Var) = (A.D[1] .= max(A.D[1], 0))

function Base.Broadcast.broadcast!{N}(f,y::Var{N},x::Var...)
    for k=1:N
      y.D[k] .= f.((z.D[k] for z in x)...)
    end
    y.Z .= f.((z.Z for z in x)...)
end

function Base.Broadcast.broadcast!(f,y::CSvar,x::CSvar...)
  broadcast!(f,y.U,(z.U for z in x)...)
  broadcast!(f,y.V,(z.V for z in x)...)
end

sum(x::Var) = sum(x.Z) + sum(sum(Dk) for Dk in x.D)
sum(x::CSvar) = sum(x.U) + sum(x.V)

# Pushforward by T: (t,x)-> (t,s*x) (preserves continuity eq)
function dilategrid!{N}(x::CSvar{N},s)
  x.ll[2:end] *= s
  dilategrid!(x.U,s)
  dilategrid!(x.V,s)
  x
end

function dilategrid!{N}(U::Var{N},s)
  U.ll[2:end] *= s
  U.D[1]      .= U.D[1]/s^(N-1)
  U.D[2:end]  .= U.D[2:end]/s^(N-2)
  U.Z         .= U.Z/s^(N-1)
  U
end

"Project on the boundary conditions"
function projBC!{N}(U::Svar{N},ρ_0,ρ_1)
  U.D[1][1,:]   = ρ_0[:]
  U.D[1][end,:] = ρ_1[:]
  for k = 2:N
    Rpre  = CartesianRange(U.cs[1:k-1])
    Rpost = CartesianRange(U.cs[k+1:end])
    _nullboundary!(U.D[k],Rpre, Rpost,U.cs[k]+1)
  end
end
projBC!(x::CSvar,ρ_0,ρ_1) = projBC!(x.U,ρ_0,ρ_1)

@noinline function _nullboundary!(M::AbstractArray,Rpre,Rpost,dd) # GENERAL TOOLS
  for Ipost in Rpost
    for Ipre in Rpre
      M[Ipre, 1, Ipost] = 0.0
      M[Ipre, dd, Ipost] = 0.0
    end
  end
end

# A = ∫∫ (1/p) |ω|^p/ρ^(p-1) + s^p (1/q) |ζ|^q/ρ^(q-1)
"Compute energy functional"
function energy{N}(V::Cvar{N},s,p,q)
  fp,fq  = zeros(V.cs),zeros(V.cs)
  ind    = V.D[1].>0
  q>=1? fp[ind] = (s^p/q) * (abs(V.Z[ind]).^q)./(V.D[1][ind].^(q-1)): nothing
  p>=1? fq[ind] = (sum(V.D[k][ind].^p for k=2:N))./V.D[1][ind].^(p-1) : nothing
  return sum(fp+fq)*prod(V.ll)/prod(V.cs)
end
energy(x::CSvar,s,p,q) = energy(x.V,s,p,q)

"Compute how much the continuity equation is disatisfied"
function remainderCE{N}(U::Svar{N})
  v = - U.Z
  for k=1:N
    v .= v + diff(U.D[k],k)*U.cs[k]/U.ll[k]
  end
  return v
end

"Compute how much the continuity equation is disatisfied"
distfromCE{N}(U::Svar{N}) = sum(remainderCE(U).^2)*prod(U.ll)/prod(U.cs)
distfromCE(x::CSvar) = distfromCE(x.U)

"Compute how much the interpolation constraint is disatisfied"
distfromInterp(x::CSvar) = L2dist(interp(x.U),x.V)

function L2dist(U::Var,V::Var)
  @assert U.ll == V.ll
  (sum(sum((U.D[k]-V.D[k]).^2) for k in 1:length(V.D)) + sum((U.Z-V.Z).^2))*prod(U.ll)/prod(U.cs)
end

#########################################################
## For post processing
########################################################
function speedandgrowth{N}(x::CSvar{N};maxratio=100)
  ind = x.V.D[1] .> maximum(x.V.D[1])/maxratio
  g = zeros(x.cs)
  g[ind] = x.V.Z[ind] ./ x.V.D[1][ind]
  v = [zeros(x.cs) for i=1:N-1]
  for k=1:N-1
    v[k][ind] .= x.V.D[k+1][ind] ./ x.V.D[1][ind]
  end
  return v,g
end
