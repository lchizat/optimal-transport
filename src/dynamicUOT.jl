module DynamicUOT
#=
Functions for dynamic optimal transport
=#
import
    Base: .+, Broadcast.broadcast, sum, diff

export
    computeGeodesic,
    speedandgrowth

include("grids.jl")

"Compute the highest real root of aX^3+bX^2+cX+d"
function root(a,b,c,d)
  p = -(b/a)^2/3 + c/a
  q = 2*(b/a)^3/27 - b*c/a^2/3 + d/a
  Δ = q.^2 + 4/27*p.^3
  if Δ > 0
    u = ∛((-q + sqrt(Δ))/2)
    v = ∛((-q - sqrt(Δ))/2)
    z = u + v - b/ a /3
  elseif Δ < 0
    u = ((-q + im*sqrt(-Δ))/2)^(1/3)
    z = real(u + conj(u) - b/ a /3)
  else
    z =  real(3*q ./ p - b/ a /3)
  end
  return z
end

"Finds a real root of f by Newton method"
function rootNewton(f,f′,x0 ,tol = 1e-15)
  fx = f(x0)
  while abs(fx) > tol
    x0 -= fx/f′(x0)
    fx = f(x0)
  end
  return x0
end

################################################
"Return prox_F(V) where F is the energy functional and V on centered grid"
function proxF!{N}(dest::Cvar{N},V::Cvar{N}, γ, p, q)
  if     p==1 && q < 1 # W1
    dest.D[1] .= V.D[1]
    proxA!(dest.D[2:end],V.D[2:end],γ)
  elseif p==2 && q < 1 # W2
    proxB!(dest.D[1],dest.D[2:end],V.D[1],V.D[2:end],γ)
  elseif p==1 && q==1 # ``Bounded Lipschitz''
    dest.D[1] .= V.D[1]
    proxA!(dest.D[2:end],V.D[2:end],γ)
    proxA!(dest.Z,V.Z,γ)
  elseif p==2 && q==2 # WF
    proxB!(dest.D[1],cat(1,dest.D[2:end],[dest.Z]),
          V.D[1],cat(1,V.D[2:end],[V.Z]),γ)
  elseif p==2 && q==1 # Partial W2
    proxB!(dest.D[1],dest.D[2:end],V.D[1],V.D[2:end],γ)
    proxA!(dest.Z,V.Z,γ)
  elseif p==1 && q==2 # W1-FR
    proxA!(dest.D[2:end],V.D[2:end],γ)
    proxB!(dest.D[1],dest.Z,V.D[1],V.Z,γ)
  else
    error("Functional not implemented")
  end
end

mdl(M::Array{Float64}) = abs(M) # module
mdl{N}(M::Array{Array{Float64,N},1}) = (sqrt(sum(M[k].^2 for k=1:length(M))))

function proxA!(dest, M, γ) # prox of sum abs(M_i)
  softth = max(1 - γ./mdl(M),0.0)
  if isa(M,Array{Float64})
    dest .= softth .* M
  else
    for k=1:length(M)
      dest[k] .= softth .* M[k]
    end
  end
end

function proxB!(destR,destM,R,M,γ) #prox of sum |M_i|^2/R_i
  a = 1.0
  b = 2γ - R
  c = γ^2 - 2γ*R
  d = -(γ/2)*mdl(M).^2 - γ^2*R
  destR .= max(0.0,root.(a,b,c,d))
  DD = zeros(R)
  DD[destR.>0] = 1.0 - γ./(γ + destR[destR.>0])
  if isa(M,Array{Float64})
    destM .= DD .* M
  else
    for k=1:length(M)
      destM[k] .= DD .* M[k]
    end
  end
end

"""
Solve Δu + f = 0 with Neumann BC on the staggered grid
-> use  DCT II / DCT III because x(-1/2)=x(N+1/2)=0
with the contraints: mean(u-f)=0 and 0 boundary cond.
if source=true: solves instead   Δu -u + f =0
plan /iplan are "prepared" DCT! (for speed)
"""
function poisson!(f::Array{Float64}, ll::Array{Float64,1},
                source::Bool, plan="none", iplan="none")
  d = ndims(f)
	N = [size(f)...]
	h = ll./N # sampling interval
  dims = ones(Int64,d)
	D = zeros(size(f)) # array of frequency multipliers

  for k = 1:d
    dims[:] = 1
    dims[k] = N[k]
    dep = Array(Float64,dims...) # multipliers along 1 dimension
    for i = 0:(length(dep)-1)
      dep[i+1] = (2*cos( π * i /N[k]) - 2.)/h[k]^2
    end
    broadcast!(+,D,D,dep)
	end
	source ? (D -= 1) :	(D[1] = 1)

	if isequal(plan,"none") | !isequal(iplan,"none")
			plan  = v -> FFTW.r2r!(v,FFTW.REDFT10,1:d) #DCT 10
			iplan = v -> FFTW.r2r!(v,FFTW.REDFT01,1:d) #iDCT 01
	end

  plan(f) #on place
  renorm = prod(N)*2^d
  for i =1:length(f)
    f[i] = - f[i]/D[i]/renorm
  end
  iplan(f)
end


"Projection on the continuity equation constraint"
function projCE!{N}(dest::Svar{N},U::Svar{N},ρ_0, ρ_1,source::Bool,
                            plan="none",iplan="none")
  @assert dest.ll == U.ll
  projBC!(U,ρ_0,ρ_1) # for exact algo: create a new Svar instead
  p = - remainderCE(U)
  poisson!(p, U.ll, source, plan, iplan)

  for k=1:N
    dpk = diff(p,k)*U.cs[k]/U.ll[k]
    Rpre  = CartesianRange(U.cs[1:k-1])
    Rpost = CartesianRange(U.cs[k+1:end])
    _minus_interior!(dest.D[k],U.D[k],dpk,Rpre,Rpost, 2:U.cs[k])
  end
  projBC!(dest,ρ_0,ρ_1)
  source ? dest.Z .= U.Z - p : nothing
end

@noinline function _minus_interior!(dest::AbstractArray, M,dpk,Rpre,Rpost,indx)
  for Ipost in Rpost
    for i in indx
      for Ipre in Rpre
        dest[Ipre,i, Ipost] = M[Ipre,i, Ipost] - dpk[Ipre,i-1,Ipost]
      end
    end
  end
end

"Projection on the interpolation constraint"
function projinterp!{N}(dest::CSvar{N},x::CSvar{N},Q)
  @assert dest.ll == x.ll
  interp!(dest.U,x.V) # interpolate the variable padded with 0
  dest.U.D .= x.U.D + dest.U.D
  for k =1:N
    invQ_mul_A!(dest.U.D[k], Q[k], dest.U.D[k], k)
  end
	dest.U.Z .= (x.U.Z + x.V.Z)/2
  interp!(dest.V,dest.U)
end

"Multiply each slice by Q^-1 using permutedims"
function invQ_mul_A!(dest, Q, src, dim::Integer)
    order = [dim; setdiff(1:ndims(src), dim)]
    srcp = permutedims(src, order)
    invQ = v -> Q\v
    tmp   = mapslices(invQ,srcp,1)
    iorder = [2:dim; 1; dim+1:ndims(src)]
    permutedims!(dest, reshape(tmp, size(dest)[order]), iorder)
    dest
end

"Precompute the interpolation operators"
function precomputeProjInterp(cs)
  B = Array{Any}(length(cs))
  for k = 1:length(cs)
    n = cs[k]
    Q = Tridiagonal(ones(n),cat(1,5.,6*ones(n-1),5.),ones(n))/4
    B[k] = lufact(Q)
  end
  return B
end

"Perform in place Douglas-Rachford iteration"
function stepDR!(w,x,y,z,prox1!,prox2!,α)
  #(1) x = 2*z-w  (2) y = prox1(x) (3) w = w+α(y-z) (4) z = prox2(w)
  broadcast!((z,w)->(2*z-w),x,z,w)
  prox1!(y,x)
  broadcast!((w,y,z)->w+α*(y-z),w,w,y,z)
  prox2!(z,w)
end

"Compute the geodesic between ρ_0 and ρ_1, T time steps"
function computeGeodesic(ρ_0::Array{Float64}, ρ_1::Array{Float64}, T::Integer, ll::Array{Float64,1};
  p=2., q=2., δ = 1.0, niter = 10^3)
  @assert δ > 0
  source = q>=1.0

  if !source
    println("Computing geodesic for standard optimal transport...")
    ρ_1 *= sum(ρ_0)/sum(ρ_1) # rescale the masses
    δ = 1.0 # \in [0,2]
    α, γ = 1.8, max(maximum(ρ_0),maximum(ρ_1))/2
  else
    println("Computing a geodesic for optimal transport with source...")
    α, γ = 1.8, δ^(ndims(ρ_0)-1)*max(maximum(ρ_0), maximum(ρ_1))/15
  end

  # Initilialization using linear interpolation of densities
  w,x,y,z = ntuple(n->CSvar(ρ_0, ρ_1, T, ll),4)
  # change of variable (such that δ = 1)
  dilategrid!(w,1/δ),dilategrid!(x,1/δ),dilategrid!(y,1/δ),dilategrid!(z,1/δ)

  # planning/ initialisation
  plan  = FFTW.plan_r2r!(zeros(x.cs), FFTW.REDFT10)
  iplan = FFTW.plan_r2r!(zeros(x.cs), FFTW.REDFT01)
  Q = precomputeProjInterp(x.cs)
  Flist,Clist  = ntuple(n->zeros(niter),2)

  # define proximal operators
  prox1!(y,x) = (projCE!(y.U,x.U,ρ_0*δ^ndims(ρ_0), ρ_1*δ^ndims(ρ_0),source,plan,iplan),
                proxF!(y.V,x.V, γ, p, q))
  prox2!(y,x) = projinterp!(y,x,Q)

  # Iterations
  onePercent = floor(niter/100)
  for i=1:niter
    if mod(i,onePercent) == 0
      perc = div(i,onePercent)
      print("\rProgress: $(perc)%");
    end
    stepDR!(w,x,y,z,prox1!,prox2!,α)
    Flist[i] = energy(z, δ, p, q)
    Clist[i] = distfromCE(z)
    #Clist[i] = distfromInterp(z)
  end
  projCE!(z.U,z.U,ρ_0*δ^ndims(ρ_0), ρ_1*δ^ndims(ρ_0),source,plan,iplan)
  projpositive!(z)
  dilategrid!(z,δ) # go back to the initial spatial domain
  interp!(z.V,z.U)
  #minF  = energy(z,δ,p,q)
  println();println("Done.")

  return z, (Flist, Clist)
end

end # module
