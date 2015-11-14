module DynamicOT
#=
Functions for dynamic optimal transport
=#
using Grids

export
    findRoot, proxFunctional!,
    poisson!, projConstraints!,
    projInterp!, interp_proj_fast!,
    precomputeProjInterp,
    doStepDR!, solveGeodesic,
    findRootNewton


################################################
# Computes root of a polynomial
################################################
function findRoot(a,b,c,d)
  # highest real root of aX^3+bX^2+cX+d
  		p = -(b/a)^2/3 + c/a
  		q = 2*(b/a)^3/27 - b*c/a^2/3 + d/a
  		Δ = q.^2 + 4/27*p.^3
      if Δ > 0
    		u       = ∛((-q + sqrt(Δ))/2)
		    v       = ∛((-q - sqrt(Δ))/2)
   		 	z = u + v - b/ a /3
      elseif Δ < 0
  		  u = ((-q + im*sqrt(-Δ))/2)^(1/3)
    		z = real(u + conj(u) - b/ a /3)
      else
    		z =  real(3*q ./ p - b/ a /3)
  		end
  return z
end

#=
function findRoot(f,f′,x0 ,tol = 1e-15)
  # highest real root of f by Newton method ( f is its derivative)
  ff = f(x0)
  while abs(ff) > tol
    x0 -= ff/f′(x)
    ff = f(x0)
  end
  return x0
end
=#

################################################
# Proximal operator of the functional
################################################

function proxFunctional!(dest, V, γ, p, q)
  dimΩ = length(V.cdim)-1
  if p==1 && q == 0 # W_1 / Kantorovitch-Rubinstein distances
    m = 0.0
    for i = 1:length(V.ρ)
      m = 0.0
      dest.ρ[i] = V.ρ[i]
      for k = 1:dimΩ
        m += V.ω[k][i]^2
      end
      softth = max(1-γ/sqrt(m),0)
      for k = 1:dimΩ
        dest.ω[k][i] = softth*V.ω[k][i]
      end
    end
  elseif p==2 && q==0 # W_2 - 2- Wasserstein (std Benamou-Brenier)
    ρ = 0.0 ; m = 0.0; a = 0.0; b = 0.0; c = 0.0; d = 0.0; D = 0.0;
    for i = 1:length(V.ρ)
      ρ = V.ρ[i]; m = 0.0
      for k = 1:dimΩ
        m += V.ω[k][i]^2
      end
      a = 1
      b = 2γ - ρ
      c = γ^2 - 2γ*ρ
      d = -(γ/2)*m - γ^2*ρ
      dest.ρ[i] = max(0,findRoot(a,b,c,d))
      (dest.ρ[i] == 0) ? (D = 0) : (D = 1 - γ/(γ + dest.ρ[i]))
      for k = 1:dimΩ
        dest.ω[k][i] = D * V.ω[k][i]
      end
    end

  elseif p==1 && q==1 # Hanin / partial W_1 (Piccoli Rossi order 1)
    m = 0.0
    for i = 1:length(V.ρ)
      m = 0.0
      dest.ρ[i] = V.ρ[i]
      for k = 1:dimΩ
        m += V.ω[k][i]^2
      end
      softth = max(1-γ/sqrt(m),0)
      for k = 1:dimΩ
        dest.ω[k][i] = softth*V.ω[k][i]
      end
      dest.ζ[i] = max(1-γ/abs(V.ζ[i]),0)*V.ζ[i]
    end

  elseif p==2 && q==1 # Partial W_2 (Piccoli Rossi order 2)
    # standard proximal for the transport part
    ρ = 0.0 ; m = 0.0; a = 0.0; b = 0.0; c = 0.0; d = 0.0; D = 0.0;
    for i = 1:length(V.ρ)
      ρ = V.ρ[i]; m = 0.0
      for k = 1:dimΩ
        m += V.ω[k][i]^2
      end
      a = 1
      b = 2γ - ρ
      c = γ^2 - 2γ*ρ
      d = -(γ/2)*m - γ^2*ρ
      dest.ρ[i] = max(0,findRoot(a,b,c,d))
      (dest.ρ[i] == 0) ? (D = 0) : (D = 1 - γ/(γ + dest.ρ[i]))
      for k = 1:dimΩ
        dest.ω[k][i] = D * V.ω[k][i]
      end
      dest.ζ[i] = max(1-γ/abs(V.ζ[i]),0)*V.ζ[i]
    end

    elseif p==1 && q==2 # Interpolating W1 & Fisher-Rao
    # standard proximal for the growth part
    ρ = 0.0 ; m = 0.0; a = 0.0; b = 0.0; c = 0.0; d = 0.0; D = 0.0;
    for i = 1:length(V.ρ)
      ρ = V.ρ[i]
      m = V.ζ[i]^2
      a = 1
      b = 2γ - ρ
      c = γ^2 - 2γ*ρ
      d = -(γ/2)*m - γ^2*ρ
      dest.ρ[i] = max(0,findRoot(a,b,c,d))
      (dest.ρ[i] == 0) ? (D = 0) : (D = 1 - γ/(γ + dest.ρ[i]))
      dest.ζ[i] = D * V.ζ[i]
      m = 0.0
      for k = 1:dimΩ
        m += V.ω[k][i]^2
      end
      softth = max(1-γ/sqrt(m),0)
      for k = 1:dimΩ
        dest.ω[k][i] = softth*V.ω[k][i]
      end
    end

  elseif p==2 && q==2 #interpolating distance WF
    ρ = 0.0 ; m = 0.0; a = 0.0; b = 0.0; c = 0.0; d = 0.0
    for i = 1:length(V.ρ)
      ρ = V.ρ[i]; m = V.ζ[i]^2 # <- here the difference
      for k = 1:dimΩ
        m += V.ω[k][i]^2
      end
      a = 1
      b = 2γ - ρ
      c = γ^2 - 2γ*ρ
      d = -(γ/2)*m - γ^2*ρ
      dest.ρ[i] = max(0,findRoot(a,b,c,d))
      D = 1 - γ/(γ + dest.ρ[i])
      for k = 1:dimΩ
        dest.ω[k][i] = D * V.ω[k][i]
      end
      dest.ζ[i] = D * V.ζ[i]
    end
  elseif p==Inf && q==1 # tolerant TV
    error("Not written")
  elseif p==Inf && q==2 # tolerant FR
    error("Not written")
  elseif p==1 && q==Inf # tolerant W_1
    error("Not written")
  elseif p==2 && q==Inf # tolerant W_2
    error("Not written")
  else # general method for any p,q
    f = x -> 0. ; f′ = x -> 0.
      for i = 1:length(V.ρ)
      ρ = V.ρ[i]; m = V.ζ[i]^2
      for k = 1:dimΩ
        m += V.ω[k][i]^2
      end
      f  =
      f′ =
      dest.ρ[i] = max(0,findRoot(a,b,c,d))
      D = 1 - γ/(γ + dest.ρ[i])
      for k = 1:dimΩ
        dest.ω[k][i] = D * V.ω[k][i]
      end
      dest.ζ[i] = D * V.ζ[i]
    end
  end # fi

end #proxFunctional!

################################################
# Solves the poisson equation with Dirichlet conditions
################################################

function poisson!(f::Array{Float64}, lengths::Array{Float64},
                source::Bool,
                plan="none", iplan="none")
""""   solves Δu + f = 0
       Neumann on the staggered grid -> use  DCT II / DCT III
       because x(-1/2)=x(N+1/2)=0
	     with the contraints: mean(u-f)=0 and 0 boundary cond.
	     if withmu: solves instead   Δu -u + f =0
	     plan /iplan are "prepared" DCT! (for speed)
"""

  d = length(size(f))
	N = [size(f)...]
	h = lengths./N # sampling interval
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
			plan  = v -> FFTW.r2r!(v,FFTW.REDFT10,[1:d]) #DCT 10
			iplan = v -> FFTW.r2r!(v,FFTW.REDFT01,[1:d]) #iDCT 01
	end

  plan(f) #on place
  renorm = prod(N)*2^d
  for i =1:length(f)
    f[i] = - f[i]/D[i]/renorm
  end
  iplan(f)
end # poisson!

################################################
# Projection on the constraints
################################################
function projConstraints!(dest::Staggered,U::Staggered,
                   ρ_0::Array{Float64}, ρ_1::Array{Float64},
						       plan="none",iplan="none")
	# project U on the constraints, write in dest
	# plan=plan_fft(A,dims), A should be of same type and size
  @assert dest.cdim   == U.cdim
  @assert dest.source == U.source
  @assert dest.lengths== U.lengths
  cdim = U.cdim
  dimΩ = length(cdim)-1
  (dimΩ > 3) && error("Too many dimensions")

	U′ = projBoundaryCond(U, ρ_0, ρ_1)
  #projBoundaryCond!(U, ρ_0, ρ_1) # slightly faster...but untrue
	p = - divergence(U′)
	poisson!(p, U.lengths, U.source, plan, iplan)

  dest.ρ[1,:,:,:] = ρ_0 ; dest.ρ[end,:,:,:] = ρ_1
  for i = 2:cdim[1] #do not touch boundaries
    dest.ρ[i,:,:,:] = U.ρ[i,:,:,:] -
      (p[i,:,:,:] - p[i-1,:,:,:]) * U.cdim[1]/U.lengths[1]
  end

  dest.ω[1][:,1,:,:] = 0; dest.ω[1][:,end,:,:] = 0
  for i = 2:cdim[2]
    dest.ω[1][:,i,:,:] = U.ω[1][:,i,:,:] -
      (p[:,i,:,:] - p[:,i-1,:,:]) * U.cdim[2]/U.lengths[2]
  end

  if dimΩ >= 2
    dest.ω[2][:,:,1,:] = 0; dest.ω[2][:,:,end,:] = 0
    for i = 2:cdim[3]
      dest.ω[2][:,:,i,:] = U.ω[2][:,:,i,:] -
        (p[:,:,i,:] - p[:,:,i-1,:]) * U.cdim[3]/U.lengths[3]
    end
  end

  if dimΩ >= 3
    dest.ω[3][:,:,:,1] = 0; dest.ω[3][:,:,:,end] = 0
    for i = 2:cdim[3]
      dest.ω[3][:,:,:,i] = U.ω[3][:,:,:,i] -
        (p[:,:,:,i] - p[:,:,:,i-1]) * U.cdim[4]/U.lengths[4]
    end
  end

	if U.source
		  dest.ζ[:] = U.ζ[:] - p[:]
	end
end # projConstraints!


################################################
## Projection on interpolation constraints
################################################

function projInterp!(U::Staggered,V::Centered,
                     U0::Staggered,V0::Centered,Q)
 	# U=Q^{-1}(U0 + I^*V0), V=interp(U)
  #(U,V) are the destinations

  @assert U0.cdim   == V0.cdim
  @assert U0.source == V0.source
  @assert U0.lengths== V0.lengths

	dimΩ = length(U.cdim)-1
  U.ρ[1,:,:,:] = U0.ρ[1,:,:,:] + V0.ρ[1,:,:,:]/2
  for i= 2:U.cdim[1]
    U.ρ[i,:,:,:] =
      U0.ρ[i,:,:,:] + (V0.ρ[i-1,:,:,:] + V0.ρ[i,:,:,:])/2
  end
  U.ρ[end,:,:,:] = U0.ρ[end,:,:,:] + V0.ρ[end,:,:,:]/2
  invQ = v -> Q[1]\v
  U.ρ   = mapslices(invQ,U.ρ,1)

  U.ω[1][:,1,:,:] = U0.ω[1][:,1,:,:] + V0.ω[1][:,1,:,:]/2
  for i= 2:U.cdim[2]
    U.ω[1][:,i,:,:] =
      U0.ω[1][:,i,:,:] + (V0.ω[1][:,i-1,:,:] + V0.ω[1][:,i,:,:])/2
  end
  U.ω[1][:,end,:,:] = U0.ω[1][:,end,:,:] + V0.ω[1][:,end,:,:]/2
  invQ = v -> Q[2]\v
  U.ω[1]   = mapslices(invQ,U.ω[1],2)

  if dimΩ >=2
    U.ω[2][:,:,1,:] = U0.ω[2][:,:,1,:] + V0.ω[2][:,:,1,:]/2
    for i= 2:U.cdim[3]
      U.ω[2][:,:,i,:] =
        U0.ω[2][:,:,i,:] + (V0.ω[2][:,:,i-1,:] + V0.ω[2][:,:,i,:])/2
    end
    U.ω[2][:,:,end,:] = U0.ω[2][:,:,end,:] + V0.ω[2][:,:,end,:]/2
    invQ = v -> Q[3]\v
    U.ω[2]   = mapslices(invQ,U.ω[2],3)
  end

  if dimΩ >= 3
    U.ω[3][:,:,:,1] = U0.ω[3][:,:,:,1] + V0.ω[3][:,:,:,1]/2
    for i= 2:U.cdim[4]
      U.ω[3][:,:,:,i] =
        U0.ω[3][:,:,:,i] + (V0.ω[3][:,:,:,i-1] + V0.ω[3][:,:,:,i])/2
    end
    U.ω[3][:,:,:,end] = U0.ω[3][:,:,:,end] + V0.ω[3][:,:,:,end]/2
    invQ = v -> Q[4]\v
    U.ω[3]   = mapslices(invQ,U.ω[3],4)
  end

	if U0.source
		U.ζ[:] = (U0.ζ[:] + V0.ζ[:])/2
	end
  interp!(V,U)
end # projInterp!

#############################################
#Precomputation of some operators
################################################

function precomputeProjInterp(cdim)
	# computes all the proj operators for [dims...]
	B = cell(length(cdim))
	for k = 1:length(cdim)
		n = cdim[k]
		Q = Tridiagonal(ones(n),cat(1,5.,6*ones(n-1),5.),ones(n))/4
	  B[k] = lufact(Q)
	end
	return B
end # precomputeProjInterp

###############################################
## Efficient Douglas-Rachford step
###############################################
function doStepDR!(WU::Staggered,WV::Centered,
                   XU::Staggered,XV::Centered,
                   YU::Staggered,YV::Centered,
                   ZU::Staggered,ZV::Centered,
                   proxG1::Function, proxG2::Function, proxG3::Function,
                   α = 1.0)
  # one iteration of DR , all in place, no temporary variables
  dimΩ = length(WU.cdim)-1

  # 1. X = 2Z - W
  XU.ρ[:] = 2ZU.ρ[:] - WU.ρ[:]
  XV.ρ[:] = 2ZV.ρ[:] - WV.ρ[:]

  for k = 1:dimΩ
    XU.ω[k][:] = 2ZU.ω[k][:]  - WU.ω[k][:]
    XV.ω[k][:] = 2ZV.ω[k][:]  - WV.ω[k][:]
  end
  if WU.source
    XU.ζ[:] = 2ZU.ζ[:] - WU.ζ[:]
    XV.ζ[:] = 2ZV.ζ[:] - WV.ζ[:]
  end

  # 2. Y = Prox X
  proxG1(YU,XU)
  proxG2(YV,XV)

  # 3. W = W + α ( Y - Z)
  WU.ρ[:] =  WU.ρ[:] + α*(YU.ρ[:] - ZU.ρ[:])
  WV.ρ[:] =  WV.ρ[:] + α*(YV.ρ[:] - ZV.ρ[:])
  for k = 1:dimΩ
    WU.ω[k][:] =  WU.ω[k][:] + α*(YU.ω[k][:] - ZU.ω[k][:])
    WV.ω[k][:] =  WV.ω[k][:] + α*(YV.ω[k][:] - ZV.ω[k][:])
  end
  if WU.source
    WU.ζ[:] =  WU.ζ[:] + α*(YU.ζ[:] - ZU.ζ[:])
    WV.ζ[:] =  WV.ζ[:] + α*(YV.ζ[:] - ZV.ζ[:])
  end

  # 4. Z = Prox(W)
  proxG3(ZU,ZV,WU,WV)
end # doStepDR!


################################################
## Find the geodesics
################################################

function solveGeodesic(ρ_0, ρ_1, T;
                       order = (2.,2.), δ = 1.0, niter = 10^3)
  @assert size(ρ_0 ) == size(ρ_1)
  @assert length(size(ρ_0)) < 4
  @assert δ > 0
  cdim = tuple(T,size(ρ_0)...)
  dimΩ = length(size(ρ_0))
  p = order[1]
  q = order[2]
  ρ_0 = float64(ρ_0)
  ρ_1 = float64(ρ_1)

  if q==0
    println("Computing geodesic for standard optimal transport...")
    source = false
		ρ_1 *= sum(ρ_0)/sum(ρ_1) # rescale the masses
		δ = 1.0
		γ = max(maximum(ρ_0),maximum(ρ_1))/15
		α = 1.8
  else
    println("Computing a geodesic for optimal transport with source...")
    source = true
		γ = δ^(dimΩ)*max(maximum(ρ_0), maximum(ρ_1))/15
		α = 1.8 # \in [0,2]
  end
  # Initilialization using linear interpolation of densities
  WU = Staggered(ρ_0,ρ_1, T, source)
  dilateDomain!(WU,1/δ) # change of variable (s.t. δ = 1)

  # all Grids predefined (no temporary)
  XU = Staggered(WU) ;  YU = Staggered(WU) ; ZU = Staggered(WU) #copies
  WV = interp(WU); XV = interp(WU);  YV = interp(WU); ZV = interp(WU)

  # planning/ initialisation
	plan  = FFTW.plan_r2r!(zeros(cdim),
					FFTW.REDFT10, [1:length(cdim)])
	iplan = FFTW.plan_r2r!(zeros(cdim),
					FFTW.REDFT01, [1:length(cdim)])
	Flist  = zeros(niter); Clist = zeros(niter); Ilist = zeros(niter);

  # define proximal operators (shorten the code)
  proxG1(dest,U) = projConstraints!(dest, U, ρ_0*δ^dimΩ, ρ_1*δ^dimΩ,
                                    plan, iplan)
  proxG2(dest,V) = proxFunctional!(dest, V, γ, p, q)
	Q = precomputeProjInterp(cdim)
	proxG3(U,V,U0,V0) = projInterp!(U,V,U0,V0,Q)

  # Iterations
  onePercent = floor(niter/100)
	for i=1:niter
		if mod(i,onePercent) == 0
      perc = div(i,onePercent)
			print("\rProgress: $(perc)%");
		end
    doStepDR!(WU,WV,XU,XV,YU,YV,ZU,ZV, proxG1,proxG2,proxG3, α )

		Flist[i]  = evalFunctional(ZV, δ = δ,  method = "General",
                               order = (p,q))
		Clist[i] = sum(divergence(ZU).^2)
    #Ilist[i] = distTV(WV,interp(WU))
	end
  proxG1(ZU,ZU) # project on constraints
  projPositive!(ZU)
  dilateDomain!(ZU,δ) # go back to the initial spatial domain
  minF  = evalFunctional(interp(ZU), δ = δ,  method = "General", order = (p,q))
  println();println("Done.")

	return (ZU, minF, Flist, Clist, Ilist)
end # solveGeodesic


end # module

