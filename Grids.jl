module Grids
#=
Tools for staggered and centered grids
=#
#using

import Base.dot

export
    Staggered, Centered,
    projPositive!, dilateDomain!,
    projBoundaryCond!, projBoundaryCond,
    interp, interp!, distTV,
    divergence, divergence!,
    evalFunctional

#######################################
abstract Grid # definition of the father abstract type
#######################################

############################################
# Staggered grid : definition & constructors
# immutable type: one can only change values in the arrays,
# not the entire array.
############################################

type Staggered <: Grid
  cdim::Array{Int64,1} # samples (time and space) for the centered grid
  source::Bool         # with or without source
  lengths::Array{Float64}     # lengths of the time × space hypercube
  ρ::Array{Float64}           # mass
  ω::Array{Array{Float64}}		# momentum
	ζ::Array{Float64}           # source

	function Staggered(cdim,source,lengths,ρ,ω,ζ)
    @assert length(ω) == length(cdim)-1
    @assert length(lengths) == length(cdim)
		@assert size(ζ) == tuple(cdim...)
    @assert size(ρ) == tuple(cdim[1]+1,cdim[2:end]...) # staggered in time
		cbis = copy(cdim)
    for k = 1:length(cdim)-1 # staggered in space
      cbis[k+1] += 1
			@assert size(ω[k]) == tuple(cbis...)
      cbis[k+1] -= 1
		end
		return new(cdim,source,lengths,ρ,ω,ζ)
	end
end # type

function Staggered(ρ::Array{Float64},ω::Array{Array{Float64}},
                   ζ::Array{Float64})
  cdim = [size(ρ)...]
  cdim[1] -= 1
  source = true
  lengths= ones(length(cdim))
  return Staggered(cdim,source,lengths,ρ,ω,ζ)
end # constructor

function Staggered(ρ::Array{Float64},ω::Array{Array{Float64}})
  cdim      = [size(ρ)...]
  cdim[1] -= 1
  source = false
  lengths= ones(length(cdim))
  ζ      = zeros(cdim...)
  return Staggered(cdim,source,lengths,ρ,ω,ζ)
end # constructor

function Staggered(cdim::Array{Int64,1}, source::Bool, randomfill=false)
  lengths= ones(length(cdim))
  if randomfill
    ρ = rand(cdim[1]+1,cdim[2:end]...)
  else
    ρ = zeros(cdim[1]+1,cdim[2:end]...)
  end

  if source && randomfill
    ζ = randn(cdim...)
  else
    ζ = zeros(cdim...)
	end

  ω = cell(length(cdim)-1)
  cbis = copy(cdim)
  for k = 1:length(cdim)-1
    cbis[k+1] += 1
    if randomfill
      ω[k] = randn(cbis...)
    else
      ω[k] = zeros(cbis...)
    end
    cbis[k+1] -= 1
  end

  return Staggered(cdim,source,lengths,ρ,ω,ζ)
end # constructor

function Staggered(ρ,lengths,source)
  cdim = [size(ρ)...]
  cdim[1] -= 1

  ω = cell(length(cdim)-1)
  cbis = copy(cdim)
  for k = 1:length(cdim)-1
    cbis[k+1] += 1
    ω[k] = zeros(cbis...)
    cbis[k+1] -= 1
  end
  ζ = zeros(cdim...)
  return Staggered(cdim,source,lengths,ρ,ω,ζ)
end #constructor

function Staggered(U::Staggered) #make a copy
  cdim = copy(U.cdim)
  lengths = copy(U.lengths)
  ρ = copy(U.ρ)
  ω = deepcopy(U.ω)
  ζ = copy(U.ζ)
  return Staggered(cdim,U.source,lengths,ρ,ω,ζ)
end #constructor

function Staggered(ρ_0,ρ_1, T, source)
  @assert size(ρ_0) == size(ρ_1)
  spatialbox = [size(ρ_0)...]./maximum(size(ρ_0))
  lengths = cat(1,1.0, spatialbox)
    # Initilialization w/ linear interpolation of densities
	cdimt = [T+1, size(ρ_0)... ]
  ρ = zeros(cdimt...)
	t_lin = linspace(0,1,T+1)
  for i = 1:T+1
    t = t_lin[i]
    ρ[i,:] = t*ρ_1[:] + (1-t)*ρ_0[:]
  end
  # all Grids predefined (no temporary)
  return Staggered(ρ,lengths,source) ;
end


#########################################################
# Centered grid. definition & constructors
#########################################################

type Centered <: Grid
  cdim::Array{Int64,1}     # samples (time and space)
  source::Bool          # with or without source
  lengths::Array{Float64} # lengths of the time × space hypercube
  ρ::Array{Float64}           # mass
  ω::Array{Array{Float64}}		# momentum
	ζ::Array{Float64}           # source

  function Centered(cdim,source,lengths,ρ,ω,ζ)
		@assert length(ω) == length(cdim)-1
    @assert length(lengths) == length(cdim)
		@assert size(ζ) == tuple(cdim...)
    @assert size(ρ) == tuple(cdim...)
		for i = 1:length(cdim)-1
			@assert size(ω[i]) ==  tuple(cdim...)
		end
		return new(cdim,source,lengths,ρ,ω,ζ)
  end
end # type

function Centered(ρ::Array{Float64},ω::Array{Array{Float64}},
                  ζ::Array{Float64})
  cdim   = [size(ρ)...]
  source = true
  lengths= ones(length(cdim))
  return Centered(cdim,source,lengths,ρ,ω,ζ)
end # constructor

function Centered(ρ::Array{Float64},ω::Array{Array{Float64}})
  cdim   = [size(ρ)...]
  source = false
  lengths= ones(length(cdim))
  ζ      = zeros(cdim...)
  return Centered(cdim,source,lengths,ρ,ω,ζ)
end # constructor

function Centered(cdim::Array{Int64,1}, source::Bool, randomfill=false)
  lengths= ones(length(cdim))
  if randomfill
    ρ = rand(cdim...)
  else
    ρ = zeros(cdim...)
  end

  if source && randomfill
    ζ = rand(cdim...)
  else
    ζ = zeros(cdim...)
	end

  ω = cell(length(cdim)-1)
  for i=1:length(cdim)-1
    if randomfill
      ω[i] = rand(cdim...)
    else
      ω[i] = zeros(cdim...)
    end
  end
  return Centered(cdim,source,lengths,ρ,ω,ζ)
end # constructor


#########################################################
# Basic methods on Grids
#########################################################

function projPositive!(A::Grid)
  for i in 1:length(A.ρ)
    A.ρ[i] = max(A.ρ[i], 0)
  end
end # proj_plus!

function distTV(A::Grid,B::Grid)
  @assert typeof(A) == typeof(B)
  d = 0
  d += sum(abs(A.ρ-B.ρ))
  for k = 1:length(A.cdim)-1
    d += sum(abs(A.ω[k]-B.ω[k]))
  end
  d += sum(abs(A.ζ-B.ζ))
  return d
end

# dot , times  , - , +
# dist
# max
# min
# mtimes

################################################
## Change of variable: pushforward by dilation
################################################
function dilateDomain!(U::Grid,δ::Number)
  # apply the pushforward of T: (t,x)-> (t,δx)
  # in such a way to remain solution to the constraints
  # (see Lemma in the paper)
  d = length(U.cdim)-1
  U.lengths[2:end] = δ*U.lengths[2:end]
  U.ρ[:] = U.ρ[:]/δ^d
  U.ω[:] = U.ω[:]/δ^(d-1)
  U.ζ[:] = U.ζ[:]/δ^d
end

#########################################################
# Projection on the boundary conditions
#########################################################

function projBoundaryCond(U::Staggered,
                  ρ_0::Array{Float64},ρ_1::Array{Float64})
  dimΩ = length(U.cdim)-1
	if dimΩ > 4
		error("More than 5 space dimensions not supported")
	end
  U′=Staggered(U);
  # ρ(0,⋅)=ρ_0 and ρ(1,⋅)=ρ_1
  U′.ρ[1,:]   = ρ_0
  U′.ρ[end,:] = ρ_1
	# Neumann : ω⋅n = 0
	            U′.ω[1][:,[1,end],:,:,:,:] = 0
  if dimΩ>1   U′.ω[2][:,:,[1,end],:,:,:] = 0 end
  if dimΩ>2 	U′.ω[3][:,:,:,[1,end],:,:] = 0 end
  if dimΩ>3 	U′.ω[3][:,:,:,:,[1,end],:] = 0 end
  if dimΩ>4 	U′.ω[4][:,:,:,:,:,[1,end]] = 0 end
  # easy to extend, but certainly useless.
  return U′
end # projBoundaryCond

function projBoundaryCond!(U::Staggered,
                  ρ_0::Array{Float64},ρ_1::Array{Float64})
  dimΩ = length(U.cdim)-1
	if dimΩ > 4
		error("More than 5 space dimensions not supported")
	end
  # ρ(0,⋅)=ρ_0 and ρ(1,⋅)=ρ_1
  U.ρ[1,:]   = ρ_0
  U.ρ[end,:] = ρ_1
	# Neumann : ω⋅n = 0
	            U.ω[1][:,[1,end],:,:,:,:] = 0
  if dimΩ>1   U.ω[2][:,:,[1,end],:,:,:] = 0 end
  if dimΩ>2 	U.ω[3][:,:,:,[1,end],:,:] = 0 end
  if dimΩ>3 	U.ω[3][:,:,:,:,[1,end],:] = 0 end
  if dimΩ>4 	U.ω[4][:,:,:,:,:,[1,end]] = 0 end
  # easy to extend, but certainly useless.
end # projBoundaryCond!
#########################################################
# Functional : Centered -> R_+
#########################################################
function evalFunctional(V::Centered;
                  δ = 1.0, method="General", order =(2,2))
  F = 0.0
  ϵ = 0.0
  dimΩ = length(V.cdim)-1
  p = order[1]; q = order[2];

  if isequal(method,"General")
    # F = ∫∫ (1/p) |ω|^p/ρ^(p-1) + δ^p (1/q) |ζ|^q/ρ^(q-1)
    f = zeros(V.ρ);
    fp = 0.0; fq = 0.0
    for i = 1:length(V.ρ)
      fp = 0.0; fq = 0.0
      if V.ρ[i]>ϵ
        for k = 1:dimΩ
          fp += abs(V.ω[k][i])^p/p
        end
        V.source ? fq = abs(V.ζ[i])^q/q : fq = 0.0;
        f[i] = fp/V.ρ[i]^(p-1) + δ^p * fq/V.ρ[i]^(q-1)
      end
        F = sum(f)*prod(V.lengths)/prod(V.cdim)
    end
  end

  if isequal(method,"L2") # not used
    # F = ∫∫ (1/2) |ω|^2/ρ + ∫∫ (1/2) ζ^2
    f = zeros(V.ρ);
    fp = 0.0; fq = 0.0
    for i =1:length(V.ρ)
      fp = 0.0;
      if V.ρ[i]>ϵ
        for k = 1:dimΩ
          fp += abs(V.ω[k][i])^2/2
        end
        f[i] = fp/V.ρ[i] + δ^2 * V.ζ[i]^2/2
      end
        F = sum(f)*prod(V.lengths)/prod(V.cdim)
    end
  end

  return F
end


#########################################################
# Interpolation : Staggered -> Centered
#########################################################
function interp(U::Staggered)
  # returns interp(U)
  # devectorized
  dimΩ = length(U.cdim)-1
  if dimΩ > 3 error("Too Many space dimensions") end
  ρ = zeros(U.cdim...)
  ω = cell(dimΩ)
  ζ = zeros(U.cdim...)
  for i = 1:U.cdim[1]
    ρ[i,:] = (U.ρ[i,:]+U.ρ[i+1,:])/2
  end

  if dimΩ > 0
    ω[1] = zeros(U.cdim...)
    for i = 1:U.cdim[2]
        ω[1][:,i,:,:] = (U.ω[1][:,i,:,:] + U.ω[1][:,i+1,:,:])/2
    end
    if dimΩ > 1
      ω[2] = zeros(U.cdim...)
      for i = 1:U.cdim[3]
        ω[2][:,:,i,:] = (U.ω[2][:,:,i,:] + U.ω[2][:,:,i+1,:])/2
      end
    end
    if dimΩ > 2
      ω[3] = zeros(U.cdim...)
      for i = 1:U.cdim[4]
        ω[3][:,:,:,i] = (U.ω[3][:,:,:,i] + U.ω[3][:,:,:,i+1])/2
      end
    end
  end
  for i = 1:length(U.ζ)
    ζ[i] = U.ζ[i]
  end
  return Centered(copy(U.cdim),U.source,copy(U.lengths),ρ,ω,ζ)
end # interp


function interp!(V::Centered,U::Staggered)
  # writes the result of interp(U) in V
  # devectorized
  @assert V.cdim    == U.cdim
  @assert V.lengths == U.lengths
  @assert V.source  == U.source
  dimΩ = length(U.cdim)-1
  if dimΩ > 3 error("Too Many space dimensions") end
  for i = 1:U.cdim[1]
    V.ρ[i,:] = (U.ρ[i,:]+U.ρ[i+1,:])/2
  end

  if dimΩ > 0
    for i = 1:U.cdim[2]
        V.ω[1][:,i,:,:] = (U.ω[1][:,i,:,:] + U.ω[1][:,i+1,:,:])/2
    end
    if dimΩ > 1
      for i = 1:U.cdim[3]
        V.ω[2][:,:,i,:] = (U.ω[2][:,:,i,:] + U.ω[2][:,:,i+1,:])/2
      end
    end
    if dimΩ > 2
      for i = 1:U.cdim[4]
        V.ω[3][:,:,:,i] = (U.ω[3][:,:,:,i] + U.ω[3][:,:,:,i+1])/2
      end
    end
  end

  for i = 1:length(V.ζ)
    V.ζ[i] = U.ζ[i]
  end

end # end of interp!


#########################################################
# Divergence operator
#########################################################

function divergence(U::Staggered)
	# computes the disgression to the continuity equation.
	# v = ∂_t ρ + ∇⋅ω - ζ   # nabla-> del
	# l is the space cube length
	v = zeros(U.cdim...)
	f = u -> u[2:end]-u[1:end-1] # discrete differentiation

  v  += mapslices(f,U.ρ,1) * U.cdim[1]/U.lengths[1]
	for i = 1:length(U.cdim)-1
		v += mapslices(f,U.ω[i],i+1)*U.cdim[i+1]/U.lengths[i+1]
	end

	if U.source
		v -= U.ζ
	end
	return v
end # divergence

function divergence!(v::Array{Float64}, U::Staggered)
  # computes the disgression to the continuity equation.
	# v = ∂_t ρ + ∇⋅ω - ζ   # nabla-> del
	# l is the space cube length
  # on place, devectorized
  dimΩ = length(U.cdim)-1
  if dimΩ > 3 error("Too Many space dimensions") end

  for i = 1:U.cdim[1]
    v[i,:] = (U.ρ[i+1,:]-U.ρ[i,:])*U.cdim[1]/U.lengths[1]
  end

  if dimΩ > 0
    for i = 1:U.cdim[2]
      v[:,i,:,:] +=
        (U.ω[1][:,i+1,:,:] - U.ω[1][:,i,:,:])*U.cdim[2]/U.lengths[2]
    end
  end
  if dimΩ > 1
    for i = 1:U.cdim[3]
      v[:,:,i,:] +=
        (U.ω[2][:,:,i+1,:] - U.ω[2][:,:,i,:])*U.cdim[3]/U.lengths[3]
    end
  end
  if dimΩ > 2
    for i = 1:U.cdim[4]
      v[:,:,:,i] +=
        (U.ω[3][:,:,:,i+1] - U.ω[3][:,:,:,i])*U.cdim[4]/U.lengths[4]
    end
  end
  if U.source
    for i = 1:length(v)
      v[i] -= U.ζ[i]
    end
  end
end # divergence!

end # module
