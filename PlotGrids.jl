module PlotGrids
#=
Tools for ploting geodesics and relevant informations
=#
using PyPlot, Grids

import PyPlot.plot

export
    plot1d,  plot2d,
    plotVelocity

function plot1d(U::Staggered; time = 0.5)
  dimΩ = length(U.cdim)-1
  @assert dimΩ == 1
  t = time;
  if t >= 1.0 || t <= 0.0
    error("Choose a time in ]0,1[ for φ to be defined.")
  end
  T = U.cdim[1]
  V = interp(U)
  ϵ = maximum(U.ρ)/1e1
  ind = ϵ .< abs( V.ρ )
  φ = zeros(V.cdim...)
  φ[ind] = V.ζ[ind]./V.ρ[ind]
  φ[φ .== 0] = NaN

  v = zeros(V.cdim...)
  v[ind] = V.ω[1][ind]./V.ρ[ind]
  v[v .== 0] = NaN

    fig, ax1 = subplots(figsize=[16,5])
    ax1[:set_ylabel]("Density",color="r");
    for tl in ax1[:get_yticklabels]()
        tl[:set_color]("r")
    end
    K = U.cdim[2]
    x = linspace(0,1,K)
    x′= linspace(0-1/K, 1+1/K, K+2)
    ax1[:fill](x′,[0,U.ρ[1,:]',0],"k",alpha=0.2)
    ax1[:fill](x′,[0,U.ρ[end,:]',0],"b",alpha=0.2)
    ax1[:fill](x′,[0,V.ρ[floor(t*T)+1,:]',0],"r",alpha=0.5)
    ax1[:fill](x′,[0,V.ζ[floor(t*T)+1,:]',0],"y",alpha=0.3)

    ax2 = ax1[:twinx]()
    ax2[:set_ylabel]("Rate of growth, velocity")
    ax2[:plot](x,φ[floor(t*T)+1,:]',"k",alpha=1)
    ax2[:plot](x,v[floor(t*T)+1,:]',"g",alpha=1)
    title(L"(grey) $\rho_0$, (blue) $\rho_1$ (red) $\rho_{t}$ (yellow) source $\zeta_t$ (black) rate of growth $\zeta/\rho$ (green) velocity $\omega/\rho$")

end # plot

function plot2d(U::Staggered, npics, nrows)
  dimΩ = length(U.cdim)-1
  @assert dimΩ == 2
  T = U.cdim[1]
  V = interp(U)

  ncolumns = div(npics-1,nrows)+1
  fig, ax = subplots(figsize=[16,10],nrows,ncolumns)
  for k = 1:npics
    if k == 1
      ρ = reshape(U.ρ[1,:],U.cdim[2:end]...)
    elseif k==npics
      ρ = reshape(U.ρ[end,:],U.cdim[2:end]...)
    else
      t = floor(k*(T-1)/(npics-1))+1
      ρ = reshape(V.ρ[t,:],U.cdim[2:end]...)
    end
    # need to convert the numbering...
    r = div(k-1,ncolumns)+1
    c = k-(r-1)*ncolumns
    n = ((c-1)*nrows)+r
    ax[n][:imshow](ρ,cmap="gist_gray",interpolation="none")
  end

  for k = 1:length(ax)
    ax[k][:axis]("off")
  end
end # plot2d

function plotVelocity(U::Staggered;time=0.5)
    t = time;
  if t >= 1.0 || t <= 0.0
    error("Choose a time in ]0,1[ for φ to be defined.")
  end
  T = U.cdim[1]
  V=interp(U)
  k = floor(T*t)+1
  d = 1
  ϵ = 1e-4
  figure(figsize=[5,5])
  X = linspace(0,U.lengths[3],int(size(U.ρ,3)/d));
  Y = linspace(0,U.lengths[2],int(size(U.ρ,2)/d));
  # Compute arrays
  ρ  = V.ρ[k,1:d:end,1:d:end]
  ρ = reshape(ρ,(U.cdim[2],U.cdim[3]))
  ζ  = V.ζ[k,1:d:end,1:d:end]
  ζ = reshape(ζ,(U.cdim[2],U.cdim[3]))
  ω_x = V.ω[2][k,1:d:end,1:d:end]
  ω_x = reshape(ω_x,(U.cdim[2],U.cdim[3]))
  ω_y = -V.ω[1][k,1:d:end,1:d:end]
  ω_y = reshape(ω_y,(U.cdim[2],U.cdim[3]))

  v_x = zeros(ρ); v_y = zeros(ρ)
  v_x[ρ.>ϵ] = ω_x[ρ.>ϵ]./ρ[ρ.>ϵ]
  v_y[ρ.>ϵ] = ω_y[ρ.>ϵ]./ρ[ρ.>ϵ]

  ω = sqrt(ω_x.^2 + ω_y.^2)
 # ω[1]=0; ω[2]=.0002;
  v = sqrt(v_x.^2 + v_y.^2)
quiver(X,flipdim(Y,1),v_x,v_y,ρ,cmap="Greys",scale=2.5)
axis("off"); #axis([0, U.lengths(2),0, U.lengths(3)])
axis("scaled")

end # plotVelocity

end # module