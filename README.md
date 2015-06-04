# optimal-transport
This Julia toolbox perform solves the optimal transport problem and handle various extensions.
Using the dynamical formulation, 

    min_(r,m,mu) (1/2) \sum_(x,t) ( |m(x,t)|^2/r  + \delta^2  mu(x,t)^2/r)     (1)
    subject to d_t r + div (m) = mu                                             (2)
               r(0,.)=r0 ; r(1,.)=r1                                            (3)
               
it performs Douglas-Rachford proximal spliting to minimize 

          G_1(U) + G_2(V) + G_3(U,V)
          
where: - U is the variables on a centered grid and G_1 the functional (1)
       - V is the variables on a staggered grid and G_2 the convex indicator of the affine constraints (2,3)
       - G_3 is the convex indicator of the interpolation constraint (i.e U= interpolation (V) ).

Supported extensions:
- optimal transport between unbalanced measures,
        - Fisher-Rao penalization on the source
        - L1 penalization on the source (partial transport)
        - L2 penalization on the source

In construction:
- Riemannian geometry
