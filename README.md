# Numerical methods for (unbalanced) optimal transport
This code is not maintained anymore and written in a deprecated version of Julia. 
For an stable implementation of unbalanced optimal transport, see [Gabriel Peyré's code](https://github.com/gpeyre/2017-MCOM-unbalanced-ot/blob/master/matlab/sinkhorn_log.m).

--------------------------------------

This Julia toolbox provides several tools for solving optimal transport, the unbalanced extensions and related problems.

What you can find here:
- a computation of (unbalanced) optimal transport geodesics through extensions of the Benamou-Brenier dynamic formulation.
- a demonstration of "scaling algorithms" for computing (unbalanced) optimal transport using the Kantorovich formulation
- color transfer using optimal transport ;
- Wasserstein gradient flow of the total variation functional (in 1d)
- a tumor growth model computed as a gradient flow with an unbalanced optimal transport metric.
- (in construction) (unabalanced) OT barycenters for big problems (using multiscale).


Check the associated article where the mathematical framework and algorithms are described:
- [An interpolating distance between optimal transport and Fisher-Rao](http://arxiv.org/abs/1506.06430)
- [Unbalanced optimal transport: geometry and Kantorovich formulation](https://arxiv.org/abs/1508.05216)
- [Scaling algorithms for unbalanced optimal transport](https://arxiv.org/pdf/1607.05816)
- in preparation(with S. Di Marino) : a tumor growth Hele-Shaw problem as a gradient flow

and see the associated notebooks for a simple overview.

Extensions in preparation (if time permits!):
- extension of the dynamic solver to Riemannian manifolds
- gradient flow of the total variation in 2D
