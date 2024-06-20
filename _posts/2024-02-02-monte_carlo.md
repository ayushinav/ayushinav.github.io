---
layout: post
title:  Monte Carlo
date:   2024-02-02 16:40:16
description: Because basics are important
tags: probability math
categories: 
toc:
  sidebar: left
---
# Why Monte Carlo
More generally, the method is used to perform numerical integration. There exist various other methods for the task., eg. Trapezoid rule, Simpson's rule, Runge-Kutta method of various orders, Guass quadrature and many more. These methods provide a convergence rate of $$\mathcal{O}(n^{-2})$$ or better for $$n$$ nodes. If we know something about the function, we can reduce the number of points and still get the desired accuracy, eg. if a function is known to be quadratic, we can get the exact integral from 3 nodes. 

While these methods may appear good so far, they do not scale well with higher dimensions. Since, these methods rely upon discretizing the space, the number of points increases exponentially with the number of dimensions. In a $$d$$-dimensional space, if we have $$n$$ points in each dimension, the total number of points becomes $$n^d$$. Monte Carlo method of integrating function is essentially mesh-free. It is performed by sampling random points from a certain distribution and evaluating the function at those points. This draws motivation from the Gauss Quadrature to integrate functions.

**Brief intro to Gauss quadrature:**
Gauss quadrature is an efficient way to estimate the integral of a function. It provides better accuracy because unlike Trapezoid rule, Simpson's rules, the node points are also free. Intuitively, it would make more sense to put more points in the region where function peaks to get better estimate. 

The method particularly integrates functions of the form $$\int_a^b f(x) w(x) dx$$ where $$w(x)$$ has properties of a density function. When evaluating $$\int_a^b f(x) dx$$, we simply substitute a $$f^*(x) = f(x)/w(x)$$ and then estimate $$\int_a^b f^*(x) w(x) dx$$.

The properties of $$w(x)$$ determine the orthogonal polynomials, whose roots become the node points and the weights are estimated by a recursion relationship, getting to the form

$$\int_a^b f(x) w(x) dx = \sum_i^n A_i f(x_i)$$

****
Something similar works in Monte Carlo integration. When integrating a function of the form $$\int_a^b f(x) w(x) dx$$, we draw samples from the density function specified by $$w(x)$$ and evaluate $$f(x)$$ at these points, i.e.

$$
\int_a^b f(x) w(x) dx = \sum_i f(x_i) \; , \; x \sim w(x)
$$

The above has its foundations in the law of large numbers. When calculating mean of a distribution, we take the expectation:

$$\mu = \frac{1}{n}\sum_i^n x_i ; \; x_i \sim w(x) \xrightarrow[]{n \rightarrow \infty} \int_{\Omega} w(x) dx$$

and similarly, for the mean of the function on some distribution:

$$\mu(f(\cdot)) = \frac{1}{n}\sum_i^n f(x_i) ; \; x_i \sim w(x) \xrightarrow[]{n \rightarrow \infty} \int_{\Omega} f(x) w(x) dx$$

The above is also written as:

$$
\begin{align*}
  \mathbf{E}_{x \sim w(x)}[f(x)] & = \sum_i f(x_i) \; ; x_i \sim w(x) \\
  & = \int w(x) f(x) dx
\end{align*}
$$

Until now, we'd assumed that sampling from $$w(x)$$ is easy. This might be true for certain families of distribution, but not always. The [Sampling](../../../2024/sampling/) provides dives into how distributions are sampled.
