---
layout: post
title:  Variational Inference
date:   2015-03-15 16:40:16
description: Understanding variational inference and applying it to geophysical inversion
tags: 
categories: 
---
<h3>Theory of variational inference </h3>

We have the obs data $$x$$ sampled from an unknown complex distribution $p(x)$. Assuming a latent variable $$z \sim p(z)$$, were $$p(z)$$ is some simple distribution and also assuming $$p(x|z)$$ is simple, we can compute $$p(x)$$ by 
$$$$p(x)= \int p(x|z)p(z)dz$$$$
Assuming $$p(x)$$ is parameterized on $$\theta$$ (as an example $$\theta$$ can represent the mean and variance of a normal distribution, a complex version can be called the parameters of a GMM, a linear combination of multiple gaussians), then it is easy to see that $$p(x|z)$$ will be parameterized on $$\theta$$ where $$\theta$$ might come from $$z$$, that is the various means and variance of the different gaussians, and probably also which gaussian to sample from. This means 
$$p_{\theta}(x)= \int p_{\theta}(x|z)p(z)dz$$
Now, our objective is to estimate $$p(x)$$, that is, the parameters of $$p(x)$$, that is $$\theta$$. By probabilistic regression, we want to maximize the log-likelihood $$ \log p(x)$$. 

$$
\begin{aligned}
 \therefore \theta &= \underset{\theta}{\arg \max} \underset{i}\sum \log p_{\theta}(x) \\
 &= \underset{\theta}{\arg \max} \underset{i}\sum \log \int p_{\theta}(x|z)p(z)dz
\end{aligned}
$$

We can use any iterative optimization scheme to maximize the above objective. If we start  with some initial $$\theta$$, we have an estimate of $$p_{\theta}(x|z)$$ but we do not know $$p(z)$$. 

<!-- Unless we know the exact analytical expression for the integral, we can use any iterative optimi

In our case, assuming we know $$\log p_{\theta}(x|z)$$ for that iteration, the evaluation of the above integration requires estimating $$p(z)$$. Using Baye's rule $$p(z)= p(z|x) p(x)$$. Since we do not know $$p(x)$$ (called evidence), the above is intractable. Even if we made some assumption, it would requrie drawing a bunch of samples from z and finding the expectation $$\mathbb{E}_{z \sim p(z)}[p_{\theta}(x|z)]$$, or in simpler terms, evaluating that integral numerically. When computing the gradient of that objective w.r.t $$\theta$$, this would mean drawing the samples $$z$$ every time in the iterative optimization. In short,  -->