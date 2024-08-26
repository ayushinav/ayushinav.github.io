---
layout: post
title:  Variational Inference
date:   2023-11-09 16:40:16
description: Understanding variational inference and applying it to geophysical inversion
tags: probability inversion geophysics math
categories: 
toc:
  sidebar: left
---
## Problem statement

In geophysical inversion involving uncertainty quantification, we want to calculate the posterior 
<!--  -->
$$p(m|d)$$, where $$m$$ are the model parameters we want to invert for and $$d$$ are the observed data. Given the *a priori* distribution $$p(m)$$ and the likelihood function $$p(d|m)$$, the bayesian formulation is

$$
p(m|d) \propto p(d|m) p(d)
$$

An obvious candidate to the above problem is MCMC, which has asymptotic convergence guarantees. This however requires a large number of samples. Variational inference turns this inference problem into am optimization one.

# Theory of variational inference 

Variational Inference approximates the posterior 
<!--  -->
$$p(m|d)$$ with a distribution $$q(m)$$, parameterized on, say $$\phi$$. Note that $$q(m)$$ is still conditioned on $$d$$. One would be $$q(m) = \mathcal{N}(\mu(d), \sigma(d)^2)$$, where $$\mu$$ and $$\sigma$$ are the outputs from a network with $$d$$ as input and parameterized on $$\phi$$. We want $$q$$ to be as similar to $$p(m|d)$$. In other words, we want the KL divergence $$KL(q(m)|| p(m|d))$$ between the two to be as small as possible:

$$
\begin{aligned}
KL(q(m|d)|| p(m|d)) 
&= \mathbb{E}_{m \sim q(m|d)}\left[\log \frac{q(m|d)}{p(m|d)}\right] \\
&= \mathbb{E}_{m \sim q(m|d)}\left[\log q(m|d)\right]- \mathbb{E}_{m \sim q(m)}\left[\log p(d| m)\right] - \mathbb{E}_{m \sim q(m)} \left[\log p(m)\right] + \mathbb{E}_{m \sim q(m)}\left[\log p(d)\right] \\
&= \mathbb{E}_{m \sim q(m|d)}\left[\log q(m|d)\right]- \mathbb{E}_{m \sim q(m)}\left[\log p(d| m)\right] - \mathbb{E}_{m \sim q(m)} \left[\log p(m)\right] + \log p(m) \\
& (\log p(m) \text{ is marginal and independent of sampling } m)
\end{aligned}
$$

The first three terms on the right hand side constitute the Evidence Lower Bound, or ELBO, defined as:

$$
\begin{aligned}
& \operatorname{ELBO}(q) = \mathbb{E}_{m \sim q(m|d)}\left[\log p(m, d)\right] + \mathbb{E}_{m \sim q(m)}\left[\log p(m)\right] - \mathbb{E}_{m \sim q(m)}\left[\log q(m)\right] \\
\implies & KL(q(m|d)|| p(m|d)) + \operatorname{ELBO}(q) = \log p(m) = \text{Constant}
\end{aligned}
$$

Therefore, maximizing ELBO minimizes the KL divergence. Minimizing ELBO can be done in the following steps:
* Sample $$d_i \sim p(d)$$
* Get $$ELBO(q)$$ by sampling $$m \sim q(m|d)$$
* Calculate $$\nabla_{\phi} \operatorname{ELBO}(q)$$
* Update $$\phi$$ using gradient ascent


## Going beyond
The theory of variational inference was mostly developed to obtain a parameteric distribution of the observed data variable $$d$$ and assuming some latent variable $$m$$. The marginal distribution is

$$
p(d) = \int p(d|m) p(m) dm 
$$

In this case, we do not know what 
<!--  -->
$$p(d|m)$$ looks like, and let's say it's parameterized on $$\theta$$. Let's have $$q(m)$$ again but this time it is to be used as a proposal distribution. For any sample $$d_i$$, estimating the likelihood takes the form

$$
\begin{aligned}
 \log p_{\theta}(d_i) &= \log \int p_{\theta}(d_i|m)p(m)dm \\
 & \text{For any probabilty } q(m) \text { we have}\\
 &= \log \int \frac{p_{\theta}(d_i|z)p(m) }{q(m)} q(m)dm \\
 &= \log \mathbb{E}_{m \sim q(m)} \left[ \frac{p_{\theta}(d_i|m)p(m)}{q(m)}\right] \\
 & \text{Using Jensen's inequality} \\
 & \ge \mathbb{E}_{m \sim q(m)} \left[\log \frac{p_{\theta}(d_i|m)p(m)}{q(m)}\right] \\
 &= \mathbb{E}_{m \sim q(m)} \left[\log \frac{p_{\theta}(d_i|m)p(m)}{q(m)}\right] \\
 &= \mathbb{E}_{m \sim q(m)} \left[\log p_{\theta}(d_i|m)\right] + \mathbb{E}_{m \sim q(m)} \left[p(m)\right] - \mathbb{E}_{m \sim q(m)} \left[\log q(m) \right] \\
 &= ELBO(q)
\end{aligned} 
$$

THus, the same ELBO becomes a lower bound to the marginal distribution of the observed variable. We now have to optimize with respect to $$\theta$$, and minimizing ELBO has one more step:
* Sample $$d_i \sim p(d)$$
* Get $$ELBO(q)$$ by sampling $$m \sim q(m|d)$$
* Calculate $$\nabla_{\phi} \operatorname{ELBO}(q)$$
* Update $$\phi$$ using gradient ascent
* Calculate $$\nabla_{\theta} \operatorname{ELBO}(q)$$
* Update $$\theta$$ using gradient ascent

## $$\nabla_\theta \operatorname{ELBO}(q)$$

While all the steps looked fairly simple until now, there are a few nitty gritties. For a sample 
* Sample $$d_i \sim p(d)$$
$$d_i \sim p(d)$$, we pass it throught the network to obtain $$q_\phi(m|d)$$, eg., the network outputs the mean and variance of a normal distribution. Before any optimization, $$\phi$$ is initialized in any way. 
* Get $$ELBO(q)$$ by sampling $$m \sim q(m|d)$$
Sampling a few samples from $$q(m|d)$$, the expectaion can be estimated. 
* Calculate $$\nabla_{\phi} \operatorname{ELBO}(q)$$
* Update $$\phi$$ using gradient ascent
* Calculate $$\nabla_{\theta} \operatorname{ELBO}(q)$$
We note that only the first term here is dependent on $$\theta$$. $$p(m)$$ is the *a priori* distribution of $$m$$ and $$q(m|d)$$ is parameterized on $$\phi$$. For one sample, say $$m^*$$, we pass it through the network to obtain the distribution $$p_\theta(d|m)$$, we can easily differentiate through $$\theta$$, using backpropagation. 
* Update $$\theta$$ using gradient ascent

## $$\nabla_\phi \operatorname{ELBO}(q)$$
Most of the steps are similar as above until we get to the point where we have to calculate 
$$\nabla_\phi \operatorname{ELBO}(q)$$. The first thing we notice is we have to differentiate through the sampling process. To compute this, we make use of the reparameterization trick. This means we assume another variable $$\tilde{m}$$ such that we can write

$$
m \sim q_\phi(m) \Longleftrightarrow m = g_{\phi}(\tilde{m}) \quad \text{where} \quad \tilde{m} \sim h(\tilde{m})
$$

This is better understood via an example. Let $$q(m)$$ be $$\mathcal{N}(\mu, \sigma^2)$$, then defining $$ \tilde{m} \sim \mathcal{N}(0, 1)$$

$$
m \sim \mathcal{N}(\mu, \sigma^2) \Longleftrightarrow m = \mu + \sigma \tilde{m} \quad \text{where} \quad \tilde{m} \sim \mathcal{N}(0, 1)
$$

This way, we can sample from $$\tilde{m}$$ to get the samples from the same distribution as $$m$$. We, therefore, sample $$\tilde{m}$$ and not $$m$$. The ELBO takes the form

$$
\begin{aligned}
\operatorname{ELBO}{q} 
&= \mathbb{E}_{m \sim q(m)}\left[ logp_\theta{d_i|m} + p(m) - log q(m) \right] \\
&= \mathbb{E}_{\tilde{m} \sim h(\tilde{m})}\left[ logp_\theta{d_i|g_{\phi}(\tilde{m})} + p(g_{\phi}(\tilde{m})) - log q(g_{\phi}(\tilde{m})) \right] \\
\end{aligned}
$$

We can now differentiate through the expression w.r.t $$\phi$$, in the similar way as we did for $$\theta$$. To summarize, the training process transforms into:

* Sample $$d_i \sim p(d)$$ \\
$$d_i \sim p(d)$$, we pass it throught the network to obtain $$q_\phi(m|d)$$. Sample from $$q_{\phi}(m)$$ using the reparameterization trick.
* Get $$ELBO(q)$$ by sampling $$m \sim q(m|d)$$ \\
Sampling a few samples from $$q(m|d)$$, the expectaion can be estimated. 
* Calculate $$\nabla_{\phi} \operatorname{ELBO}(q)$$ \\
Get the gradient using backpropagation using the reparameterization trick.
* Update $$\phi$$ using gradient ascent
* Calculate $$\nabla_{\theta} \operatorname{ELBO}(q)$$ \\
We note that only the first term here is dependent on $$\theta$$. $$p(m)$$ is the *a priori* distribution of $$m$$ and $$q(m|d)$$ is parameterized on $$\phi$$. For one sample, say $$m^*$$, we pass it through the network to obtain the distribution $$p_\theta(d|m)$$, we can easily differentiate through $$\theta$$, using backpropagation. 
* Update $$\theta$$ using gradient ascent

## For geophysical inversion
The implementaion for probabilistic geophysical inverse problems is fairly similar. In this domain, we have the observed data $$d$$ and its error bars, generally the standard deviation associated with each data point.

$$
\begin{aligned}
p(m|d) & \propto p(d|m)p(m) \\
posterior \; distribution & \propto likelihood * prior \; distribution
\end{aligned}
$$

The likelihood is generally the negative exponential of the misfit between the observed data and the forward response from a given model, that is 

$$
p(d|m)= \exp \left[- (\mathcal{F}(m)- d)^T W (\mathcal{F}(m)- d) \right]
$$

where $$\mathcal{F}$$ is the forward model operator and $$W$$ is the weight matrix, usually the inverse of the variance in the data. The prior distribution is something where the *a priori* knowledge comes in. If in the misfit term, we were supposed to have a regularizer, say $$\| m - m_0\|^2$$ for a reference model $$m_0$$, then it manifests in the *a priori* distribution as

$$
p(m)= \exp \left[- \| m - m_0\|^2 \right]
$$

Which is just a scaled gaussian with unit variance. The posterior then becomes

$$
p(m|d) \propto \exp \left[- (\mathcal{F}(m)- d)^T W (\mathcal{F}(m)- d) \right] \exp \left[- \| m - m_0\|^2 \right]
$$

As is evident, we do not have to parameterize using $$\theta$$. We again assume $$q_{\phi}(m)$$ to approximate this posterior using variational inference. This we parameterize on $$\phi$$, which can be just the parameters of the family of distribution we are using to approximate the posterior, or the weights of the network. 

## Constraints/ Structure?
While constraints are included in the *a priori* term, it is worth noting that the approach to a lot of geophysical inverse problems assume that the model parameters bear no correlation with each other. However, a decent approach would have a way to incorporate structure. The mean field approximation assumes that the model parameters do not bear any correlation with each other. Would it be possible to do that by including that in the regularizer, that is the *a priori*, similar to what happens in [Occam 1D](https://marineemlab.ucsd.edu/~steve/bio/Occam1D.pdf), and RTO-TKO? 

OR we still sample them independently, but use a mapping that enforces the structure that is then passed into the forward operator? Remember that constraints in deep learning are applied via a similar idea, eg., using the softmax function as the last layer of the neural network to output probabilities.


# References
* [Variational Inference: A Review for Statisticians](https://arxiv.org/abs/1601.00670#:~:text=One%20of%20the%20core%20problems,calculation%20involving%20the%20posterior%20density.)
* [Stanford CS330: Variational Inference and Generative Models: Lecture 11](https://youtu.be/iL1c1KmYPM0)
* [2021 3.1 Variational inference, VAE's and normalizing flows - Rianne van den Berg](https://youtu.be/-hcxTS5AXW0)