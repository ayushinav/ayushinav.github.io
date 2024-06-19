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
# Theory of variational inference 

Given the observed data variable $$x$$ and assuming some latent variable $$z$$, variational inference tries to approximate the posterior $$p(z|x)$$ with a family of known simpler distributions.
The posterior, as we know, is 

$$
p(z|x)= \frac{p(x,z)}{p(x)}= \frac{p(x|z)p(z)}{p(x)} \propto p(x|z)p(z)
$$

which, in the probablistic literature, is written as

$$
posterior \; distribution \propto likelihood * prior \; distribution
$$

KL-divergence is a popular metric to check the similarity between two distributions, say $$p_1(x)$$ and $$p_2(x)$$, as

$$
\begin{aligned}
KL(p_1(x)|| p_2(x))= \mathbb{E}_{x \sim p_1(x)}\left[\log \frac{p_1(x)}{p_2(x)}\right]
\end{aligned}
$$

 which in our case translates to

$$
\begin{aligned}
KL(q(z)|| p(z|x)) &= \mathbb{E}_{z \sim q(z)}\left[\log \frac{q(z)}{p(z|x)}\right] \\
&= \mathbb{E}_{z \sim q(z)}\left[\log q(z)\right]- \mathbb{E}_{z \sim q(z)}\left[\log p(z,x)\right] + \mathbb{E}_{z \sim q(z)}\left[\log p(x)\right] \\
&= \mathbb{E}_{z \sim q(z)}\left[\log q(z)\right]- \mathbb{E}_{z \sim q(z)}\left[\log p(z,x)\right] + \log p(x) \\
&(\log p(x) \text{ is marginal, therefore independent of sampling } z)
\end{aligned}
$$

In the above expression, the last term $$\log p(x)$$ is constant wrt $$x$$. Defining 

$$
ELBO(q)= \mathbb{E}_{z \sim q(z)}\left[\log p(z,x)\right]- \mathbb{E}_{z \sim q(z)}\left[\log q(z)\right]
$$
we can say that making the two distributions as similar as possible, implying minimizing $$KL(q(z)||p(z|x))$$ is equivalent to maximizing $$ELBO(q)$$ since the first term in the above equation is constant wrt $$z$$. Also, we can write

$$
\begin{aligned}
\log p(x) &= + ELBO(q)+ KL(q(z)|| p(z|x)) \\
&\ge ELBO(q)
\end{aligned}
$$

Therefore, as the KL divergence decreases, we have the $$ELBO(q)$$, called the *Evidence Lower Bound*, become a tighter lower bound to the marginal probability $$p(x)$$. 

## A better prespective

This allows us to look at problem from another approach, in terms of approximating $$p(x)$$. Now, let's have the observed data $$x$$ sampled from an unknown complex distribution 
$$p(x)$$. Assuming a latent variable $$z \sim p(z)$$, were $$p(z)$$ is some simple distribution and also assuming $$p(x|z)$$ is simple, we can compute $$p(x)$$ by 

$$
p(x)= \int p(x|z)p(z)dz
$$

Assuming $$p(x)$$ is parameterized on $$\theta$$ (as an example $$\theta$$ can represent the mean and variance of a normal distribution, a complex version can be called the parameters of a GMM, a linear combination of multiple gaussians), then it is easy to see that $$p(x|z)$$ will be parameterized on $$\theta$$ where $$\theta$$ might come from $$z$$, that is the various means and variance of the different gaussians, and probably also which gaussian to sample from. This means 
$$p_{\theta}(x)= \int p_{\theta}(x|z)p(z)dz$$
Now, our objective is to estimate $$p(x)$$, that is, the parameters of $$p(x)$$, that is $$\theta$$. By probabilistic regression, we want to maximize the log-likelihood $$ \log p(x)$$. 

$$
\begin{aligned}
 \therefore \theta &= \underset{\theta}{\arg \max} \underset{i}\sum \log p_{\theta}(x_i) \\
 &= \underset{\theta}{\arg \max} \underset{i}\sum \log \int p_{\theta}(x_i|z)p(z)dz
\end{aligned}
$$

In general, we need a lot of $$z$$'s to approximate the above integration well, making this approach intractable. Before we go into how to make this computationally feasible, let's try to modify the objective function a bit. We have

$$
\begin{aligned}
 \log p_{\theta}(x_i) &= \log \int p_{\theta}(x_i|z)p(z)dz \\
 & \text{For any probabilty } q(z) \text { we have}\\
 &= \log \int \frac{p_{\theta}(x_i|z)p(z) }{q(z)} q(z)dz \\
 &= \log \mathbb{E}_{z \sim q(z)} \left[ \frac{p_{\theta}(x_i|z)p(z)}{q(z)}\right] \\
 & \text{Using Jensen's inequality} \\
 & \ge \mathbb{E}_{z \sim q(z)} \left[\log \frac{p_{\theta}(x_i|z)p(z)}{q(z)}\right] \\
 &= \mathbb{E}_{z \sim q(z)} \left[\log \frac{p_{\theta}(x_i,z)}{q(z)}\right] \\
 &= \mathbb{E}_{z \sim q(z)} \left[\log p_{\theta}(x_i,z)\right] - \mathbb{E}_{z \sim q(z)} \left[\log q(z) \right] \\
 &= ELBO(q)
\end{aligned} 
$$

Now, we see more clearly and mathematically how $$ELBO(q)$$ provides a lower bound to 
$$\log p(x)$$. One can again do the same steps and see that minimizing $$KL(q(z)|| p(z|x))$$ leads to more tightening of the lower bound on the marginal probability $$p(x)$$.

# Training
So far, we can conclude that maximizing ELBO can be used as an optimization problem to minimize the KL divergence between the posterior and its approximation. We did not talk about how we can make this problem tractable. This is widely done using mean field approximation. Mean-field approximation assumes that each latent variable can be drawn independt of the other, that is, no two variables bear any correlation among themselves, that is, if we assumed a multivariate Gaussian for $$q(z)$$, then this approximation means that the covariance matrix is just a diagonal matrix. While this assumption limits the usage when there IS a correlation, that may be problem dependent and we tackle that later. Therefore, using the mean-field approximation, we have 

$$
q(z)= \underset{i}{\Pi}q_i(z_i)
$$

Coming back to the training part. We are solving the optimization problem:

$$
\begin{aligned}
\theta &= \underset{\theta}{\arg \max}\; ELBO(q) \\
&= \underset{\theta}{\arg \max} \; \mathbb{E}_{z \sim q(z)}\left[\log p_{\theta}(z,x)\right]- \mathbb{E}_{z \sim q(z)}\left[\log q(z)\right]
\end{aligned}
$$

## Given
A likelihood expression $$p(x|z)$$. In many cases, we assume a mixture model, typically GMMs, if all we are given is the observed data and we don't know any likelihood expression. 
## Initialization
* We assume an initial $$\theta$$.
* We assume a $$q_i(z)$$. For a gaussian family, this would imply that the parameters of $$q_i$$ are $$\mu_i$$ and $$\sigma_i$$.

## Iterations

* So given $$x$$'s, the first step is to sample $$x_i$$
* Compute the $$\nabla_{\theta}ELBO(q)$$ by:
    * Sample $$z$$ from $$q_i(z)$$. In practice, sampling just one $$z_i$$ works fine, but sampling more would lead to better accuracy. Also, because we mostly use Gaussian distributions, we have analytical expression. Not really sure here. 
    * Compute the Expectation for $$ELBO$$ and then its gradient. Computing gradient should be simple because we would have a likelihood term.
* Update $$\theta$$
* Compute the gradient of $$ELBO$$ wrt $$q_i$$, that is, $$\mu_i$$ and $$\sigma_i$$, using the already sampled $$z$$'s.
* Update $$q_i$$, that is, $$\mu_i$$ and $$\sigma_i$$.

Note that, we are sampling $$x_i$$ and updating $$q_i$$ for each $$x_i$$. This method of optimization is called coordinate ascent, where we optimize for each dimension independently. We can onbiously include mini-batches or update for all the parameters together, using simple gradient ascent.

## Amortizing

When we have a lot of samples, instead of having a $$q_i$$ for every sample, we can use a network/ any function approximator to output the $$\mu_i$$ and $$\sigma_i$$ for an $$x_i$$. Then we would update the parameters of the network $$\phi$$. 

We can build our intuition around the fact that $$q(z)$$ is trying to approximate 
$$p(z|x)$$, that is, much like $$p(x|z)$$ was parameterized on $$\theta$$, we parameterize $$p(z|x)$$ on $$\phi$$. Thus, if we also have a network for $$p_{\theta}(x|z)$$ that takes in $$z$$ to output the conditional probability for that $$z_i$$, we would have another network for $$p(z|x)$$ that takes $$x$$ to ouutput the conditional probability for that $$x_i$$. 

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

As is evident, we do not have to parameterize using $$\theta$$. We again assume $$q_{\phi}(m)$$ to approximate this posterior using variational inference. This we parameterize on $$\phi$$, which can be just the parameters of the family of distribution we are using to approximate the posterior, or the weights of the network. The iteration scheme, therefore, develops as:
* Sample $$d_i$$, or maybe choose the whole data vector $$d$$.
* Compute the $$\nabla_{\theta}ELBO(q)$$ by:
    * Sample $$m$$ from $$q_i(z)$$.
    * Compute the Expectation for $$ELBO$$ and then its gradient. Computing gradient should be simple because we would have a likelihood term.
* Compute the gradient of $$ELBO$$ wrt $$q_i$$, that is, $$\phi$$.
* Update $$q_i$$, that is, $$\phi$$.

## Constraints/ Structure?
While constraints are included in the *a priori* term, it is worth noting that the approach to a lot of geophysical inverse problems assume that the model parameters bear no correlation with each other. However, a decent approach would have a way to incorporate structure. The mean field approximation assumes that the model parameters do not bear any correlation with each other. Would it be possible to do that by including that in the regularizer, that is the *a priori*, similar to what happens in [Occam 1D](https://marineemlab.ucsd.edu/~steve/bio/Occam1D.pdf), and RTO-TKO? 

OR we still sample them independently, but use a mapping that enforces the structure that is then passed into the forward operator? Remember that constraints in deep learning are applied via a similar idea, eg., using the softmax function as the last layer of the neural network to output probabilities.


# References
* [Variational Inference: A Review for Statisticians](https://arxiv.org/abs/1601.00670#:~:text=One%20of%20the%20core%20problems,calculation%20involving%20the%20posterior%20density.)
* [Stanford CS330: Variational Inference and Generative Models: Lecture 11](https://youtu.be/iL1c1KmYPM0)
* [2021 3.1 Variational inference, VAE's and normalizing flows - Rianne van den Berg](https://youtu.be/-hcxTS5AXW0)