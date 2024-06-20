---
layout: post
title:  Probability
date:   2023-07-06 16:40:16
description: Because basics are important
tags: probability math 
categories: 
toc:
  sidebar: left
---
# Definitions
For discrete random variable $$x$$ i.i.d. from $$X= \{ x_1, x_2, \cdots x_n\}$$, the probability mass function (PMF) is 

$$
f:\mathrm{R} \rightarrow [0,1] : \sum_i^n f(x_i)= 1 = \sum_{x \in X} f(x)
$$

The cumulative distribution function (CDF) is then:

$$
F_X(x)= \mathrm{P}(X \le x)= \sum_{i:x_i<x} f(x)
$$

A random variable is continuous if its CDF is:
$$
F_X(x)= \mathrm{P}(X \le x)= \int_{-\infty}^x f(u) du \quad \texttt{and} \quad f(x):\mathrm{R} \rightarrow [0, \infty] \quad \texttt{and} \quad \int_{-\infty}^{\infty} f(x)= 1
$$

where $$f(x)$$ is the probability density function (PDF) of $$x$$.
Then 

$$
\mathrm{P}(a \le X \le b)= \int_a^b f(x) dx
$$

Note that for a continuous variable $$X$$, $$\mathrm{P}(X)= 0 \: \forall \quad X \in \mathrm{R}$$, in that case  

$$\mathrm{P}(x \le X \le x+dx)= f(x) dx
$$

can give sense of point probability of the continuous variable.

# Expectation
Discrete case: 

$$
\mathrm{E} \{X\} = \sum_{x\in X}x \mathrm{P}(x)
$$

Continuous case: 

$$
\mathrm{E} \{X\}= \int_{-\infty}^{\infty} x f(x) dx
$$

In continuous case, expectation of $$g(x)$$ with $$x$$ i.i.d $$f(x)$$ is (replace $$x$$ with $$g(x)$$ in the previous eqn)

$$
\mathrm{E} \{ g(x)\}= \int_{-\infty}^{\infty} g(x) f(x) dx
$$

# Joint probability
Probability of all the events across all the variables occurring together (Imagine the independent case for intuition)

Discrete  case: 

$$
\mathrm{P}(x,y)= \mathrm{P}(x|y)\mathrm{P}(y)= \mathrm{P}(y|x)\mathrm{P}(x) \quad \texttt{(Chain rule or Baye's rule)}
$$

When they are independent, $$ \mathrm{P}(x,y)= \mathrm{P}(x) \mathrm{P}(y)$$

Continuous case: When the variables are independent, the joint pdf $$f_{XY}(x,y)$$ cannot be expressed in separable terms, when they are independent, the joint pdf can be simplified as $$f_{XY}(x,y)= f_X(x) f_Y(y)$$, i.e., two separate pdfs of different variables.

# Marginal probability
Probability of event $$x$$ for all outcomes of $$Y$$.

Discrete case: 

$$
\mathrm{P}(x)= \sum_{y \in Y} \mathrm{P}(x, y)= \sum_{y \in Y} \mathrm{P}(x|y)\mathrm{P}(y)
$$

OR

$$
\mathrm{P}(x)= \sum_{i} \mathrm{P}(x, y_i)= \sum_{i} \mathrm{P}(x|y_i)\mathrm{P}(y_i)
$$

Continuous case: 

$$
f_X(x)= \int_{-\infty}^{\infty} f_{XY}(x,y)dy
$$

# Condition probability
Probability of x happening given y happens 

Discrete case: 

$$
\mathrm{P}(x|y)= \frac{\mathrm{P}(x,y)}{p(y)}
$$

Continuous case: 

$$f_{X|Y}(x|Y)= \frac{f_{XY}(x,y)}{f_Y(y)}
$$

# Notation
$$X,Y$$ are the domains, $$x,y$$ are the specific points in those domains, the variables we work on. $$P(x)$$ means $$P(X= x)$$ 

CDF is just the sum of all the probabilities of all the points smaller than the current point. It's curve is monotonically increasing. \newline
Discrete case: 

$$ F_X(x)= \mathrm{P}(X \le x)= \sum_{i:x_i<x} f(x)$$

Continuous case: 

$$F_X(x)= \int_{-\infty}^x f(u) du \quad \texttt{OR} \quad f(x)= \frac{d}{dx} F(x)$$


