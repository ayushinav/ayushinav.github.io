It took me a while to grasp how to implement the continuous version, aka Integral Kernel Operator, so I'll also put down what I've done.
The general form of NO is

```math
\mathcal{G}_{\theta} = \mathcal{Q} \circ \sigma_T(W_{T-1} + \mathcal{K}_{T-1}+ b_{T-1}) \circ \dots \circ \sigma_1(W_0 + \mathcal{K}_0+ b_0) \circ \mathcal{P}
```

where 


```math
(\mathcal{K}_t(v_t))(x) = \int_{D_t} \kappa^{(t)}(x,y) v_t(y) dy \quad \forall x \in D_t
```

$\mathcal{P}$ and $\mathcal{Q}$ are local operators that lift and project. $W_t$ works similarly. Ignoring bias for now, for the continuous variant we'd want to be able to take inputs at any arbitrary points rather than fixed discretization, generally required by neural operators. 

The role of $\mathcal{P}$ is to lift the input discretization to a higher-dimension space. For point functions, this would just be another function, approximated by a network. It'd thus give $\mathcal{P}(a(x))$ as output where $a(x)$ is the input. For a 1D input and a 1D output, the input should be just a single node. 
$\mathcal{Q}_t \text{ and } W_t$ would also be constructed similarly, all three being networks that take input at one point and give an output at one point. As an example, they'd be 
`Chain(Dense(1 => 16), Dense(16 => 16), Dense(16 => 1))`

Constructing the kernel $\mathcal{K}_t$ requires some efforts, but has been implemented by a network approximating $\kappa^{(t)}$ that takes in 2 inputs, one of them being the input point and the other that will be used to compute the integral. Since the integral is computed on the domain $D_t$, we'd also need to compute the domains to evaluate $\mathcal{K}_t$. Though not the best way, right now, I'm passing the boundary points of the domain through the previous layers and sort them to get the next domains. This should work fine if the activation functions are monotonically increasing/ decreasing, and and the maps $\mathcal{P}_t, \mathcal{Q}_t$ and $W_t$ being linear transformation. Though I am doubtful how well this'd hold for $\mathcal{K}_t$.

Since we want to have the flexibility of using any network, these are implemented as `CompactLuxLayers` having the same functionality as `ContainerLayers` but concisely.