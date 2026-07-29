## Schrödinger Bridge for Time Series (SBTS)

Let $\mu$ be the distribution of a time series valued in $\mathbb{R}^d$,
observed over a discrete time grid $`0 = t_0 < t_1 \cdots < t_N = T`$
We construct a model generating time series samples following
$\mu \in \mathcal{P}((\mathbb{R}^d)^N)$ given real observations.

The **Schrödinger Bridge Time Series (SBTS)** problem is formulated as:

$$\min_{\alpha} \text{KL}(\mathbb{P} | \mathbb{W}^\sigma ) = \min_{\alpha}  \frac{1}{2} \mathbb{E}_{\mathbb{P}} \left[ \int_0^T \bigg|\frac{\alpha_t}{\sigma}\bigg|^2 \, dt \right]$$

such that $dX_t = \alpha_t dt + \sigma dW_t^{\mathbb{P}}$ with $W$ a Brownian motion
under $\mathbb{P}$, $X_0 = \mathbf{0}$, and
$(X_{t_1}, \cdots, X_{t_N}) \overset{\mathbb{P}}{\sim} \mu$. Once we learn the optimal drift $\alpha^\star$, we can generate new time series samples via $\mathbb{P}^\star$: $dX_t = \alpha_t^\star dt + \sigma dW_t^{\mathbb{P}^\star}$ 

------
## Deep Hedging

We consider the deep hedging of an ATM call option with payoff
$`g(S_T) = (S_T - S_0)^+`$.

The goal is to minimize over the initial capital $`p`$ (premium) and
the parameters of the neural network $`\Delta`$ the **replication error**:

```math
\min_{p, \Delta} \, \mathbb{E} [\left| \text{PnL}^{p, \Delta} \right|]^2
```

where the Profit and Loss is defined as:

```math
\text{PnL}^{p, \Delta} = p + \sum_{i=0}^{N-1} \Delta(t_i, S_{t_i})(S_{t_{i+1}} - S_{t_i}) - g(S_T)
```

where $`p`$ is the initial premium collected, $`\Delta(t_i, S_{t_i})`$ is the
hedge ratio (number of shares held) at time $`t_i`$, and $`g(S_T) = (S_T - S_0)^+`$
is the option payoff at maturity.


Think of this as:

> **(money you have) = (money you started with) + (money you made/lost trading) − (money you have to pay out)**

