## Schrödinger Bridge for Time Series (SBTS)

Let $\mu$ be the distribution of a time series valued in $\mathbb{R}^d$,
observed over a discrete time grid $T = \{t_1, \cdots, t_N = T\}$.
We construct a model generating time series samples following
$\mu \in \mathcal{P}((\mathbb{R}^d)^N)$ given real observations.

The **Schrödinger Bridge Time Series (SBTS)** problem is formulated as:

$$\min_{\alpha} \text{KL}(\mathbb{P} || \mathbb{W}^\sigma ) = \min_{\alpha}  \frac{1}{2} \mathbb{E}_{\mathbb{P}} \left[ \int_0^T \|\frac{\alpha_t}{\sigma}\|^2 \, dt \right]$$

such that $dX_t = \alpha_t dt + \sigma dW_t^{\mathbb{P}}$ with $W$ a Brownian motion
under $\mathbb{P}$, $X_0 = \mathbf{0}$, and
$(X_{t_1}, \cdots, X_{t_N}) \overset{\mathbb{P}}{\sim} \mu$.
