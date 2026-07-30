## Schrödinger Bridge for Time Series (SBTS)

Let $\mu$ be the distribution of a time series valued in $\mathbb{R}^d$,
observed over a discrete time grid $`0 = t_0 < t_1 \cdots < t_N = T`$
We construct a model generating time series samples following
$\mu \in \mathcal{P}((\mathbb{R}^d)^N)$ given real observations.

The **Schrödinger Bridge Time Series (SBTS)** problem is formulated as:

$$\min_{\alpha} \text{KL}(\mathbb{P} | \mathbb{W}^\sigma ) = \min_{\alpha}  \frac{1}{2} \mathbb{E}_{\mathbb{P}} \left[ \int_0^T \bigg|\frac{\alpha_t}{\sigma}\bigg|^2 \, dt \right]$$

such that $dX_t = \alpha_t dt + \sigma dW_t^{\mathbb{P}}$ with $W$ a Brownian motion
under $\mathbb{P}$, $X_0 = \mathbf{0}$, and
$(X_{t_1}, \cdots, X_{t_N}) \overset{\mathbb{P}}{\sim} \mu$. Once we learn the optimal drift $\alpha^\star$, we can generate new time series samples via $\mathbb{P}^\star$: $dX_t = \alpha_t^\star dt + \sigma dW_t^{\mathbb{P}^\star}$.

## Deep Hedging

We consider the deep hedging of an ATM call option with payoff 
$`g(S_T) = (S_T - S_0)^+`$.
The goal is to minimize over the initial capital $`p`$ (premium) and
the parameters of the neural network $`\Delta`$ the **replication error**:

```math
\min_{p, \Delta} \, \mathbb{E} [\left| \text{PnL}^{p, \Delta} \right|^2]
```

where the Profit and Loss is defined as:

```math
\text{PnL}^{p, \Delta} = p + \sum_{i=0}^{N-1} \Delta(t_i, S_{t_i})(S_{t_{i+1}} - S_{t_i}) - g(S_T)
```

where $`p`$ is the initial premium collected, $`\Delta(t_i, S_{t_i})`$ is the
hedge ratio (number of shares held) at time $`t_i`$, and $`g(S_T) = (S_T - S_0)^+`$
is the option payoff at maturity.
Think of this as: **(money you have) = (money you started with) + (money you made/lost trading) − (money you have to pay out)**.

A tighter PnL distribution centered near zero indicates a more accurate hedge, as it reflects lower replication error and reduced sensitivity to model misspecification.

## SBTS for Deep Hedging

We use **SBTS to generate synthetic time series samples** of asset prices,
which are then used to train the deep hedging model via a Deep Neural Network.

```math
\text{Real price paths} \xrightarrow{\text{SBTS}} \text{Synthetic scenarios} \xrightarrow{\text{Deep Hedging}} \text{Optimal hedge } \Delta
```

## Results — Google (GOOGL) ATM Call Hedging

- Mean of PnL and its Std (replication error).

| Model | Premium | Training Set | Validation Set | Test Set |
| :--- | :--- | :--- | :--- | :--- |
| **Data** | 0.0413 | -0.002271 (0.014906) | -0.016039 (**0.014768**) | -0.014327 (0.016627) |
| **SBTS** | 0.0422 | -0.001897 (0.014832) | **-0.015814** (0.014964) | **-0.012974** (**0.014302**) |

Deep hedging model trained on Schrödinger Bridge synthetic samples, achieved **14\% lower replication error (PnL std)** and **reduced mean PnL bias by 9\%** vs. real-data-trained baseline **on out-of-sample backtests (test set)**.

- Empirical PnL distribution of the deep hedging
strategy trained on SBTS-generated scenarios vs real GOOGL price data.

![PnL GOOGL](https://raw.githubusercontent.com/vuminh-hoangd/Deep-Hedging-SBTS/main/PnL-googl-test.png)

## Repository Structure

<details>
<summary><b>Click to expand full repository structure</b></summary>

```text
.
├── DeepHedging/
│   ├── DataGenerator.py
│   └── DeepHedging.py
├── metrics/
│   ├── fbm_stock_metrics/
│   │   ├── configs/
│   │   │   ├── config.py
│   │   │   ├── evaluation_config_fBM.yaml
│   │   │   └── evaluation_config_Stock.yaml
│   │   ├── data/
│   │   │   ├── X_fBM.pt
│   │   │   └── X_stock.pt
│   │   └── src/
│   │       ├── evaluations/
│   │       │   ├── augmentations.py
│   │       │   ├── eval_helper.py
│   │       │   ├── evaluations.py
│   │       │   ├── hypothesis_test.py
│   │       │   ├── loss.py
│   │       │   ├── metrics.py
│   │       │   ├── plot.py
│   │       │   ├── scores.py
│   │       │   ├── summary.py
│   │       │   └── test_metrics.py
│   │       ├── base.py
│   │       ├── train_regressor.py
│   │       └── utils.py
│   ├── discriminative_score.py
│   ├── eval_functions.py
│   ├── get_params.py
│   └── predictive_score.py
├── models/
│   ├── hyperparams_selection/
│   │   ├── markovian_optimal_multi.py
│   │   └── markovian_optimal_uni.py
│   ├── cv_sbts.py
│   ├── sbts_multi.py
│   ├── sbts_multi_markovian.py
│   ├── sbts_uni.py
│   └── sbts_uni_markovian.py
├── models_weights/
│   ├── aapl_data.weights.h5
│   ├── aapl_sbts_mark.weights.h5
│   ├── aapl_sbts_plain.weights.h5
│   ├── googl_data.weights.h5
│   ├── googl_sbts_mark.weights.h5
│   ├── googl_sbts_plain.weights.h5
│   ├── nvda_data.weights.h5
│   ├── nvda_sbts_mark.weights.h5
│   ├── nvda_sbts_plain.weights.h5
│   ├── spy_data.weights.h5
│   ├── spy_sbts_mark.weights.h5
│   └── spy_sbts_plain.weights.h5
├── notebooks/
│   ├── DeepHedging(APPL).ipynb
│   ├── DeepHedging (GGL).ipynb
│   ├── DeepHedging (NVDA).ipynb
│   └── DeepHedging (SPY).ipynb
├── utils/
│   ├── data_generation.py
│   ├── data_loading.py
│   ├── data_preprocessing.py
│   └── data_stationarity.py
├── requirements.txt
└── PnL-googl-test.png
```

</details>

---

## References

1. **Nonparametric generative modeling for time series via Schrödinger bridge**  
   Mohamed Hamdouche, Pierre Henry-Labordère, and Huyên Pham.  
   *Journal of Machine Learning Research (JMLR)*, 27(112):1–23, 2026.  
   [Link to paper](https://www.jmlr.org/papers/volume27/23-1162/23-1162.pdf)

2. **Robust time series generation via Schrödinger Bridge: a comprehensive evaluation**  
   Alexandre Alouadi, Baptiste Barreau, Laurent Carlier, and Huyên Pham.  
   *Proceedings of the 6th ACM International Conference on AI in Finance (ICAIF '25)*, 2025.  
   [arXiv:2503.02943](https://arxiv.org/abs/2503.02943)
