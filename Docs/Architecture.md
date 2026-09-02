# Hive: System Architecture

## 1. Executive overview

Hive is a shared-parameter, multi-asset neural portfolio allocator. It processes a separate twenty-day feature sequence for each asset, evolves thirty interacting latent strategy states on a ring topology, converts those states into asset scores, and applies a temperature-scaled softmax to obtain portfolio weights.

The architecture is deliberately asymmetric in its information flow:

- **Across assets:** parameters are shared, but hidden states are independent.

- **Across latent strategies:** states interact locally through a five-node ring neighborhood.

- **Across time:** each asset-strategy state is updated sequentially with leaky recurrence.

- **Across strategies at readout:** a global attention mechanism selects a shared mixture of strategy signals.

- **Across portfolio assets:** a final softmax turns relative scores into long-only, fully invested weights.

The supplied implementation uses six volatility-related assets:

| Asset index | Ticker |
| --- | --- |
| 1 | `^VIX` |
| 2 | `UVXY` |
| 3 | `SVXY` |
| 4 | `VXX` |
| 5 | `VIXY` |
| 6 | `VIXM` |

The system is a research prototype. It should not be interpreted as a production trading system or investment recommendation.

## 2. End-to-end architecture

```
Yahoo Finance daily closes
          |
          v
+----------------------------+
| Per-asset feature builder  |
| log return, RSI, bands     |
+----------------------------+
          |
          v
+----------------------------+
| Expanding per-window      |
| StandardScaler            |
+----------------------------+
          |
          v
+----------------------------+
| 20-day tensor              |
| A x L x d_x = 6 x 20 x 4 |
+----------------------------+
          |
          v
+--------------------------------------------------+
| Shared NeuronGroup, executed independently/a     |
|                                                  |
|  input projection                                |
|       +                                          |
|  recurrent projection                            |
|       +                                          |
|  adaptive ring coupling                          |
|       |                                          |
|  tanh candidate                                  |
|       |                                          |
|  leaky state update                              |
+--------------------------------------------------+
          |
          v
+----------------------------+
| Final local strategy scores |
| 30 scores per asset        |
+----------------------------+
          |
          v
+----------------------------+
| Global strategy attention  |
| one shared alpha over 30    |
+----------------------------+
          |
          v
+----------------------------+
| Asset score vector         |
| 6 scores per sample        |
+----------------------------+
          |
          v
+----------------------------+
| Temperature softmax        |
| portfolio weights          |
+----------------------------+
          |
          v
+----------------------------+
| Weighted next-day return  |
| Sharpe - concentration    |
+----------------------------+
```

## 3. Runtime data contract

The forward method receives a four-dimensional tensor:

$$
X\in\mathbb{R}^{B\times A\times L\times d_x}
=\mathbb{R}^{B\times6\times20\times4}.
$$

The dimensions are ordered as batch, asset, time, and feature. The target tensor has shape

$$
Y\in\mathbb{R}^{B\times A}=\mathbb{R}^{B\times6},
$$

where $$Y_{b,a}$$ is the next-day simple return for asset $$a$$ at sample $$b$$.

The output is

$$
W=\mathrm{Hive}(X)\in\mathbb{R}^{B\times A}.
$$

Every row of $$W$$ is a simplex vector. Therefore, the forward pass implements a long-only, fully invested allocation:

$$
W_{b,a}>0,
\qquad
\sum_{a=1}^{A}W_{b,a}=1.
$$

## 4. Data ingestion and feature layer

### 4.1 Market data assembly

`build_multivol_dataset` downloads daily close prices for all configured tickers, joins them by date, and drops any date with a missing close in any asset. This creates a common calendar. The common-calendar requirement ensures that each training sample has a synchronized cross-asset target vector.

For each valid forecast origin $$i$$, the dataset builder creates:

| Object | Shape | Meaning |
| --- | --- | --- |
| `seq` | $$6\times20\times4$$ | Historical feature window for all assets |
| `ret` | $$6$$ | Next-day simple return for all assets |
| `X_list[i]` | $$6\times20\times4$$ | Tensor input after normalization |
| `y_list[i]` | $$6$$ | Tensor target |

The first usable origin is after the twenty-day lookback. The final available day is excluded from the origins because it has no next-day target in the downloaded range.

### 4.2 Feature builder

`compute_asset_features` transforms one close-price series into four columns:

| Column | Definition | Architectural role |
| --- | --- | --- |
| Log return | $$\log(C_t/C_{t-1})$$ | Short-horizon price movement |
| Normalized RSI | RSI divided by $$100$$ | Relative upward/downward pressure |
| Upper-band deviation | Distance from upper Bollinger band | Relative overextension |
| Lower-band deviation | Distance from lower Bollinger band | Relative underextension |

Feature construction occurs before window extraction. The model therefore sees a sequence of engineered observations rather than raw prices.

### 4.3 Expanding normalization

For every sample and asset, the implementation flattens the historical feature windows available up to that sample, fits a new `StandardScaler`, and transforms the current twenty-row window. The resulting scale is asset-specific and feature-specific.

This layer has two architectural properties. First, it is outside the neural network and has no trainable gradient. Second, it is stateful with respect to chronology because the set of rows used by the scaler grows as the dataset index advances.

## 5. Hive neural core

### 5.1 Shared `NeuronGroup`

`HiveVolPortfolio` owns one instance of `NeuronGroup`. That instance is reused for every asset. The group contains the complete latent strategy bank:

| Component | Input/output | Purpose |
| --- | --- | --- |
| `W_in` | $$4\to32$$ | Embed current asset features |
| `W_rec` | $$32\to32$$ | Transform each strategy’s prior state |
| `C_left` | $$32\to32$$ | First coupling transform |
| `C_right` | $$32\to32$$ | Second coupling transform |
| `u` | $$64$$ vector | Score self/neighbor state pairs |
| `readout_weight` | $$30\times32$$ | Convert strategy states to scores |
| `readout_bias` | $$30$$ | Strategy score offsets |
| `W_q` | $$32\to16$$ | Build global attention query |
| `W_k` | $$32\to16$$ | Build strategy keys |
| `b` | $$32$$ | Recurrent preactivation bias |

No asset-specific parameter copy exists. This encourages the same latent strategy vocabulary to be reused across the entire volatility universe.

### 5.2 Per-asset state bank

At the beginning of each forward pass, the model allocates

$$
H^{(a)}(0)\in\mathbb{R}^{B\times30\times32}
$$

of zeros for each asset $$a$$. The state list contains six independent tensors. At each timestep, the state for asset $$a$$ is updated using only the feature slice for asset $$a$$ and the prior state for that same asset.

The state memory for all assets is therefore conceptually

$$
H(0)\in\mathbb{R}^{B\times A\times N\times d}
=\mathbb{R}^{B\times6\times30\times32},
$$

although the implementation stores it as a Python list of six tensors rather than one four-dimensional tensor.

### 5.3 Sequential execution order

The loops are ordered by time first and asset second:

```python
for t in range(seq_len):
    for a in range(n_assets):
        h_new, w_local = group(x_seq[:, a, t, :], h_list[a])
```

This ordering has no mathematical effect on independent assets because each asset state is updated from its own prior state. It does define the execution schedule and makes the temporal recurrence explicit.

## 6. Ring-coupling subsystem

The thirty latent strategies are indexed on a circular graph. Each node receives five positions: two neighbors on the left, itself, and two neighbors on the right. The modulo operation means node $$0$$ is adjacent to nodes $$28$$ and $$29$$, while node $$29$$ is adjacent to nodes $$0$$ and $$1$$.

At each time step, the coupling subsystem performs four operations:

1. It gathers the five prior hidden states for every strategy.

1. It concatenates each gathered state with the receiving strategy’s own state.

1. It applies a learned scalar compatibility score and a neighborhood softmax.

1. It transforms neighbor states through two linear maps and multiplies the results elementwise.

The output is a $$B\times30\times32$$ tensor. It is added to the input and recurrent signals before the nonlinear state update.

This subsystem is local in graph distance but adaptive in weighting. It is not a conventional graph convolution with fixed adjacency weights, because the mixture coefficients depend on the current hidden states.

## 7. Recurrent update subsystem

For each asset, the input projection is broadcast across all thirty strategies:

$$
\mathrm{InputSignal}^{(a)}_t
=\mathbf{1}_{N}\otimes W_{\mathrm{in}}x^{(a)}_t.
$$

The recurrent signal is strategy-specific:

$$
\mathrm{RecurrentSignal}^{(a)}_t
=W_{\mathrm{rec}}H^{(a)}_{t-1}.
$$

The three signals are combined with the bias and the local coupling tensor. A hyperbolic tangent bounds the candidate state coordinatewise. A leaky interpolation then carries information forward across the sequence.

The recurrence provides temporal memory without using an explicit Transformer over the twenty timesteps. The same weights are applied at every position, so relative time is represented by state evolution rather than by a learned position embedding.

## 8. Readout subsystem

### 8.1 Strategy-level readout

After the final timestep, each state row is projected onto its learned strategy readout vector. This yields a raw score tensor

$$
P\in\mathbb{R}^{B\times A\times N}.
$$

The final local coupling weights are then used to aggregate neighboring raw scores, producing

$$
L\in\mathbb{R}^{B\times A\times N}.
$$

The distinction is important: coupling influences the hidden state throughout the sequence, and it also directly mixes the final strategy scores at readout.

### 8.2 Shared global strategy attention

The attention subsystem creates one query from the mean of all final hidden states in the batch and across assets. Keys are created from strategy states after averaging across the batch dimension. The resulting attention vector has shape

$$
\alpha\in\mathbb{R}^{N}=\mathbb{R}^{30}.
$$

The same $$\alpha$$ is applied to every batch item and asset. The final asset score matrix therefore results from broadcasting $$\alpha$$ over the local-score tensor:

$$
S_{b,a}=\sum_{n=1}^{30}\alpha_nL_{b,a,n}.
$$

This is a global coordination mechanism, not a separate attention decision for every asset. It selects a common strategy mixture based on the current batch-level context.

## 9. Portfolio head

The six asset scores are divided by temperature $$T=0.1$$ and passed through a softmax. This produces the output tensor

$$
W\in\mathbb{R}^{B\times6}.
$$

The portfolio head contains no explicit cash asset. It cannot abstain or hold zero total exposure. It always distributes 100% of portfolio mass among the six configured assets.

Because softmax is invariant to adding the same constant to every asset score,

$$
\mathrm{softmax}(S/T)
=\mathrm{softmax}((S+c\mathbf{1})/T),
$$

only relative asset scores affect allocation. The absolute score level has no direct portfolio meaning.

## 10. Training architecture

The training script uses a chronological split:

| Split | Range | Use |
| --- | --- | --- |
| Train | First 70% of windows | Parameter optimization |
| Validation | Next 15% | Model selection by Sharpe |
| Test | Final 15% | Final evaluation |

Training examples are shuffled within each epoch and batched in groups of 64. The model receives an entire batch because the Sharpe objective is computed across the batch. This means a batch is not merely a memory-efficiency device; it is part of the loss estimator.

The training sequence is:

1. Stack selected normalized windows and target return vectors.

1. Run the Hive forward pass.

1. Multiply weights by next-day returns and sum across assets.

1. Compute the batch mean and standard deviation of portfolio returns.

1. Maximize annualized Sharpe through a negative-Sharpe loss.

1. Add the squared-weight concentration penalty.

1. Backpropagate through the complete differentiable pipeline.

1. Clip the global gradient norm and update parameters with Adam.

The best validation state is saved to `model/hive.pt`. After all epochs, that state is restored before the test pass.

## 11. Complexity and memory profile

Let $$B$$ be batch size, $$A$$ asset count, $$L$$ sequence length, $$N$$ strategy count, $$d$$ hidden width, and $$k$$ neighborhood width.

The dominant recurrent operations are approximately:

| Operation | Approximate scale per timestep |
| --- | --- |
| Input projection | $$O(BA d_xd)$$ |
| Recurrent projection | $$O(BA Nd^2)$$ |
| Two coupling projections | $$O(BA Nk d^2)$$ if counted per gathered neighbor |
| Local score construction | $$O(BA Nk d)$$ |
| State storage | $$O(BANd)$$ |

The supplied implementation performs the asset loop in Python and uses tensor operations inside each asset update. Increasing $$A$$, $$N$$, or $$d$$ increases both activation memory and runtime. Increasing $$L$$ increases runtime linearly because every timestep executes the recurrence.

The model’s parameter count does not grow with asset count because the `NeuronGroup` is shared. The output and activation sizes do grow with asset count.

## 12. Failure boundaries and implementation caveats

| Area | Current behavior | Production implication |
| --- | --- | --- |
| Missing data | Drops dates missing any configured close | A single asset’s missing observation removes the entire cross-asset sample. |
| Normalization | Refits an expanding scaler for each sample | Computationally expensive and not identical to a frozen train-only preprocessing pipeline. |
| State initialization | Resets to zero for every forward call | No memory persists across separate windows or batches. |
| Cross-asset interaction | No direct hidden-state exchange | Relationships are expressed only through the final portfolio competition and shared parameters. |
| Allocation constraints | Positive weights summing to one | No leverage, shorts, cash, turnover, or maximum-position constraints. |
| Objective | Batch Sharpe plus concentration | Small or homogeneous batches can give unstable variance estimates. |
| Costs | Not included | Backtested returns may overstate implementable performance. |
| Drawdown | Uses cumulative-return differences | This differs from standard wealth-relative drawdown when wealth is not near one. |
| Checkpoint path | Saves to `model/hive.pt` | The `model` directory must exist before saving. |
| Reproducibility | Weight initialization and minibatch order are stochastic | A full reproducibility protocol would seed NumPy, PyTorch, and relevant CUDA operations. |

The most important architectural limitation is the absence of a transaction-aware state. The network chooses a fresh simplex allocation for each window, but the loss does not penalize changes from the previous allocation. A production-oriented extension would add a turnover term such as

$$
\lambda_{\mathrm{turn}}\lVert w_t-w_{t-1}\rVert_1
$$

or a differentiable cost model based on traded notional.

## 13. Extension points

The architecture can be extended without changing its central contract.

| Extension | Required change |
| --- | --- |
| More assets | Change the ticker universe and `n_assets`; shared parameters can remain unchanged. |
| More features | Change `d_x` and the input projection dimension. |
| More strategies | Change `n_strategies`; the ring index buffer and readout resize automatically. |
| Wider state | Change `d_state`; all hidden projections resize consistently. |
| Cross-asset attention | Add an explicit asset-mixing block before or after the per-asset recurrence. |
| Cash allocation | Add a cash return and include it as an additional softmax asset. |
| Shorting | Replace simplex softmax with a constrained signed-allocation parameterization. |
| Turnover control | Feed prior weights into the model or add a turnover penalty to the loss. |
| Risk budgeting | Add volatility estimates and a differentiable risk-adjusted objective. |
| Robust validation | Use walk-forward folds and fit preprocessing only on each training fold. |

## 14. Architectural summary

Hive can be understood as a hierarchy of shared computation:

1. **Feature layer:** converts each price series into four normalized indicators.

1. **Temporal layer:** integrates each asset’s twenty observations into latent states.

1. **Strategy layer:** maintains thirty interacting hypotheses on a circular topology.

1. **Attention layer:** selects a batch-contextual mixture of those hypotheses.

1. **Portfolio layer:** ranks the six assets and maps the ranking to simplex weights.

1. **Objective layer:** evaluates the allocation using realized next-day returns.

The architecture’s defining design choice is that a portfolio is produced by a **population of locally coupled latent strategies**, not by six isolated predictors. The ring provides structured local interaction, the shared parameters provide cross-asset transfer, and the final softmax converts the resulting relative beliefs into a coherent allocation.

## References

[1]: https://pytorch.org/docs/stable/generated/torch.nn.Module.html "PyTorch neural network module documentation"

[2]: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html "PyTorch Linear layer documentation"

[3]: https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html "PyTorch softmax documentation"

[4]: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html "PyTorch Adam optimizer documentation"

[5]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html "Pandas rolling-window documentation"

[6]: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html "Scikit-learn StandardScaler documentation"

[7]: https://pypi.org/project/yfinance/ "yfinance package documentation"

[8]: ./pasted_content.txt "Supplied Hive implementation"
