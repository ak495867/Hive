# Hive: Mathematical Specification

## 1. Scope and mathematical object

**Hive** is a differentiable multi-asset portfolio allocator. Given a trailing window of engineered market features for each asset, it produces a non-negative portfolio weight for every asset. The weights sum to one because the final allocation is a temperature-scaled softmax.

The implementation is not a price forecaster in the usual scalar-regression sense. It is a policy

$$
\pi_\theta:\mathbb{R}^{A\times L\times d_x}\to\Delta^{A-1},
$$

where $$A$$ is the number of assets, $$L$$ is the lookback length, $$d_x$$ is the number of features per observation, $$\theta$$ is the collection of trainable parameters, and

$$
\Delta^{A-1} = \{ w \in \mathbb{R}^{A} \mid w_a \geq 0,\ \sum_{a=1}^{A} w_a = 1 \}
$$

is the probability simplex. The portfolio return is the weighted sum of the next-period asset returns, and training maximizes an annualized sample Sharpe ratio subject to a concentration penalty.

The complete mathematical pipeline is

$$
\text{prices}
\rightarrow\text{causal features}
\rightarrow\text{rolling normalization}
\rightarrow\text{lookback tensor}
\rightarrow\text{recurrent cellular states}
\rightarrow\text{local ring coupling}
\rightarrow\text{strategy scores}
\rightarrow\text{temperature softmax}
\rightarrow\text{portfolio return}
\rightarrow\text{Sharpe objective}.
$$

The code is an educational and research prototype. It does not model transaction costs, slippage, market impact, leverage, shorting, turnover, liquidity, taxes, or execution constraints. Consequently, its objective is a simulated return objective, not a complete trading utility.

## 2. Notation and dimensions

| Symbol | Meaning | Value in the supplied implementation |
| --- | --- | --- |
| $$A$$ | Number of assets | $$6$$ |
| $$L$$ | Lookback length | $$20$$ |
| $$d_x$$ | Input feature dimension | $$4$$ |
| $$N$$ | Number of latent strategies / neurons | $$30$$ |
| $$d$$ | Hidden-state dimension | $$32$$ |
| $$k$$ | Ring neighborhood width | $$5$$ |
| $$d_k$$ | Attention-key/query dimension | $$16$$ |
| $$\tau$$ | State update rate | $$0.1$$ |
| $$T$$ | Portfolio softmax temperature | $$0.1$$ |
| $$B$$ | Mini-batch size | $$64$$ |
| $$\theta$$ | All trainable model parameters | learned by Adam |

For a batch, the input has shape

$$
X\in\mathbb{R}^{B\times A\times L\times d_x}.
$$

The slice for asset $$a$$ at time $$t$$ is denoted $$x_{b,a,t}\in\mathbb{R}^{d_x}$$. The hidden state for asset $$a$$ is

$$
H^{(a)}_b(t)\in\mathbb{R}^{N\times d},
$$

with row $$n$$ representing the state of latent strategy $$n$$ for asset $$a$$.

## 3. Market features

For each asset, let $$C_t>0$$ denote the close at day $$t$$. The feature vector is

$$
 x_t=\begin{bmatrix}
 r_t\\
 \rho_t\\
 b_t^+\\
 b_t^-
 \end{bmatrix}\in\mathbb{R}^{4}.
$$

### 3.1 Log return

The implementation computes

$$
 r_t=\log C_t-\log C_{t-1}=\log\left(\frac{C_t}{C_{t-1}}\right).
$$

The first observation is explicitly set to $$0$$ because no preceding price exists in the downloaded array.

### 3.2 Relative Strength Index feature

Define the close change

$$
\Delta C_t=C_t-C_{t-1}.
$$

The positive and negative magnitudes are

$$
U_t=\max(\Delta C_t,0),
\qquad
D_t=\max(-\Delta C_t,0).
$$

Using a ten-observation simple rolling mean,

$$
\overline U_t=\frac{1}{10}\sum_{j=0}^{9}U_{t-j},
\qquad
\overline D_t=\frac{1}{10}\sum_{j=0}^{9}D_{t-j},
$$

where the implementation fills unavailable early rolling values with zero. The unscaled RSI is

$$
\mathrm{RSI}_t
=100-\frac{100}{1+\frac{\overline U_t}{\overline D_t+10^{-9}}}.
$$

Hive uses the normalized form

$$
\rho_t=\frac{\mathrm{RSI}_t}{100}.
$$

The small constant is a numerical safeguard. It does not change the intended indicator materially except near zero downside movement.

### 3.3 Twenty-day Bollinger deviations

The twenty-day moving average and population-style rolling dispersion used by pandas are represented as

$$
\mu_t^{(20)}=\frac{1}{20}\sum_{j=0}^{19}C_{t-j},
$$

$$
 s_t^{(20)}=\mathrm{Std}_{20}(C_t,C_{t-1},\ldots,C_{t-19}).
$$

For the early rows, the moving average is backward-filled and the standard deviation is filled with zero. The upper and lower standardized deviations are

$$
 b_t^+=\frac{C_t-(\mu_t^{(20)}+2s_t^{(20)})}{s_t^{(20)}+10^{-9}},
$$

$$
 b_t^- =\frac{C_t-(\mu_t^{(20)}-2s_t^{(20)})}{s_t^{(20)}+10^{-9}}.
$$

These are not clipped to $$[-1,1]$$. They measure the signed distance of the close from the upper and lower two-standard-deviation bands in units of the rolling dispersion.

## 4. Dataset alignment and normalization

For each forecast origin $$i$$, the model receives the last $$L=20$$ feature vectors:

$$
X_i^{(a)}=\begin{bmatrix}
 x_{i-L+1}^{(a)}\\
 x_{i-L+2}^{(a)}\\
 \vdots\\
 x_i^{(a)}
\end{bmatrix}\in\mathbb{R}^{L\times d_x}.
$$

The target one-day simple return is

$$
 y_i^{(a)}=\frac{C_{i+1}^{(a)}-C_i^{(a)}}{C_i^{(a)}+10^{-8}}.
$$

Therefore, the batch target matrix is $$Y_i\in\mathbb{R}^{A}$$, and it is aligned after the final row of the input window.

For each asset and forecast origin, the code fits a `StandardScaler` to all available flattened feature rows from earlier windows plus the current window. If $$j$$ denotes a feature coordinate, the transformation is

$$
\widetilde x_{t,j}^{(i,a)}
=\frac{x_{t,j}^{(a)}-\mu_{i,a,j}}{\sigma_{i,a,j}+\epsilon},
$$

where $$\mu_{i,a,j}$$ and $$\sigma_{i,a,j}$$ are computed from the cumulative available rows and $$\epsilon$$ is the scaler's numerical safeguard. The resulting model input remains $$20\times4$$ per asset.

This is a time-ordered, expanding normalization scheme. It avoids using observations after the current forecast origin, but the current lookback window participates in estimating its own scale. This is a legitimate implementation choice for a rolling representation, although a stricter experiment would fit the scaler only on the training fold and freeze it for validation and test data.

## 5. Ring topology

The $$N$$ latent strategies are arranged on a discrete ring. Let

$$
\mathbb{Z}_N=\{0,1,\ldots,N-1\}
$$

with indices interpreted modulo $$N$$. For odd $$k$$, define the neighborhood of strategy $$n$$ as

$$
\mathcal{N}(n)=\{n-j:n\in\mathbb{Z}_N,\ j\in\{-m,\ldots,m\}\},
\qquad m=\left\lfloor\frac{k}{2}\right\rfloor.
$$

For the configured $$k=5$$, this is

$$
\mathcal{N}(n)=\{n-2,n-1,n,n+1,n+2\}\pmod N.
$$

The center strategy is included in its own neighborhood. The resulting index tensor has shape $$N\times k$$.

## 6. Local adaptive coupling

At time $$t$$, let $$h_{b,n}^{(a)}(t-1)\in\mathbb{R}^{d}$$ be the previous state of strategy $$n$$ for asset $$a$$. For each neighbor $$r\in\mathcal{N}(n)$$, construct the concatenated pair

$$
 c_{b,n,r}^{(a)}(t)=
\begin{bmatrix}
 h_{b,n}^{(a)}(t-1)\\
 h_{b,r}^{(a)}(t-1)
\end{bmatrix}\in\mathbb{R}^{2d}.
$$

A learned vector $$u\in\mathbb{R}^{2d}$$ produces an unnormalized local score

$$
 s_{b,n,r}^{(a)}(t)=u^\top c_{b,n,r}^{(a)}(t).
$$

The local mixture weights are a neighborhood softmax:

$$
 w_{b,n,r}^{(a)}(t)
=\frac{\exp(s_{b,n,r}^{(a)}(t))}
{\sum_{q\in\mathcal{N}(n)}\exp(s_{b,n,q}^{(a)}(t))}.
$$

Thus, for every $$(b,a,n)$$,

$$
 w_{b,n,r}^{(a)}(t)>0,
\qquad
\sum_{r\in\mathcal{N}(n)}w_{b,n,r}^{(a)}(t)=1.
$$

The coupling transform is unusual and important. Each neighbor state is passed through two independent learned linear maps and multiplied elementwise:

$$
 g_{b,r}^{(a)}(t)
=C_{\mathrm{left}}h_{b,r}^{(a)}(t-1)
\odot
 C_{\mathrm{right}}h_{b,r}^{(a)}(t-1).
$$

Since each factor is in $$\mathbb{R}^{d}$$, $$g_{b,r}^{(a)}(t)\in\mathbb{R}^{d}$$. The local coupling is the convex combination

$$
\mathrm{Coupling}_{b,n}^{(a)}(t)
=\sum_{r\in\mathcal{N}(n)}
 w_{b,n,r}^{(a)}(t)g_{b,r}^{(a)}(t).
$$

The word “convex” applies across neighbors for each hidden coordinate because the weights are positive and sum to one. The transformed vectors themselves need not be bounded or positive.

## 7. Leaky recurrent state dynamics

The input is shared across the $$N$$ strategies for a given asset. Define

$$
 q_b^{(a)}(t)=W_{\mathrm{in}}x_{b,a,t}+b_{\mathrm{in}}\in\mathbb{R}^{d},
$$

and the recurrent transformation

$$
 r_{b,n}^{(a)}(t)=W_{\mathrm{rec}}h_{b,n}^{(a)}(t-1)\in\mathbb{R}^{d}.
$$

The code uses the same input vector $$q_b^{(a)}(t)$$ for every strategy $$n$$. The preactivation is

$$
 z_{b,n}^{(a)}(t)=q_b^{(a)}(t)+r_{b,n}^{(a)}(t)
+\mathrm{Coupling}_{b,n}^{(a)}(t)+b,
$$

where $$b\in\mathbb{R}^{d}$$ is a learned bias. The nonlinear candidate state is $$\tanh(z)$$, and the actual state uses leaky integration:

$$
 h_{b,n}^{(a)}(t)
=(1-\tau)h_{b,n}^{(a)}(t-1)
+\tau\tanh\left(z_{b,n}^{(a)}(t)\right).
$$

With $$\tau=0.1$$, 90% of the previous state and 10% of the new bounded candidate are combined at every step. The initial state is

$$
 h_{b,n}^{(a)}(0)=0.
$$

The recurrence is asset-wise. Although all assets share the same parameterized `NeuronGroup`, no state from asset $$a$$ is passed into asset $$a'\ne a$$. Therefore, the model shares dynamics across assets but does not directly learn cross-asset state interactions.

## 8. Strategy readout and local score aggregation

After the final timestep $$L$$, each strategy produces a raw score

$$
 p_{b,n}^{(a)}
=\left\langle h_{b,n}^{(a)}(L),R_n\right\rangle+c_n,
$$

where $$R_n\in\mathbb{R}^{d}$$ is the learned row of `readout_weight` and $$c_n$$ is the learned readout bias.

The local score for strategy $$n$$ is the weighted mixture of its neighboring raw scores:

$$
 \ell_{b,n}^{(a)}
=\sum_{r\in\mathcal{N}(n)}
 w_{b,n,r}^{(a)}(L)p_{b,r}^{(a)}.
$$

The local weights are the final-step coupling weights returned by the recurrence. There is no additional nonlinear activation, sigmoid, or clipping on $$p$$ or $$\ell$$; these are unconstrained real-valued signals.

## 9. Global strategy attention

The model converts local strategy scores into one asset score using attention over the $$N$$ strategies.

First compute the batch-and-asset global context

$$
 c_g=\frac{1}{BA}\sum_{b=1}^{B}\sum_{a=1}^{A}h_b^{(a)}(L)\in\mathbb{R}^{d}.
$$

The query is

$$
 q=W_qc_g\in\mathbb{R}^{d_k}.
$$

For each strategy $$n$$, compute the key from the mean state over assets:

$$
 \bar h_n=\frac{1}{B}\sum_{b=1}^{B}h_{b,n}^{(a)}(L),
$$

where the implementation's tensor reduction yields one key per strategy after averaging over the batch dimension. In matrix notation, with $$K\in\mathbb{R}^{N\times d_k}$$,

$$
 K=W_k\bar H.
$$

The attention logits are

$$
 e_n=\frac{q^\top K_n}{\sqrt{d_k}},
$$

and the global strategy distribution is

$$
 \alpha_n=\frac{\exp(e_n)}{\sum_{m=1}^{N}\exp(e_m)}.
$$

Thus $$\alpha\in\Delta^{N-1}$$. The final asset score for batch item $$b$$ and asset $$a$$ is

$$
 s_b^{(a)}=\sum_{n=1}^{N}\alpha_n\ell_{b,n}^{(a)}.
$$

A critical implementation detail is that $$\alpha$$ is shared across the batch and assets during one forward pass because the query and keys use reductions over those dimensions. The individual asset-specific information enters through the local scores $$\ell_b^{(a)}$$, not through an asset-specific global attention distribution.

## 10. Portfolio allocation

Collect the asset scores into

$$
 s_b=\begin{bmatrix}s_b^{(1)}&\cdots&s_b^{(A)}\end{bmatrix}^\top.
$$

The portfolio weight for asset $$a$$ is

$$
 w_b^{(a)}
=\frac{\exp(s_b^{(a)}/T)}
{\sum_{j=1}^{A}\exp(s_b^{(j)}/T)}.
$$

With $$T=0.1$$, score differences are amplified by a factor of ten before normalization. The output satisfies

$$
 w_b^{(a)}>0,
\qquad
\sum_{a=1}^{A}w_b^{(a)}=1.
$$

As $$T\to0^+$$, the allocation approaches a one-hot choice of the highest-scoring asset. As $$T\to\infty$$, the allocation approaches equal weighting. The finite temperature provides smooth differentiability while controlling concentration indirectly.

## 11. Portfolio return and training objective

For target asset returns $$y_b^{(a)}$$, the simulated one-period portfolio return is

$$
 R_b=\sum_{a=1}^{A}w_b^{(a)}y_b^{(a)}=w_b^\top y_b.
$$

For a mini-batch of size $$B$$, the sample mean and standard deviation are

$$
 \bar R=\frac{1}{B}\sum_{b=1}^{B}R_b,
$$

$$
 s_R=\mathrm{Std}(R_1,\ldots,R_B)+10^{-8}.
$$

The annualized sample Sharpe estimate is

$$
 \widehat{\mathrm{Sh}}(R)
=\sqrt{252}\frac{\bar R}{s_R}.
$$

The concentration measure is the batch-average squared $$\ell_2$$ norm of the portfolio weights:

$$
 \mathcal C(W)=\frac{1}{B}\sum_{b=1}^{B}\sum_{a=1}^{A}(w_b^{(a)})^2.
$$

For a simplex vector, $$1/A\le\sum_a w_a^2\le1$$. Equal weights attain $1/A$, while a one-hot portfolio attains $$1$$. The implemented loss is

$$
 \mathcal L(\theta)
=-\widehat{\mathrm{Sh}}(R)
+\lambda_c\mathcal C(W),
\qquad \lambda_c=0.01.
$$

Minimizing this loss simultaneously maximizes the batch Sharpe estimate and discourages concentrated allocations. The concentration penalty is not a hard diversification constraint; the softmax still permits allocations arbitrarily close to one-hot when score differences are large.

## 12. Optimization

The model is optimized with Adam using learning rate $$10^{-3}$$ and weight decay $$10^{-5}$$. In conceptual form, for parameter vector $$\theta$$ and gradient $$g_t=\nabla_\theta\mathcal L_t$$,

$$
 m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
 v_t=\beta_2v_{t-1}+(1-\beta_2)g_t\odot g_t,
$$

$$
 \widehat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
 \widehat v_t=\frac{v_t}{1-\beta_2^t},
$$

$$
 \theta_{t+1}=\theta_t-\eta\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}
-\eta\lambda\theta_t.
$$

The implementation clips the global gradient norm to $$0.5$$. If the unscaled gradient is $$g$$, the clipped gradient is

$$
 g_{\mathrm{clip}}=g\cdot\min\left(1,\frac{0.5}{\lVert g\rVert_2+10^{-12}}\right).
$$

Training runs for 100 epochs. Each epoch randomly permutes training-window indices, while validation and test evaluation preserve chronological order.

## 13. Evaluation equations

The model saves the state with the highest validation Sharpe estimate. On the test set, it computes

$$
\widehat{\mathrm{Sh}}_{\mathrm{test}}
=\sqrt{252}\frac{\mathrm{mean}(R_{\mathrm{test}})}
{\mathrm{std}(R_{\mathrm{test}})+10^{-8}}.
$$

The cumulative return path is implemented as

$$
 G_t=\prod_{j=1}^{t}(1+R_j)-1.
$$

The reported maximum drawdown is computed from the cumulative-return series as

$$
 \mathrm{MaxDD}
=\max_t\left(\max_{u\le t}G_u-G_t\right).
$$

This is an absolute drawdown in cumulative-return units. A conventional wealth-relative drawdown would instead use wealth $$V_t=\prod_{j\le t}(1+R_j)$$ and compute $$\max_t(1-V_t/\max_{u\le t}V_u)$$.

## 14. Parameter count

For the configured values $$(d_x,d,N,k,d_k)=(4,32,30,5,16)$$, the trainable parameter blocks are:

| Parameter block | Count |
| --- | --- |
| $$W_{\mathrm{in}}$$ | $$32\cdot4=128$$ |
| $$W_{\mathrm{rec}}$$ | $$32\cdot32=1{,}024$$ |
| recurrent bias $$b$$ | $$32$$ |
| $$C_{\mathrm{left}}$$ | $1{,}024$ |
| $$C_{\mathrm{right}}$$ | $1{,}024$ |
| local scoring vector $$u$$ | $$64$$ |
| strategy readout weights | $$30\cdot32=960$$ |
| strategy readout bias | $$30$$ |
| query projection $$W_q$$ | $$16\cdot32=512$$ |
| key projection $$W_k$$ | $$16\cdot32=512$$ |
| **Total** | **5,310** |

The model has no separate parameter set per asset. Asset count affects activations and output size, but not the number of trainable weights.

## 15. Mathematical interpretation and limitations

Hive has four distinct mathematical stages. The recurrent and coupling equations create a latent representation. The readout and global attention create unconstrained asset scores. The temperature softmax converts scores into a simplex portfolio. The Sharpe-based loss evaluates the resulting portfolio rather than each asset prediction independently.

The current architecture has several consequences that should be made explicit:

| Implementation property | Mathematical consequence |
| --- | --- |
| Shared `NeuronGroup` across assets | Assets share parameters but maintain independent hidden states. |
| Ring neighborhoods | Strategies interact only with nearby indices, with wraparound at the boundary. |
| Self included in each neighborhood | A strategy can retain its own transformed state in its local mixture. |
| Elementwise product of two linear transforms | Coupling is multiplicative and can create second-order feature interactions. |
| Global attention reductions | One strategy-attention distribution is shared across the batch and assets in a forward pass. |
| Temperature $$0.1$$ | Small score differences can produce highly concentrated weights. |
| Sharpe objective | Optimization depends on cross-sample mean and dispersion, not pointwise prediction error. |
| No costs or turnover term | The learned policy may prefer economically infeasible reallocations. |
| Random training batches | The Sharpe estimate is noisy and depends on batch composition. |

The equations describe what the code computes. They do not establish that the resulting strategy has a persistent economic edge. Any empirical claim requires strict chronological validation, realistic costs, robustness across periods and assets, and uncertainty analysis.

## References

[1]: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html "PyTorch Linear layer documentation"

[2]: https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html "PyTorch softmax documentation"

[3]: https://arxiv.org/abs/1412.6980 "Adam: A Method for Stochastic Optimization"

[4]: https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html "PyTorch gradient clipping documentation"

[5]: https://en.wikipedia.org/wiki/Sharpe_ratio "Sharpe ratio definition and interpretation"

[6]: https://numpy.org/doc/stable/reference/generated/numpy.cumprod.html "NumPy cumulative product documentation"

[7]: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html "Scikit-learn StandardScaler documentation"

[8]: ./pasted_content.txt "Supplied Hive implementation"
