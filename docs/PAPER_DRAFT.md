# Predictive Carbon-Aware Traffic Engineering in Software-Defined Networks via Graph Neural Forecasting and Congestion Safeguards

**Ashutosh Gupta, Tanay Mohta, Avaneesh Chavan**  
Department of Computer Science and Engineering  
October 2026

---

## Abstract

Carbon-aware traffic engineering (TE) steers network flows toward geographic paths powered by low-carbon energy grids. However, naive carbon routing can inadvertently shift high-volume demand onto shared bottlenecks, precipitating link overload and route flap instability. This paper investigates whether short-horizon demand forecasting combined with uncertainty-calibrated capacity reservations and route-change hysteresis can balance environmental efficiency and network reliability under realistic telemetry delays. We implement a trace-driven, centralized software-defined networking (SDN) traffic engineering architecture on an eight-node GÉANT WAN core subgraph, driven by 15-minute historical traffic matrices paired with contemporary country-level electricity carbon intensity data from ECON-PowerCI. A spatial-temporal Graph Convolutional Network and Gated Recurrent Unit (GCN–GRU) architecture predicts multi-horizon origin–destination (OD) demand matrices for 15, 30, and 45 minutes ahead. Forecasts enter a conservative routing engine featuring: (i) an empirical $\alpha$-quantile validation residual safety margin $m_h(\alpha)$, (ii) shared reservation tracking across all horizons, and (iii) a relative improvement and hold-time hysteresis guard $(\tau, H)$ with emergency capacity overrides.

We evaluate CarbonRoute across an extensive multi-seed campaign covering 5 distinct neural initializations (2,475 simulation runs across 5 calendar days and 3 carbon calendar years). Under congested traffic with a 15-minute telemetry observation delay ($\delta = 1$ bin), the proposed GCN–GRU carbon policy reduces mean link overload duration from $126.0 \pm 0.0$ to $114.0 \pm 0.0$ minutes/day relative to reactive carbon routing (a mean reduction of $12.0$ min/day). To prevent pseudoreplication across repeated seed and carbon-year evaluations on the 5 calendar days, hypothesis testing is clustered at the day level ($N_{\text{days}} = 5$, day differences: $-15, 0, -15, -30, 0$ min/day; paired $t$-test $t = -2.14, p = 0.099$; Wilcoxon signed-rank $p = 0.25$), transparently establishing directional consistency while recognizing the sample size limits of 5 contiguous days. Concurrently, the byte-weighted forwarding carbon score increases from $323.83 \pm 0.00$ to $334.75 \pm 0.94\text{ g CO}_2/\text{kWh}$ (day-clustered $p = 0.037$) as the necessary cost of conservative detouring. Against a persistence forecasting baseline with identical safeguards, GCN–GRU carbon achieves identical overload mitigation ($114.0$ min/day) with lower carbon forwarding intensity ($-3.83\text{ g CO}_2/\text{kWh}$, day-clustered $p = 0.071$) and improved route stability ($18.68 \pm 0.95$ vs. $23.00 \pm 0.00$ route changes/day, day-clustered $p = 0.030$). An exploratory parameter sensitivity sweep on a representative test day across $\alpha \in [0.70, 0.99]$ and $\tau \in [0.00, 0.20]$ characterizes the empirical Pareto frontier, demonstrating that hysteresis dampens route changes by 35% at $\alpha=0.80$ and up to 70% under maximum dampening ($\alpha=0.99, \tau=0.20$). Finally, we provide an analytical assessment of simulation fidelity (fluid progressive filling vs. packet queuing) and a structured four-axis analysis of threats to validity.

**Keywords:** Software-Defined Networking, Carbon-Aware Routing, Graph Neural Networks, Demand Forecasting, Traffic Engineering, Route Flap Damping, Multi-Seed Evaluation, Pareto Frontier.

---

## 1. Introduction

Wide-area network (WAN) infrastructures connecting global data centers consume vast amounts of electrical power. While traditional traffic engineering (TE) optimizes for link utilization, propagation delay, and throughput, the decarbonization of cloud operations has elevated operational carbon emissions to a first-class routing metric. Because the carbon intensity of local electrical grids varies dramatically across geographic locations and diurnal cycles—driven by solar and wind availability—software-defined network (SDN) controllers can dynamically steer inter-datacenter traffic toward greener geographic paths to reduce operational greenhouse gas emissions.

However, carbon-aware routing introduces critical operational hazards:
1. **Carbon vs. Congestion Trade-Off:** Because low-carbon paths naturally attract traffic from multiple concurrent origin–destination (OD) flows, uncoordinated or reactive carbon-driven path switching causes localized traffic concentration on shared transit links. When aggregated demand exceeds physical link capacity, packet loss, queuing delays, and service degradation follow.
2. **Telemetry Observation Delay:** In operational SDNs, network telemetry (e.g., NetFlow/sFlow, OpenFlow statistics, or SNMP counters) is collected and aggregated over discrete measurement epochs (typically 5 to 15 minutes). A reactive controller that routes solely based on the most recently reported telemetry operates under delayed state information ($\delta \ge 1$ bin). When traffic surges occur faster than telemetry intervals, reactive routing creates severe transient overloads.
3. **Route Flap Instability:** In pursuit of marginal carbon reductions, frequent rerouting triggers route flapping. In physical networks, flow churn induces TCP packet reordering, micro-loops, transient queue build-ups, and control plane OpenFlow switch rule installation churn.

To resolve these challenges, this paper presents **CarbonRoute**, an integrated predictive traffic engineering architecture combining: (i) spatial-temporal Graph Convolutional Network and Gated Recurrent Unit (GCN–GRU) demand prediction, (ii) validation-calibrated non-parametric safety margins $m_h(\alpha)$ guarding against forecast underprediction, (iii) shared multi-horizon reservation tracking, and (iv) parameterized route flap hysteresis $(\tau, H)$ with emergency capacity overrides.

### Research Questions
We formally structure our experimental investigation around four core research questions:
- **RQ1 (Overload Mitigation):** Can multi-horizon traffic forecasting combined with conservative capacity reservations reduce link overload under delayed telemetry relative to reactive carbon routing?
- **RQ2 (Conservatism Cost):** What carbon forwarding intensity penalty and served traffic impact accompany validation-calibrated safety margins $m_h(\alpha)$ as the quantile $\alpha$ scales from $70\%$ to $99\%$?
- **RQ3 (Graph Neural Advantage):** Does spatial graph convolution in GCN–GRU provide measurable routing or forecasting advantages over a matched non-graph GRU and persistence zero-order hold baselines?
- **RQ4 (Pareto Frontier & Robustness):** How does the empirical Pareto frontier between carbon reduction, route flap stability, and overload mitigation behave across safety margin quantiles $\alpha$ and hysteresis thresholds $\tau$ under telemetry observation delays?

---

## 2. Related Work and Positioning

### 2.1 Carbon-Aware Routing & Green Traffic Engineering
Green routing historically focused on energy-proportional networking, putting idle line cards and switches into low-power sleep states during off-peak periods. Modern carbon-aware networking shifts focus from absolute energy reduction (Joules) to carbon intensity minimization (g $\text{CO}_2$/kWh), exploiting the temporal and spatial variance of renewable energy across geographically distributed grid interconnects. Tabaeiaghdaei et al. proposed CIRo [5], investigating inter-domain carbon-aware routing via BGP path selection. Similarly, recent efforts leverage high-resolution carbon accounting datasets like ECON-PowerCI [3, 4] to quantify regional electricity emission factors. CarbonRoute builds on these foundations but diverges crucially: while existing green routing works often assume instantaneous telemetry and ignore capacity-induced overload, CarbonRoute models delayed telemetry observation lags and shared link bottlenecks.

### 2.2 Machine Learning & GNNs for Network Forecasting
Predictive traffic engineering seeks to forecast future traffic matrices to optimize routing decisions before congestion materializes. Valadarsky et al. proposed NeuRoute [6], demonstrating that neural networks can predict demand matrices and compute proactive routing configurations in SDNs. With the rise of geometric deep learning, Graph Neural Networks (GNNs) have demonstrated strong inductive biases for topological structures. Rusek et al. developed RouteNet [7] for network modeling and delay estimation in SDNs. While GNNs excel at spatial aggregation across network graphs, their concrete performance advantage over well-tuned classical temporal baselines (such as matched GRUs and persistence zero-order holds) in closed-loop traffic engineering remains intensely debated. In this paper, we conduct rigorous paired ablations to isolate the genuine contribution of spatial convolutions in wide-area demand forecasting.

### 2.3 Route Flap Damping & Congestion Safeguards
Centralized SDN WAN controllers, such as Google B4 [8] and Microsoft SWAN [9], maintain high link utilization through multi-commodity flow optimization executed at 5- to 15-minute epochs. A primary operational concern in centralized TE is route flapping: switching paths for negligible utility gains creates OpenFlow flow table churn, transient packet reordering, and TCP throughput collapse. Classical inter-domain routing employs BGP Route Flap Damping (RFC 2439) with exponential decay penalties. CarbonRoute adapts hysteresis principles to multi-objective carbon traffic engineering, formulating an explicit relative improvement guard $\tau$, a minimum hold-time $H$, and an emergency capacity override that decouples elective carbon optimizations from critical congestion prevention.

---

## 3. System Model & Network Topology

### 3.1 Network Topology & Graph Model
Let the communications network be represented by a directed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where:
- $\mathcal{V}$ is the set of network switches/nodes ($|\mathcal{V}| = 8$: FR, DE, BE, NL, CH, IT, AT, LU).
- $\mathcal{E}$ is the set of directed links ($|\mathcal{E}| = 22$, formed by bidirectionally modeling 11 physical fiber links).
- Each directed arc $e = (u, v) \in \mathcal{E}$ possesses a maximum transmission capacity $C_e = 10\text{ Mbps}$ and a physical propagation delay $D_e$:
  $$D_e = \frac{\text{haversine}(u, v) \times 1.3}{200\text{ km/ms}} + 0.15\text{ ms}$$
  where $\text{haversine}(u, v)$ denotes the great-circle surface distance between switch coordinates, $1.3$ is the optical fiber stretch factor, $200\text{ km/ms}$ represents the speed of light in fiber, and $0.15\text{ ms}$ models switch hardware queuing and processing overhead.
- Time is discretized into uniform observation bins of duration $\Delta t = 900\text{ s}$ (15 minutes), indexed by $t \in \mathbb{N}$.

### 3.2 Offered Demand Matrix & Active OD Pair Selection
- The full origin–destination (OD) traffic matrix at bin $t$ is denoted by $\mathbf{X}_t \in \mathbb{R}_{+}^{|\mathcal{V}| \times |\mathcal{V}|}$, where $X_t(s, d)$ is the offered demand from source $s$ to destination $d$ in Mbps.
- For each OD pair $(s, d)$, candidate paths $P_{sd}$ are generated as the $K \le 3$ shortest simple loop-free paths satisfying the propagation delay stretch constraint:
  $$P_{sd} = \left\{ p \in \text{SimplePaths}(s, d) : \sum_{e \in p} D_e \le 1.5 \times \min_{p' \in \text{SimplePaths}(s, d)} \left( \sum_{e \in p'} D_e \right) \right\}, \quad |P_{sd}| \le 3$$
- The active evaluation set $\mathcal{K} \subset \mathcal{V} \times \mathcal{V}$ ($|\mathcal{K}| = 6$) is selected greedily from the training partition using a congestion potential score:
  $$\text{Score}(s, d) = \bar{X}_{\text{train}}(s, d) \cdot \left( 1 + 0.25 \cdot \sum_{e \in p_{\text{short}}(s, d)} \sum_{(s', d') \neq (s, d)} \mathbf{1}_{e \in p_{\text{short}}(s', d')} \right)$$
  subject to $|P_{sd}| \ge 2$. Active OD pairs are IT→NL, IT→FR, DE→BE, NL→FR, NL→IT, and CH→DE. All inactive pairs are masked via a static binary mask $\mathbf{M} \in \{0, 1\}^{|\mathcal{V}| \times |\mathcal{V}|}$.
- Demand matrices are linearly scaled by scalar factor $\gamma$ such that the 95th percentile of peak shortest-path link load across training bins equals $70\%$ of capacity:
  $$\gamma = \frac{0.70 \times 10.0}{\text{Quantile}_{0.95} \left( \max_{e \in \mathcal{E}} \sum_{(s, d) \in \mathcal{K}} X_t(s, d) \cdot \mathbf{1}_{e \in p_{\text{short}}(s, d)} \right)}$$

### 3.3 Carbon Intensity Modeling & Counterfactual Pairing
- For each node $v \in \mathcal{V}$, the electricity grid carbon intensity at bin $t$ is denoted by $I_v(t) \in \mathbb{R}_{+}$ ($\text{g CO}_2/\text{kWh}$, numerically equivalent to $\text{kg CO}_2/\text{MWh}$).
- Sourced from daily consumption-based country-level ECON-PowerCI data [3, 4]. Because historical 2005 traffic traces predate carbon accounting and Luxembourg data is available only post-June 2021, we construct a rigorous counterfactual pairing operator $\mathcal{T}_{\text{carbon}}(t)$. Traffic from 2005 is matched to target calendar years $Y \in \{2022, 2023, 2024\}$ by selecting the nearest date matching the same day of the week and season within $\pm 21$ days with complete 8-country data coverage.
- $I_v(t)$ remains constant over all 96 bins of each calendar day and is held static across the predictive planning horizon.

---

## 4. GCN–GRU Demand Prediction Architecture

Given a historical lookback window $W = 12$ bins ($3\text{ hours}$), the model processes input tensor $\mathbf{X}_{t-W+1 : t} \in \mathbb{R}_{+}^{W \times |\mathcal{V}| \times |\mathcal{V}|}$:

```
     Input History X_{t-W+1 : t} (12 bins x 8 x 8)
                     │
                     ▼
  ┌────────────────────────────────────────────────────────┐
  │ 1. Spatial Graph Convolution                           │
  │    z_t(v) = [ x_t(v) || sum_u A~_{vu} x_t(u) ] (16-d) │
  │    e_t(v) = ReLU( W_spatial * z_t(v) + b_spatial )     │
  └──────────────────────────┬─────────────────────────────┘
                             │ Node Embeddings (32-d)
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ 2. Temporal Recurrence                                 │
  │    h_t(v) = GRU( e_t(v), h_{t-1}(v) ) (32 hidden)      │
  └──────────────────────────┬─────────────────────────────┘
                             │ Final Hidden State
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ 3. Multi-Horizon Residual Head                         │
  │    Y~_{t+h} = [ X_t + W_head^(h)*h_t + b^(h) ]+ . M   │
  │    for horizons h in {1, 2, 3} (15, 30, 45 min)        │
  └────────────────────────────────────────────────────────┘
```

1. **Spatial Graph Convolution:**
   Let $\mathbf{A} \in \{0, 1\}^{|\mathcal{V}| \times |\mathcal{V}|}$ be the adjacency matrix and $\mathbf{\tilde{A}} = \mathbf{\tilde{D}}^{-1/2} (\mathbf{A} + \mathbf{I}) \mathbf{\tilde{D}}^{-1/2}$ be the symmetric normalized adjacency matrix with self-loops. For each node $v \in \mathcal{V}$, the local outgoing demand vector $\mathbf{x}_t(v) \in \mathbb{R}^{|\mathcal{V}|}$ is concatenated with its normalized graph neighborhood:
   $$\mathbf{z}_t(v) = \left[ \mathbf{x}_t(v) \,\|\, \sum_{u \in \mathcal{V}} \tilde{A}_{v, u} \mathbf{x}_t(u) \right] \in \mathbb{R}^{2|\mathcal{V}|} = \mathbb{R}^{16}$$
   A linear projection followed by ReLU activation produces node embedding $\mathbf{e}_t(v) \in \mathbb{R}^{d}$ ($d = 32$):
   $$\mathbf{e}_t(v) = \text{ReLU}\left( \mathbf{W}_{\text{spatial}} \mathbf{z}_t(v) + \mathbf{b}_{\text{spatial}} \right)$$

2. **Temporal Recurrence:**
   The sequence of node embeddings is fed into a shared Gated Recurrent Unit (GRU):
   $$\mathbf{h}_t(v) = \text{GRU}\left( \mathbf{e}_t(v), \mathbf{h}_{t-1}(v) \right), \quad \mathbf{h}_t(v) \in \mathbb{R}^{32}$$

3. **Residual Forecast Head:**
   The final recurrent hidden state $\mathbf{h}_t(v)$ feeds a linear projection predicting residual deviations for future horizons $h \in \{1, 2, 3\}$ ($15, 30, 45\text{ min}$):
   $$\mathbf{\hat{Y}}_{t+h} = \left[ \mathbf{X}_t + \mathbf{W}_{\text{head}}^{(h)} \mathbf{h}_t + \mathbf{b}_{\text{head}}^{(h)} \right]_{+} \odot \mathbf{M}$$
   where $[\cdot]_{+}$ denotes non-negative clamping ($\max(\cdot, 0)$) and $\odot$ is element-wise multiplication by active OD pair mask $\mathbf{M}$. Total trainable parameters: $7,672$.

---

## 5. Congestion Safeguards & Path Selection

### 5.1 Validation-Calibrated Safety Margin Residual Quantiles ($\alpha$)
To guard against forecast underprediction without parametric distribution assumptions, we calibrate an empirical safety margin on the chronological validation split $\mathcal{D}_{\text{val}}$:
1. For each horizon $h \in \{1, 2, 3\}$, compute positive underprediction residuals:
   $$\mathcal{R}_h = \left\{ \max\left( X_{t+h}(s, d) - \hat{Y}_{t+h}(s, d), 0 \right) : t \in \mathcal{D}_{\text{val}}, (s, d) \in \mathcal{K} \right\}$$
2. For a specified confidence percentile $\alpha \in [0.70, 0.99]$ (baseline $\alpha = 0.90$), compute the pooled empirical quantile:
   $$m_h(\alpha) = \text{Quantile}_{\alpha}(\mathcal{R}_h)$$
3. The conservative demand forecast used by the routing planner is:
   $$\hat{d}_{sd}^{(t+h)}(\alpha) = \hat{Y}_{t+h}(s, d) + m_h(\alpha)$$

### 5.2 Carbon-Aware Path Planning with Shared Reservation Tracking
At decision bin $t$, under observation delay $\delta \in \{0, 1\}$ bins:
1. **Conservative Horizon Demand:** For each active pair $k = (s, d) \in \mathcal{K}$, define peak horizon demand:
   $$\hat{d}_k^{\max} = \max_{h \in \{1, 2, 3\}} \hat{d}_{sd}^{(t+h)}(\alpha)$$
2. **Processing Order:** Sort pairs in descending order of $\hat{d}_k^{\max}$, breaking ties deterministically by pair index.
3. **Shared Reservation Update:** Maintain projected link load tensor $L(e, h)$ across horizons $h \in \{1, 2, 3\}$. For pair $k$ currently routed on $p_k^{\text{curr}}$:
   - **Remove Old Reservation:**
     $$L_0(e, h) = L(e, h) - \hat{d}_k^{(t+h)} \cdot \mathbf{1}_{e \in p_k^{\text{curr}}}, \quad \forall e \in \mathcal{E}, h \in \{1, 2, 3\}$$
   - **Evaluate Candidates:** For each candidate path $p \in P_k$, calculate projected load and utilization:
     $$L_p(e, h) = L_0(e, h) + \hat{d}_k^{(t+h)} \cdot \mathbf{1}_{e \in p}, \quad U_p(e, h) = \frac{L_p(e, h)}{C_e}$$
     Path peak utilization: $U_p^{\max} = \max_{e \in p, h} U_p(e, h)$. Global peak utilization: $G_p = \max_{e \in \mathcal{E}, h} U_p(e, h)$.
   - **Feasibility Filter:** Candidate $p$ is capacity-feasible if $U_p^{\max} \le U_{\text{limit}} = 0.80$. Let $\mathcal{F}_k = \{ p \in P_k : U_p^{\max} \le 0.80 \}$.
   - **Forwarding Carbon Cost:** The forwarding emissions cost charges only intermediate forwarding senders (the destination node $d$ does not forward):
     $$\mathcal{C}(p) = \sum_{v \in p \setminus \{d\}} I_v(t)$$
   - **Candidate Selection:**
     - If $\mathcal{F}_k \neq \emptyset$: $p^* = \arg\min_{p \in \mathcal{F}_k} \left( \mathcal{C}(p), G_p, \sum_{e \in p} D_e, p \right)$.
     - If $\mathcal{F}_k = \emptyset$ (Infeasible): $p^* = \arg\min_{p \in P_k} \left( G_p, U_p^{\max}, \mathcal{C}(p), \sum_{e \in p} D_e, p \right)$.

### 5.3 Route Flap Hysteresis & Damping Rule ($\tau, H$)
To suppress route flapping while maintaining responsiveness to impending overload:
- Let $H = 2$ bins ($30\text{ min}$) be the hold-time threshold and $\tau \ge 0$ (baseline $\tau = 0.05$, i.e., 5%) be the relative objective improvement threshold.
- Let $t_{\text{last}}(k)$ be the step when pair $k$ was last rerouted.
- Route change $p_k \leftarrow p^*$ is executed if and only if:
  $$\underbrace{\left( U_{p_k^{\text{curr}}}^{\max} > U_{\text{limit}} \right)}_{\text{Emergency Capacity Override}} \;\lor\; \left[ \underbrace{\left( t - t_{\text{last}}(k) \ge H \right)}_{\text{Hold Time Guard}} \;\land\; \underbrace{\left( \frac{\mathcal{C}(p_k^{\text{curr}}) - \mathcal{C}(p^*)}{\max(\mathcal{C}(p_k^{\text{curr}}), 10^{-9})} \ge \tau \right)}_{\text{Relative Improvement Guard}} \right]$$
- If rerouted, set $t_{\text{last}}(k) \leftarrow t$. Update projected link load: $L(e, h) \leftarrow L_0(e, h) + \hat{d}_k^{(t+h)} \cdot \mathbf{1}_{e \in p^*}$.

### 5.4 Joint Carbon-Budgeted Combinatorial Routing (`BudgetPlanner`)
While Section 5.2 defines a greedy sequential path allocator prioritizing overload elimination via conservative reservations, we additionally formulate a joint multi-commodity combinatorial selector. Because each of the $|\mathcal{K}| = 6$ active OD pairs has 2 candidate paths within the $1.5\times$ propagation delay stretch bound, the complete network routing space consists of $|\prod_{k \in \mathcal{K}} P_k| = 2^6 = 64$ simultaneous global route assignments $\mathbf{p} = (p_1, \dots, p_6)$.

At decision bin $t$, the joint selector evaluates all 64 assignments simultaneously:
1. **Predicted Peak Link Utilization:**
   $$U_{\max}(\mathbf{p}) = \max_{e \in \mathcal{E}, h \in \{1, 2, 3\}} \frac{\sum_{k \in \mathcal{K}} \mathbf{1}_{e \in p_k} \hat{d}_k^{(t+h)}}{C_e}$$
2. **Predicted Offered Carbon Cost:**
   $$\Phi(\mathbf{p}) = \sum_{k \in \mathcal{K}} \left( \frac{1}{3} \sum_{h=1}^3 \hat{d}_k^{(t+h)} \right) \cdot \sum_{v \in p_k \setminus \{\text{dst}(k)\}} I_v(t)$$
3. **Reactive Reference Carbon Budget:**
   Using the same multi-horizon demand forecast $\mathbf{\hat{d}}$, compute the predicted carbon cost of the contemporaneous reactive routing assignment $\mathbf{p}_{\text{reactive}}$. The carbon budget is:
   $$\mathcal{B} = \Phi(\mathbf{p}_{\text{reactive}}) \times (1 + \beta)$$
   where $\beta \ge 0$ is the permitted budget excess fraction (default $\beta = 0.0$, strictly disallowing any predicted excess over reactive routing).
4. **Combinatorial Selection:**
   The selector filters for carbon-budgeted assignments $\mathcal{F}_{\text{joint}} = \{ \mathbf{p} : \Phi(\mathbf{p}) \le \mathcal{B} \}$ and minimizes predicted peak utilization:
   $$\mathbf{p}^* = \arg\min_{\mathbf{p} \in \mathcal{F}_{\text{joint}}} \left( U_{\max}(\mathbf{p}), \Phi(\mathbf{p}), \sum_{k \in \mathcal{K}} \mathbf{1}_{p_k \neq p_k^{\text{curr}}}, \sum_{k} \sum_{e \in p_k} D_e \right)$$
5. **Emergency Overrides:**
   The elective hold-time $H = 2$ bins and relative threshold $\tau = 0.05$ are immediately overridden if the current assignment has $U_{\max}(\mathbf{p}^{\text{curr}}) > 0.80$ or breaches the carbon budget ($\Phi(\mathbf{p}^{\text{curr}}) > \mathcal{B}$), ensuring rapid recovery when reactive conditions improve.

---

## 6. Service & Environmental Attribution

### 6.1 Fluid Progressive-Filling Allocator
At bin $t$, actual offered demands $\mathbf{d}_t = [d_1, \dots, d_{|\mathcal{K}|}]^T$ are revealed. Max-min fair served flow rates $\mathbf{s}_t = [s_1, \dots, s_{|\mathcal{K}|}]^T$ are calculated via iterative equal-rate progressive filling:
1. Initialize served rates $s_k^{(0)} = 0$ for all $k \in \mathcal{K}$, active flow set $\mathcal{A}^{(0)} = \{ k \in \mathcal{K} : d_k > 0 \}$, and iteration $m = 0$.
2. While $\mathcal{A}^{(m)} \neq \emptyset$ and $m \le |\mathcal{K}| + |\mathcal{E}|$:
   - For each link $e \in \mathcal{E}$, calculate active flow count $n_e^{(m)} = \sum_{k \in \mathcal{A}^{(m)}} A_{e, k}$ and residual capacity:
     $$R_e^{(m)} = \max\left( C_e - \sum_{k \in \mathcal{K}} A_{e, k} s_k^{(m)}, 0 \right)$$
   - Compute max-min fair allocation increment:
     $$\Delta^{(m)} = \min \left( \min_{k \in \mathcal{A}^{(m)}} (d_k - s_k^{(m)}), \min_{e : n_e^{(m)} > 0} \frac{R_e^{(m)}}{n_e^{(m)}} \right)$$
   - Update allocations: $s_k^{(m+1)} = s_k^{(m)} + \Delta^{(m)}$ for all $k \in \mathcal{A}^{(m)}$, and $s_k^{(m+1)} = s_k^{(m)}$ for $k \notin \mathcal{A}^{(m)}$.
   - Identify saturated links: $\mathcal{S}^{(m)} = \{ e \in \mathcal{E} : \sum_k A_{e, k} s_k^{(m+1)} \ge C_e - 10^{-9} \}$.
   - Deactivate satisfied or bottlenecked flows:
     $$\mathcal{A}^{(m+1)} = \left\{ k \in \mathcal{A}^{(m)} : s_k^{(m+1)} < d_k - 10^{-9} \text{ and } \forall e \in p_k, e \notin \mathcal{S}^{(m)} \right\}$$
   - Increment $m \leftarrow m + 1$.
3. Output final served rates: $s_k = \min(s_k^{(m)}, d_k)$.

### 6.2 Carbon Attribution & Service Metrics
- **Served Volume:** For flow $k$ over bin $t$ ($\Delta t = 900\text{ s}$), served data volume is $V_k(t) = s_k(t) \cdot \frac{900}{8}\text{ MB}$.
- **Served Traffic Fraction:**
  $$\eta_{\text{served}} = \frac{\sum_{t=1}^T \sum_{k \in \mathcal{K}} s_k(t)}{\sum_{t=1}^T \sum_{k \in \mathcal{K}} d_k(t)} \times 100\%$$
- **Byte-Weighted Forwarding Carbon Score:**
  $$S_{\text{forwarding}} = \frac{\sum_{t=1}^T \sum_{k \in \mathcal{K}} V_k(t) \cdot \sum_{v \in p_k(t) \setminus \{\text{dst}(k)\}} I_v(t)}{\sum_{t=1}^T \sum_{k \in \mathcal{K}} V_k(t)}$$
  Units: $\text{g CO}_2/\text{kWh}$ summed over forwarding hops.
- **Overload Duration:** Total minutes during which at least one directed link has offered load exceeding $100\%$ capacity:
  $$T_{\text{overload}} = \sum_{t=1}^T 15 \cdot \mathbf{1}_{\left( \max_{e \in \mathcal{E}} \sum_{k} A_{e, k}(t) d_k(t) > C_e \right)}\quad (\text{minutes/day})$$

---

## 7. Simulation Fidelity vs. Real Queuing Dynamics

A rigorous evaluation of traffic engineering must distinguish between macroscopic flow allocation and microscopic packet dynamics. Our simulation employs a deterministic, equal-rate progressive-filling allocator operating on 15-minute aggregated demand matrices. Here we explicitly delineate the assumptions, advantages, and limitations of this fluid formulation relative to packet-level queuing.

### 7.1 Macroscopic Fluid Model vs. Packet Queuing
1. **Flow vs. Packet Abstraction:** Progressive filling models data flows as continuous, infinitely divisible fluids. It allocates bottleneck capacities according to max-min fairness. In operational IP/MPLS and SDN deployments, traffic arrives as discrete packet trains governed by end-to-end transport protocols (e.g., TCP Cubic, BBR). 
2. **Buffer Dynamics and Delay:** The fluid allocator does not model finite switch buffer depth, packet drop-tail mechanics, Active Queue Management (AQM, e.g., RED, CoDel), or Round-Trip Time (RTT) inflation. When offered load exceeds 100% of link capacity, the fluid simulator registers an instantaneous service deficit. In a physical network, transient overload causes buffer bloat, packet loss, TCP retransmission timeouts, and congestion window collapse.
3. **Route Flap Penalty and Flow Churn:** In our fluid simulation, route transitions occur instantaneously at the start of a 15-minute bin without packet loss. In operational networks, rerouting flows induces transient packet reordering, micro-loops, and OpenFlow/BGP rule installation latency (1–10 ms per switch rule update). Frequent route flapping degrades TCP goodput through spurious fast retransmits. The hysteresis mechanisms in CarbonRoute ($\tau \ge 0.05, H = 2$ bins) are designed specifically to mitigate this physical instability.

### 7.2 Justification for Centralized Traffic Engineering
Despite these microscopic simplifications, the macroscopic fluid model is well justified for wide-area traffic engineering:
- Centralized SDN WAN TE controllers (such as Google B4 [8] and Microsoft SWAN [9]) execute optimization algorithms on 5- to 15-minute control epochs, matching our 15-minute bin duration.
- At this timescale, macroscopic link bandwidth reservations dictate steady-state trunk utilization; microscopic packet dynamics operate multiple orders of magnitude faster beneath the TE layer.
- By enforcing a conservative 80% link utilization ceiling ($U_{\text{limit}} = 0.80$) and adding validation-derived underprediction allowances, CarbonRoute preserves a 20% capacity headroom specifically to absorb unmodeled packet burstiness and queuing transients.

---

## 8. Threats to Validity

We structure the limitations of this study across four classical validity dimensions.

### 8.1 Internal Validity
- **Causal Isolation:** A primary threat in predictive routing is future information leakage. We eliminate this threat by strictly isolating the forecasting and planning APIs: at decision bin $t$, the neural predictor and reactive baselines observe demands strictly through $t-\delta$ ($\delta \in \{0, 1\}$). Routes are immutably committed before actual bin $t$ demand is revealed to the service engine.
- **Data Leakage in Uncertainty Calibration:** The safety margin residual quantiles $m_h(\alpha)$ are calibrated exclusively on the chronological validation split. Normalization scalers and OD selection metrics are fitted solely on training rows. No test set statistics influence routing decisions.
- **Window Autocorrelation:** The primary evaluation utilizes five contiguous calendar days. While day-clustered bootstrapping captures between-day variance, serial autocorrelation across adjacent days may underestimate true confidence interval widths.

### 8.2 External Validity
- **Graph Topology & Scale:** The evaluation is conducted on an 8-node, 22-directed-link core subgraph of GÉANT with six active demand pairs. While representative of research WAN backbones, large-scale commercial transit networks and hyperscale datacenter WANs feature higher path diversity and different degree distributions.
- **Traffic Profile & Synthetic Workloads:** Demands originate from the historical 2005 SNDlib GÉANT trace scaled to 70% training peak load. While the $1.6\times$ congested and $2.5\times$ bursty workloads stress the safeguards, modern cloud traffic exhibits diurnal patterns and burst characteristics that may differ from 2005 academic network traces.
- **Grid Carbon Granularity:** We utilize daily country-level consumption-based carbon intensities from ECON-PowerCI. Operational deployments would benefit from sub-hourly, locational marginal emission rates (e.g., from WattTime or ElectricityMaps).

### 8.3 Construct Validity
- **Forwarding Carbon Attribution Metric:** Our primary environmental metric, the byte-weighted forwarding carbon score, attributes emissions based on the sum of forwarding senders' grid carbon intensities per served megabyte under an equal energy-per-hop convention. It serves as a normalized proxy for routing greenness rather than a direct measurement of physical router power draw (Joules/megabit).
- **Temporal Counterfactual Pairing:** Because 2005 traffic pre-dates modern carbon accounting, pairing 2005 matrices with 2022–2024 grid data creates a synthetic counterfactual scenario. It evaluates how a carbon-aware TE controller would behave under realistic traffic variability and real grid carbon variations, but does not reconstruct historical emissions.

### 8.4 Conclusion Validity
- **Statistical Power & Model Multi-Seeding:** Initial pilot runs evaluated a single neural seed (seed 42). To ensure valid statistical conclusions, all reported metrics, confidence intervals, and hypothesis tests are aggregated across five independent training seeds ($S = [42, 43, 44, 45, 46]$) spanning 2,475 simulation runs, confirming that reported trade-offs are robust to stochastic weight initialization.
- **Statistical Unit & Degrees of Freedom:** While our campaign runs 2,475 simulations (75 evaluations per policy condition across 5 model seeds, 3 carbon years, and 5 traffic windows), the underlying traffic inputs originate from 5 calendar test days. Treating all 75 seed-year realizations as independent observations would commit pseudoreplication, artificially deflating standard errors and generating deceptively tiny p-values ($p < 10^{-9}$). To maintain statistical integrity, all hypothesis tests cluster by calendar day ($N_{\text{days}} = 5$). We explicitly emphasize that while the observed overload reduction (mean $-12.0$ min/day) is directionally consistent across every seed and year, 5 contiguous evaluation days provide only 4 degrees of freedom ($t = -2.14, p = 0.099$, Wilcoxon $p = 0.25$). Thus, this paper demonstrates an empirical improvement in replay rather than an asymptotic statistical proof, establishing the necessity for multi-month live testbed evaluations.

---

## 9. Empirical Evaluation & Results

### 9.1 Multi-Seed Demand Forecasting Accuracy
We evaluate forecast accuracy across 5 independent model seeds on the held-out test set (781 prediction windows per seed, 3,905 total evaluation windows). Table 1 presents MAE and RMSE across 15-, 30-, and 45-minute horizons.

#### Table 1: Multi-Seed Demand Forecasting Accuracy (Test Set, 781 Windows $\times$ 5 Seeds)
*Aggregated across 5 random seeds ($S = [42, 43, 44, 45, 46]$). Values reported as Mean $\pm$ Standard Deviation with paired significance tests across seeds.*

| Model | Architecture Description | 15-min MAE (Mbps) | 30-min MAE (Mbps) | 45-min MAE (Mbps) | Mean MAE (Mbps) | Mean RMSE (Mbps) | $p$-value vs. GNN |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **GCN–GRU** | Spatial GCN + Shared Node GRU | $0.3611 \pm 0.0031$ | $0.5108 \pm 0.0057$ | $0.6124 \pm 0.0051$ | $0.4948 \pm 0.0044$ | $1.9813 \pm 0.0058$ | [Reference] |
| **Matched GRU** | Identity Adjacency + Node GRU | $0.3505 \pm 0.0011$ | $0.4959 \pm 0.0023$ | $0.5984 \pm 0.0020$ | $0.4816 \pm 0.0018$ | $1.9841 \pm 0.0016$ | $p = 0.0017$ |
| **Persistence** | Zero-Order Hold $\mathbf{X}_t$ | $0.3479 \pm 0.0000$ | $0.4976 \pm 0.0000$ | $0.6121 \pm 0.0000$ | $0.4858 \pm 0.0000$ | $2.0322 \pm 0.0000$ | $p = 0.0117$ |

Across all three horizons, the matched GRU achieves a mean MAE of $0.4816\text{ Mbps}$, marginally outperforming both persistence ($0.4858\text{ Mbps}$) and GCN–GRU ($0.4948\text{ Mbps}$). The spatial convolution in GCN–GRU does not yield a lower forecast MAE on this 8-node topology, answering **RQ3**: graph message passing provides no significant accuracy advantage on sparse WAN subgraphs where temporal autocorrelation dominates spatial cross-correlation. However, as demonstrated below, routing stability and carbon efficiency interact differently with forecast dynamics.

### 9.2 Primary Routing Performance Under Telemetry Lag
We evaluate routing performance under the primary operational challenge: congested traffic ($1.6\times$ load) with a 15-minute telemetry observation delay ($\delta = 1$ bin).

First, Table 2 reports the comprehensive multi-seed performance aggregated across all 5 random seeds (75 paired evaluation days per policy).

#### Table 2: Multi-Seed Routing Performance Under Telemetry Lag ($\delta = 1$ bin, Congested Workload, 5 Seeds, 75 Paired Runs)
*Aggregated across 5 random seeds ($S = [42, 43, 44, 45, 46]$). Values reported as Mean $\pm$ Standard Deviation with day-clustered significance tests ($N_{\text{days}}=5$).*

| Policy | Forwarding Carbon Score ($\text{g CO}_2/\text{kWh}$) | Served Demand (%) | Overload Duration (min/day) | Route Changes (/day) | Day-Clustered $p$-val vs. Reactive ($N_{\text{days}}=5$) | Day-Clustered $p$-val vs. Persistence ($N_{\text{days}}=5$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Shortest Path** | $419.36 \pm 0.00$ | $91.22 \pm 0.00\%$ | $165.0 \pm 0.0$ | $0.00 \pm 0.00$ | $p = 0.045$ | — |
| **Reactive Carbon** | $323.83 \pm 0.00$ | $91.89 \pm 0.00\%$ | $126.0 \pm 0.0$ | $19.40 \pm 0.00$ | [Baseline] | — |
| **Predictive TE** | $343.59 \pm 1.95$ | $92.21 \pm 0.03\%$ | $108.0 \pm 0.0$ | $8.16 \pm 0.88$ | $p = 0.099$ | — |
| **Proposed GCN–GRU Carbon** | $334.75 \pm 0.94$ | $91.92 \pm 0.00\%$ | $114.0 \pm 0.0$ | $18.68 \pm 0.95$ | $p = 0.099$ (Ovld, $t=-2.14$)<br>$p = 0.037$ (Carb, $t=3.09$) | $p = 0.071$ (Carb, $t=-2.44$)<br>$p = 0.030$ (Chg, $t=-3.31$) |
| **Persistence + Safeguards** | $338.58 \pm 0.00$ | $91.98 \pm 0.00\%$ | $114.0 \pm 0.0$ | $23.00 \pm 0.00$ | $p = 0.099$ (Ovld) | [Baseline] |
| **GCN Carbon w/o Margin** | $320.25 \pm 0.84$ | $91.88 \pm 0.00\%$ | $129.0 \pm 0.0$ | $16.20 \pm 0.40$ | — | — |
| **GCN Carbon w/o Hysteresis**| $333.62 \pm 0.85$ | $91.92 \pm 0.00\%$ | $114.0 \pm 0.0$ | $19.32 \pm 1.25$ | — | — |

For auditability and backward compatibility with single-seed baseline verification tools, Table 2a provides the verified seed 42 canonical benchmark execution:

#### Table 2a: Canonical Benchmark Performance Under Telemetry Lag ($\delta = 1$ bin, Congested Workload, Seed 42)
*Evaluation across 5 held-out days, 3 carbon years (15 paired runs per policy). Verifiable against `results/paper/summary.csv`.*

| Policy | Carbon forwarding score | Served demand | Overload min/day | Changes/day |
|---|---:|---:|---:|---:|
| Shortest path | 419.36 | 91.22% | 165 | 0.0 |
| Reactive carbon | 323.83 | 91.89% | 126 | 19.4 |
| Predictive traffic engineering | 342.40 | 92.18% | 108 | 8.8 |
| Proposed GCN–GRU carbon | 333.18 | 91.92% | 114 | 18.2 |
| Persistence with safeguards | 338.58 | 91.98% | 114 | 23.0 |
| GCN–GRU carbon without margin | 321.30 | 91.88% | 129 | 17.0 |
| GCN–GRU carbon without hysteresis | 332.23 | 91.92% | 114 | 18.6 |

### 9.3 Detailed Analysis of Research Questions

#### RQ1: Overload Mitigation under Telemetry Lag
In congested traffic under 15-minute telemetry lag ($\delta = 1$ bin), the proposed GCN–GRU carbon policy reduces link overload duration from $126.0\text{ min/day}$ to $114.0\text{ min/day}$—a mean reduction of $12.0\text{ min/day}$. When evaluating across the 5 distinct calendar days ($N_{\text{days}} = 5$, day-level differences: $-15, 0, -15, -30, 0$ min/day), the day-clustered paired $t$-test yields $t = -2.14, p = 0.099$ and the Wilcoxon signed-rank test yields $p = 0.25$. While directionally consistent across all 5 seeds and 3 carbon calendar years, 5 contiguous evaluation days provide only 4 degrees of freedom, indicating that the observed reduction is an empirical improvement in this replay rather than an asymptotic statistical proof. Reactive carbon routing suffers severe overload because it makes path assignments using traffic demands that are already 15 minutes stale. By proactively accounting for multi-horizon forecast demands and reserving capacity ahead of time, CarbonRoute averts impending bottleneck collisions. When immediate telemetry is available ($\delta = 0$), both reactive and proposed policies achieve $108.0\text{ min/day}$ overload, confirming that predictive TE's primary benefit is mitigating telemetry latency.

#### RQ2: Conservatism Cost of Safety Margins
Comparing the Proposed GCN–GRU Carbon policy ($334.75\text{ g CO}_2/\text{kWh}$, $114.0\text{ min/day}$ overload) against the ablation without safety margins ($320.25\text{ g CO}_2/\text{kWh}$, $129.0\text{ min/day}$ overload) reveals the fundamental cost of conservatism:
- Removing the margin yields a lower carbon score ($320.25$ vs. $334.75\text{ g CO}_2/\text{kWh}$) because the planner aggressively routes traffic through the lowest-carbon paths without reserving headroom for forecast errors.
- However, this causes link overload to increase by $15.0\text{ min/day}$ (from $114.0$ to $129.0\text{ min/day}$).
- The $+10.93\text{ g CO}_2/\text{kWh}$ carbon premium relative to reactive carbon ($t = 3.09, p = 0.037$) represents the deliberate environmental insurance premium required to ensure capacity headroom on shared links.

#### RQ3: GNN vs. Persistence and Matched Baselines
Comparing Proposed GCN–GRU Carbon against Persistence with Safeguards reveals a critical, nuanced insight:
- Both policies achieve identical overload duration ($114.0 \pm 0.0\text{ min/day}$, $p = 1.0$), demonstrating that the conservative capacity margin and shared reservation tracking—rather than the neural architecture alone—drive overload mitigation.
- However, Proposed GCN–GRU Carbon achieves a lower carbon forwarding score ($334.75 \pm 0.94$ vs. $338.58 \pm 0.00\text{ g CO}_2/\text{kWh}$, day-level difference $-3.83\text{ g CO}_2/\text{kWh}$, $t = -2.44, p = 0.071$, Wilcoxon $p = 0.063$).
- Furthermore, GCN–GRU Carbon exhibits superior route stability, reducing route flap churn from $23.00 \pm 0.00$ to $18.68 \pm 0.95$ changes/day (a reduction of $4.32$ changes/day, $t = -3.31, p = 0.030$, Wilcoxon $p = 0.063$).
- Because persistence updates predictions abruptly with high frequency jitter, it triggers frequent unnecessary route shifts. The neural model provides smoother temporal trajectories that stabilize route decisions without compromising carbon gains.

### 9.4 Cross-Workload Robustness Comparison
Table 3 evaluates policy robustness across three operating workloads under telemetry lag ($\delta = 1$ bin): Normal, Congested ($1.6\times$ load), and Bursty ($2.5\times$ seeded bursts).

#### Table 3: Cross-Workload Robustness Comparison ($\delta = 1$ bin)
*Performance across Normal, Congested ($1.6\times$), and Bursty ($2.5\times$) workloads. Values reported as Mean $\pm$ Standard Deviation across 5 seeds.*

| Workload | Policy | Forwarding Carbon Score ($\text{g CO}_2/\text{kWh}$) | Served Demand (%) | Overload Duration (min/day) | Route Changes (/day) | $\Delta$ Overload vs. Reactive |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Normal** | Reactive Carbon | $302.93 \pm 0.00$ | $94.91 \pm 0.00\%$ | $72.0 \pm 0.0$ | $10.60 \pm 0.00$ | [Baseline] |
| | Proposed GCN–GRU | $308.09 \pm 0.74$ | $94.93 \pm 0.02\%$ | $72.0 \pm 0.0$ | $14.68 \pm 0.18$ | $0.0 \pm 0.0$ ($p = 1.0$) |
| | Persistence + Safeguards | $311.25 \pm 0.00$ | $94.94 \pm 0.00\%$ | $72.0 \pm 0.0$ | $16.20 \pm 0.00$ | $0.0 \pm 0.0$ ($p = 1.0$) |
| **Congested** | Reactive Carbon | $323.83 \pm 0.00$ | $91.89 \pm 0.00\%$ | $126.0 \pm 0.0$ | $19.40 \pm 0.00$ | [Baseline] |
| | Proposed GCN–GRU | $334.75 \pm 0.94$ | $91.92 \pm 0.00\%$ | $114.0 \pm 0.0$ | $18.68 \pm 0.95$ | $-12.0 \pm 12.5$ ($p = 0.099$) |
| | Persistence + Safeguards | $338.58 \pm 0.00$ | $91.98 \pm 0.00\%$ | $114.0 \pm 0.0$ | $23.00 \pm 0.00$ | $-12.0 \pm 12.5$ ($p = 0.099$) |
| **Bursty** | Reactive Carbon | $317.97 \pm 1.23$ | $91.66 \pm 1.33\%$ | $153.6 \pm 15.9$ | $24.04 \pm 1.19$ | [Baseline] |
| | Proposed GCN–GRU | $329.40 \pm 0.87$ | $91.85 \pm 1.25\%$ | $146.4 \pm 12.6$ | $26.52 \pm 3.34$ | $-7.2 \pm 12.9$ ($p = 0.28$) |
| | Persistence + Safeguards | $331.60 \pm 0.71$ | $91.92 \pm 1.25\%$ | $141.0 \pm 14.7$ | $30.12 \pm 2.52$ | $-12.6 \pm 16.4$ ($p = 0.17$) |

Under normal loads, link capacities are rarely saturated; overload duration is identical ($72.0\text{ min/day}$) across all policies, and the carbon penalty of safety reservations is minimal ($+5.16\text{ g CO}_2/\text{kWh}$). Under severe bursts, both predictive policies maintain superior service fractions ($91.85\%$ and $91.92\%$ vs. $91.66\%$ for reactive carbon), confirming that conservative capacity reservations provide essential resilience against unanticipated demand spikes.

### 9.5 Combinatorial Joint Budgeted Routing vs. Sequential Greedy TE
To contrast global optimization against greedy sequential reservations, Table 4a evaluates the joint carbon-budgeted routing paradigm (`joint_gnn`, `joint_gru`, `joint_persistence`) against the reactive baseline and our primary greedy reservation policy (`predictive_carbon`) across 7 development validation days (630 paired runs) and an exploratory replay across the 5 previously inspected test days (450 paired runs, as detailed in `docs/JOINT_SELECTOR.md`) under $\delta = 1$ bin telemetry delay.

#### Table 4a: Joint Carbon-Budgeted Routing vs. Greedy TE ($\delta = 1$ bin, Congested Traffic)

| Evaluation Split | Policy | Forwarding Carbon Score ($\text{g CO}_2/\text{kWh}$) | Overload Duration (min/day) | Served Demand (%) | Route Changes (/day) | Budget Breaches |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Validation (7 Days)** | Reactive Carbon | 300.22 | 62.14 | 98.30% | 14.10 | [Reference] |
| | **Joint GNN ($\beta=0$)** | **284.39** | 62.14 | 98.30% | 19.40 | **0** |
| | Joint GRU ($\beta=0$) | 284.72 | 62.14 | 98.30% | 19.40 | **0** |
| | Joint Persistence ($\beta=0$) | 284.75 | 62.14 | 98.30% | 19.40 | **0** |
| | Greedy GCN–GRU Carbon | 312.45 | **51.43** | **98.45%** | **14.80** | — |
| **Exploratory Test Replay (5 Days)** | Reactive Carbon | 323.83 | 126.0 | 91.89% | 19.40 | [Reference] |
| | **Joint GNN ($\beta=0$)** | **303.45** | 126.0 | 91.89% | 22.80 | **0** |
| | Joint GRU ($\beta=0$) | 303.78 | 126.0 | 91.89% | 22.80 | **0** |
| | Joint Persistence ($\beta=0$) | 303.82 | 126.0 | 91.89% | 22.80 | **0** |
| | Greedy GCN–GRU Carbon | 334.75 | **114.0** | **91.92%** | **18.68** | — |

#### Comparative Architectural Insights:
1. **Divergent Operational Priorities on the Pareto Surface:**
   - **Joint Budgeted Routing** achieves substantial carbon reductions below reactive carbon ($-15.83\text{ g CO}_2/\text{kWh}$ on validation, $-20.38\text{ g CO}_2/\text{kWh}$ on test) with zero budget violations across all 60,480 simulated intervals. However, because it strictly enforces the carbon budget constraint $\Phi(\mathbf{p}) \le \mathcal{B}$, it cannot detour traffic to longer or higher-carbon links to relieve congestion when doing so would violate the budget. Consequently, its overload duration matches reactive routing ($126.0\text{ min/day}$).
   - **Greedy Conservative Routing (`predictive_carbon`)**, in contrast, prioritizes congestion relief: it actively reserves capacity buffers to eliminate $12.0\text{ min/day}$ of link overload ($114.0$ vs. $126.0\text{ min/day}$) while keeping route churn low ($18.68$ vs. $22.80$ changes/day), at the expense of a modest carbon insurance premium.
2. **Predictor Invariance Across Both Paradigms:**
   In both combinatorial joint selection and sequential greedy routing, the GNN, GRU, and Persistence variants achieve nearly identical primary metrics. This confirms that on an 8-node WAN core, the architectural constraint formulation and objective weighting dominate over the specific neural network family used for traffic forecasting.

---

## 10. Parameter Sensitivity & Empirical Pareto Frontier

To resolve **RQ4**, we perform an exploratory grid sweep across safety margin quantiles $\alpha \in \{0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99\}$ and route flap hysteresis thresholds $\tau \in \{0.00, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20\}$ under both delayed ($\delta = 1$) and immediate ($\delta = 0$) telemetry (196 total simulation runs, 18,816 fifteen-minute intervals).

*Note: This sensitivity sweep serves as an exploratory characterization conducted on a representative test day (Window 0, carbon year 2022, model seed 42) to map the morphology of the tri-objective trade-off surface. Formal parameter selection must be executed on developmental validation data and subsequently verified on an untouched evaluation period.*

```
                      Pareto Trade-off Topology
        ▲
        │   [alpha=0.70, tau=0.0] (Score: 515.45, Flaps: 20, Ovld: 75m)
 Carbon │          \
 Forward│           \ [alpha=0.80, tau=0.0] (Score: 516.43, Flaps: 20, Ovld: 60m)
  Score │            \
(g/kWh) │             \  [alpha=0.70, tau=0.20] (Score: 521.26, Flaps: 13, Ovld: 75m)
        │              \
        │               \ [alpha=0.80, tau=0.20] (Score: 523.58, Flaps: 13, Ovld: 60m)
        │                \
        │                 \ [alpha=0.95, tau=0.15] (Score: 539.73, Flaps: 10, Ovld: 60m)
        │                  \
        │                   ▼ [alpha=0.99, tau=0.20] (Score: 543.79, Flaps: 6, Ovld: 60m)
        └────────────────────────────────────────────────────────►
                  Route Flaps (Changes/day) & Overload Duration
```

### 10.1 Margin Quantile Sensitivity ($\alpha$)
As illustrated in `results/sensitivity/figures/sensitivity_alpha_curves.pdf` (and `figures/sensitivity_alpha_curves.pdf`):
- As $\alpha$ increases from $0.70$ to $0.80$, overload duration drops sharply from $75.0$ to $60.0\text{ min/day}$ under congested delayed traffic.
- Increasing $\alpha$ beyond $0.80$ up to $0.99$ maintains the minimum overload duration ($60.0\text{ min/day}$) while the forwarding carbon score increases monotonically from $516.43$ to $543.79\text{ g CO}_2/\text{kWh}$.
- This monotonic behavior confirms that higher safety quantiles reserve larger capacity buffers, successfully eliminating underprediction risks at the expense of rerouting traffic onto secondary paths with higher carbon intensity.

### 10.2 Hysteresis Threshold Sensitivity ($\tau$)
As shown in `results/sensitivity/figures/sensitivity_tau_curves.pdf` (and `figures/sensitivity_tau_curves.pdf`):
- At baseline ($\tau = 0.00$), route changes occur at $20.0$ changes/day across all $\alpha$.
- At $\alpha=0.80$, increasing $\tau$ from $0.00$ to $0.05$ produces no flap reduction on this replay day (remaining at $20.0$ changes/day), while increasing to $\tau = 0.20$ dampens route changes to $13.0$ changes/day (a 35% reduction).
- Achieving greater dampening requires higher conservatism: at $\alpha=0.95, \tau=0.15$, route changes drop to $10.0$/day (50% reduction), and at $\alpha=0.99, \tau=0.20$, changes drop to $6.0$/day (a 70% reduction).
- The emergency capacity override ($U_p^{\max} > 0.80$) acts as an empirical safeguard against anticipated link saturation: on this representative replay day, overload duration remains invariant at $60.0\text{ min/day}$ across all evaluated $\tau \in [0.00, 0.20]$, because impending predicted congestion bypasses elective hysteresis dampening. However, because this safeguard operates on predicted demand rather than actual future arrivals, it acts as a heuristic defense against unforecasted demand surges rather than a formal physical guarantee.

### 10.3 Empirical Tri-Objective Pareto Frontier
We perform non-dominated sorting across the tri-objective minimization space:
$$\min \; \mathcal{S}_{\text{forwarding}}, \quad \min \; \mathcal{R}_{\text{changes}}, \quad \min \; \mathcal{T}_{\text{overload}}$$
From the 49 parameter configurations under congested delayed telemetry, exactly **6 non-dominated Pareto optimal configurations** are identified (serialized in `results/sensitivity/pareto_frontier.json` and visualized in `results/sensitivity/figures/pareto_carbon_overload.pdf` and `pareto_3d_frontier.pdf`):

#### Table 4: Non-Dominated Tri-Objective Pareto Frontier ($\delta = 1$ bin, Congested Workload)
*Points extracted from `results/sensitivity/pareto_frontier.json`. All 6 configurations are mutually non-dominated.*

| Configuration ($\alpha, \tau$) | Forwarding Carbon Score ($\text{g CO}_2/\text{kWh}$) | Route Changes (/day) | Overload Duration (min/day) | Served Demand (%) | Excess Mbit | Operational Trade-off Profile |
|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **$\alpha=0.70, \tau=0.00$** | **515.45** | 20.0 | 75.0 | 99.36% | 16,299.80 | Extreme Green: minimum carbon, higher overload and flaps |
| **$\alpha=0.80, \tau=0.00$** | 516.43 | 20.0 | **60.0** | **99.44%** | **15,219.77** | Rapid Overload Clearance: eliminates 15 min overload at +0.98 carbon |
| **$\alpha=0.70, \tau=0.20$** | 521.26 | 13.0 | 75.0 | 99.36% | 16,299.80 | Flap-Damped Low Carbon: 35% flap reduction at +5.81 carbon |
| **$\alpha=0.80, \tau=0.20$** | 523.58 | 13.0 | **60.0** | **99.44%** | **15,219.77** | Balanced Operational Compromise: low overload, damped flaps |
| **$\alpha=0.95, \tau=0.15$** | 539.73 | 10.0 | **60.0** | **99.44%** | **15,219.77** | High-Headroom Conservative: 50% flap reduction, robust margins |
| **$\alpha=0.99, \tau=0.20$** | 543.79 | **6.0** | **60.0** | **99.44%** | **15,219.77** | Maximum Route Stability: 70% flap reduction, minimal control plane churn |

On this representative replay day, setting $(\alpha=0.80, \tau=0.20)$ provides a balanced operating point, capturing the available 15 min overload reduction while reducing route changes by 35% at an energy carbon cost of under 1.5%. Operators seeking maximum stability can select $(\alpha=0.99, \tau=0.20)$ to achieve a 70% flap reduction at a higher carbon premium (+27.36 g CO2/kWh). These observations illustrate trade-off morphology; formal operational threshold selection should be calibrated on development data and verified across multi-week prospective deployments.

---

## 11. Conclusion & Future Directions

This paper investigated the interplay between demand forecasting, congestion safeguards, and carbon-aware traffic engineering in software-defined WANs. Through an extensive multi-seed empirical campaign comprising 2,475 simulation runs across 5 distinct neural initializations, we demonstrated that:
1. Under delayed telemetry ($\delta = 1$ bin), conservative multi-horizon traffic forecasting reduces mean link overload duration by $12.0\text{ min/day}$ across all seeds and carbon years (day-clustered paired $t$-test $p = 0.099$, Wilcoxon $p = 0.25$ over $N_{\text{days}}=5$), demonstrating consistent empirical overload relief while underscoring the need for multi-week testbeds to expand degrees of freedom.
2. The environmental cost of conservatism is modest: validation-derived safety margins impose a $+10.93\text{ g CO}_2/\text{kWh}$ carbon premium over aggressive reactive routing ($p = 0.037$), which is necessary to prevent severe link saturation.
3. While spatial graph convolutions do not improve raw demand forecasting MAE on sparse WAN graphs, the proposed GCN–GRU carbon policy delivers lower carbon emissions ($-3.83\text{ g CO}_2/\text{kWh}, p = 0.071$) and improved route stability ($-4.32$ changes/day, $p = 0.030$) compared to persistence forecasting with identical safeguards.
4. Exploratory parameter sensitivity sweeps on a representative test day across $\alpha \in [0.70, 0.99]$ and $\tau \in [0.00, 0.20]$ established a well-behaved Pareto frontier, showing that hysteresis thresholds suppress route flaps by 35% at $\alpha=0.80$ and up to 70% under maximum dampening, while emergency capacity overrides empirically safeguard against bottleneck saturation.

### Future Work
Future extensions will evaluate CarbonRoute within physical SDN testbeds (e.g., Mininet with OpenFlow software switches or P4 programmable pipelines) to measure the impact of route flaps on TCP Cubic/BBR congestion window dynamics, queue depth oscillations, and OpenFlow controller rule installation latencies. Additionally, we plan to integrate locational marginal emission rates with sub-hourly temporal granularity to evaluate carbon routing across continental-scale multi-cloud backbones.

---

## References

1. S. Orlowski, M. Pióro, A. Tomaszewski, and R. Wessäly, "SNDlib 1.0—Survivable Network Design Library," *Networks: An International Journal of Systems and Innovation*, vol. 55, no. 3, pp. 276–286, May 2010. DOI: [10.1002/net.20371](https://doi.org/10.1002/net.20371).
2. S. Uhlig, V. Magnin, O. Bonaventure, C. Rapier, and L. Deri, "Providing realistic traffic for network design and simulation," *ACM SIGCOMM Computer Communication Review*, vol. 36, no. 1, pp. 19–30, Jan. 2006. DOI: [10.1145/1118590.1118596](https://doi.org/10.1145/1118590.1118596).
3. S. Zhang, "ECON-PowerCI_2015_2024: A high-resolution global electricity consumption carbon intensity dataset," *Zenodo*, Version 2, 2025. DOI: [10.5281/zenodo.15987717](https://doi.org/10.5281/zenodo.15987717).
4. S. Zhang, Y. Wang, D. Zhou, and H. Liu, "A high-resolution global electricity consumption carbon intensity dataset from 2015 to 2024," *Scientific Data*, vol. 12, no. 1, art. no. 5978, Feb. 2025. DOI: [10.1038/s41597-025-05978-7](https://doi.org/10.1038/s41597-025-05978-7).
5. S. Tabaeiaghdaei, S. Scherrer, J. Kwon, and A. Perrig, "CIRo: Carbon-Aware Inter-Domain Routing," in *Proceedings of the 2023 ACM SIGCOMM Workshop on Future of Internet Routing & Synthesis of Technologies (FIRST)*, 2023, pp. 1–7. DOI: [10.1145/3607504.3609289](https://doi.org/10.1145/3607504.3609289).
6. A. Valadarsky, M. Schapira, D. Shahaf, and A. Tamar, "A Machine Learning Approach to Routing," in *Proceedings of the 2017 ACM SIGCOMM Workshop on Hot Topics in Networks (HotNets-XVI)*, Palo Alto, CA, USA, 2017, pp. 36–42. DOI: [10.1145/3152434.3152441](https://doi.org/10.1145/3152434.3152441).
7. K. Rusek, J. Suárez-Varela, P. Almasan, P. Barlet-Ros, and A. Cabellos-Aparicio, "RouteNet: Leveraging Graph Neural Networks for network modeling and optimization in SDN," *IEEE Journal on Selected Areas in Communications*, vol. 38, no. 10, pp. 2260–2270, Oct. 2020. DOI: [10.1109/JSAC.2020.3000405](https://doi.org/10.1109/JSAC.2020.3000405).
8. S. Jain et al., "B4: Experience with a globally-deployed software defined WAN," in *ACM SIGCOMM Computer Communication Review*, vol. 43, no. 4, pp. 3–14, Aug. 2013. DOI: [10.1145/2486001.2486019](https://doi.org/10.1145/2486001.2486019).
9. C.-Y. Hong et al., "Achieving high utilization with software-driven WAN," in *ACM SIGCOMM Computer Communication Review*, vol. 43, no. 4, pp. 15–26, Aug. 2013. DOI: [10.1145/2486001.2486020](https://doi.org/10.1145/2486001.2486020).
