# Computational Consciousness Frameworks: Python Implementation Guide

This comprehensive reference provides **working Python code** for implementing key computational consciousness frameworks. Each section includes installation commands, annotated code, mathematical intuition, and practical guidance.

---

## 1. Integrated Information Theory (IIT) with PyPhi

### Installation
```bash
pip install pyphi
# Windows users: conda install -c wmayner pyphi
# For interactive use: pip install ipython
```

### Computing Φ (Phi) step by step

```python
import pyphi
import numpy as np

# ===== STEP 1: Define the Transition Probability Matrix (TPM) =====
# The TPM captures system dynamics: how state at time t determines state at t+1
# Format: 2D state-by-node form
# - Rows = all possible states at t (2^n rows for n nodes, little-endian order)
# - Columns = probability of each node being ON at t+1
#
# This network: Node A = OR gate, Node B = COPY gate, Node C = XOR gate

tpm = np.array([
    [0, 0, 0],  # From (0,0,0): All nodes OFF next
    [0, 0, 1],  # From (1,0,0): Only C ON
    [1, 0, 1],  # From (0,1,0): A and C ON
    [1, 0, 0],  # From (1,1,0): Only A ON
    [1, 1, 0],  # From (0,0,1): A and B ON
    [1, 1, 1],  # From (1,0,1): All ON
    [1, 1, 1],  # From (0,1,1): All ON
    [1, 1, 0]   # From (1,1,1): A and B ON
])

# ===== STEP 2: Define Connectivity Matrix (speeds up computation) =====
# Entry (i,j) = 1 if node i connects to node j
cm = np.array([
    [0, 0, 1],  # A → C
    [1, 0, 1],  # B → A, C
    [1, 1, 0]   # C → A, B
])

# ===== STEP 3: Create Network and Subsystem =====
labels = ('A', 'B', 'C')
network = pyphi.Network(tpm, connectivity_matrix=cm, node_labels=labels)
state = (1, 0, 0)  # Current state: A=ON, B=OFF, C=OFF
subsystem = pyphi.Subsystem(network, state)

# ===== STEP 4: Compute Φ (integrated information) =====
phi_value = pyphi.compute.phi(subsystem)
print(f"Φ (Big Phi) = {phi_value}")  # Output: Φ = 2.3125

# ===== STEP 5: Full System Irreducibility Analysis =====
sia = pyphi.compute.sia(subsystem)
print(f"Number of concepts: {len(sia.ces)}")
print(f"MIP (Minimum Information Partition): {sia.cut}")
print(f"Concept φ values: {sia.ces.phis}")
```

**Interpreting outputs:**
- **Φ > 0**: System has irreducible integrated information
- **Higher Φ**: More consciousness (per IIT)
- **MIP**: The partition that makes least difference—reveals structure of integration

**Common gotchas:**
- Network size limit: ~10-12 nodes maximum (O(n⁵ · 3ⁿ) complexity)
- Memory issues: Set `pyphi.config.MAXIMUM_CACHE_MEMORY_PERCENTAGE = 25`
- Speed: Enable `pyphi.config.PARALLEL = True` and `pyphi.config.CUT_ONE_APPROXIMATION = True`

---

## 2. Active Inference with pymdp

### Installation
```bash
pip install inferactively-pymdp
```

### Defining the Generative Model (A, B, C, D matrices)

```python
import numpy as np
from pymdp import utils
from pymdp.agent import Agent
from pymdp.maths import softmax
import itertools

# ===== A MATRIX: Observation Model P(observation|state) =====
# Maps hidden states → observations. Columns must sum to 1.
# Shape: (num_observations, num_states) or object array for multi-modality

n_states = 9  # 3x3 grid world
n_obs = 9

# Identity = perfect observation (agent knows exactly where it is)
A = np.eye(n_obs, n_states)

# For noisy observations, modify columns:
# A[:, ambiguous_state] = [0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0, 0]  # Multiple possible obs

# ===== B MATRIX: Transition Model P(next_state|current_state, action) =====
# Shape: (n_states, n_states, n_actions) - columns sum to 1 per action

def create_B_matrix():
    """B[next_state, current_state, action] for 3x3 grid world."""
    grid_locs = list(itertools.product(range(3), repeat=2))
    actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
    B = np.zeros((9, 9, 5))
    
    for action_id, action in enumerate(actions):
        for curr_state, (y, x) in enumerate(grid_locs):
            if action == "UP": y = max(0, y-1)
            elif action == "DOWN": y = min(2, y+1)
            elif action == "LEFT": x = max(0, x-1)
            elif action == "RIGHT": x = min(2, x+1)
            next_state = grid_locs.index((y, x))
            B[next_state, curr_state, action_id] = 1.0  # Deterministic
    return B

B = create_B_matrix()

# ===== C MATRIX: Preferences (log-probability of preferred observations) =====
# Encodes goals: higher values = more preferred. Not normalized.

goal_location = (2, 2)  # Bottom-right corner
goal_idx = list(itertools.product(range(3), repeat=2)).index(goal_location)
C = utils.onehot(goal_idx, n_obs)  # [0,0,0,0,0,0,0,0,1]

# ===== D MATRIX: Initial State Prior P(s₀) =====
# Belief about starting state. Must sum to 1.

start_idx = 0  # Top-left corner
D = utils.onehot(start_idx, n_states)
```

### Running the Active Inference Loop

```python
from scipy.stats import entropy as scipy_entropy
from pymdp.maths import spm_log_single as log_stable

def active_inference_loop(A, B, C, D, env, T=10):
    """Complete perception-action loop with expected free energy."""
    qs = D.copy()  # Initial belief
    n_actions = B.shape[2]
    policies = list(itertools.product(range(n_actions), repeat=3))  # 3-step planning
    
    # Ambiguity of A (entropy of each column)
    H_A = np.array([scipy_entropy(A[:, s]) for s in range(A.shape[1])])
    
    obs = env.reset()
    
    for t in range(T):
        # 1. INFER STATES: Bayesian update P(s|o, prior)
        prior = B[:, :, last_action].dot(qs) if t > 0 else D
        log_likelihood = log_stable(A[obs, :])
        log_prior = log_stable(prior)
        qs = softmax(log_likelihood + log_prior)
        
        # 2. COMPUTE EXPECTED FREE ENERGY for each policy
        G = np.zeros(len(policies))
        for pi_idx, policy in enumerate(policies):
            qs_pi = qs.copy()
            for action in policy:
                qs_pi = B[:, :, action].dot(qs_pi)  # Predict next state
                qo_pi = A.dot(qs_pi)                 # Predict observation
                
                # G = Risk + Ambiguity
                risk = -qo_pi.dot(C)           # Divergence from preferences
                ambiguity = H_A.dot(qs_pi)     # Expected observation uncertainty
                G[pi_idx] += risk + ambiguity
        
        # 3. INFER POLICIES: Q(π) ∝ exp(-G)
        q_pi = softmax(-16.0 * G)  # 16.0 = precision parameter
        
        # 4. SAMPLE ACTION (first action of best policy)
        action = policies[np.argmax(q_pi)][0]
        last_action = action
        obs = env.step(action)
        
        print(f"t={t}: obs={obs}, action={action}, belief_peak={np.argmax(qs)}")
```

**Matrix summary:**
| Matrix | Shape | Normalization | Role |
|--------|-------|---------------|------|
| **A** | (obs, states) | Columns sum to 1 | How states generate observations |
| **B** | (states, states, actions) | Columns sum to 1 per action | How actions change states |
| **C** | (obs,) | Not required | Goal/reward specification |
| **D** | (states,) | Sums to 1 | Starting belief |

---

## 3. Entropy and Complexity Metrics

### Installation
```bash
pip install antropy nolds pyinform ordpy
```

### Lempel-Ziv Complexity

```python
import numpy as np
import antropy as ant

def compute_lz_complexity(signal, normalize=True):
    """
    Lempel-Ziv Complexity: Counts distinct patterns in binarized sequence.
    
    Interpretation:
    - ~0.25-0.35: Regular (sine wave, seizures)
    - ~0.70-0.85: Moderate (EEG-like)
    - ~0.85-0.95: High complexity (random noise)
    
    Consciousness: Core of Perturbational Complexity Index (PCI).
    Low during anesthesia; high during conscious wakefulness.
    """
    # Binarize: values > median become 1
    binary = (signal > np.median(signal)).astype(int)
    binary_string = ''.join(map(str, binary))
    return ant.lziv_complexity(binary_string, normalize=normalize)

# Example
np.random.seed(42)
random_signal = np.random.randn(1000)
sine_wave = np.sin(2 * np.pi * 5 * np.arange(1000) / 256)

print(f"Random noise: {compute_lz_complexity(random_signal):.4f}")  # ~0.90
print(f"Sine wave: {compute_lz_complexity(sine_wave):.4f}")         # ~0.30
```

### Sample Entropy

```python
import antropy as ant

def compute_sample_entropy(signal, order=2, tolerance=None):
    """
    Sample Entropy: Probability similar patterns stay similar.
    
    Parameters:
    - order (m): Embedding dimension. m=2 standard for physiology.
    - tolerance (r): Similarity threshold. Default = 0.2 * std(signal).
    
    Interpretation:
    - ~0.0-0.5: Very regular (predictable)
    - ~1.5-2.5: High complexity (healthy physiology)
    - inf: No matches found (increase tolerance or data length)
    
    Data requirement: N > 10^m samples (m=2 needs >100)
    """
    if tolerance is None:
        tolerance = 0.2 * np.std(signal)
    return ant.sample_entropy(signal, order=order, tolerance=tolerance)

print(f"Sample Entropy: {compute_sample_entropy(random_signal):.4f}")
```

### Permutation Entropy

```python
import antropy as ant

def compute_permutation_entropy(signal, order=3, delay=1, normalize=True):
    """
    Permutation Entropy: Distribution of ordinal patterns.
    
    Parameters:
    - order (m): Pattern length. m=3-5 typical.
    - delay (tau): Time lag between samples.
    
    Interpretation (normalized 0-1):
    - ~0.99: Maximum complexity (random)
    - ~0.90-0.98: Healthy EEG
    - ~0.40-0.50: Periodic (sine wave)
    
    Data requirement: N > m! × 10 samples (m=4 needs >240)
    """
    return ant.perm_entropy(signal, order=order, delay=delay, normalize=normalize)
```

### Transfer Entropy

```python
from pyinform import transfer_entropy
import numpy as np

def compute_transfer_entropy(source, target, k=2):
    """
    Transfer Entropy: Directed information flow X → Y.
    
    Parameters:
    - source, target: INTEGER arrays (must be discrete!)
    - k: History length
    
    Interpretation:
    - TE(X→Y) ≠ TE(Y→X) - asymmetric measure
    - TE > 0: Information flows source → target
    
    CRITICAL: Data must be discretized (use median split for continuous).
    """
    source = np.asarray(source, dtype=np.int32)
    target = np.asarray(target, dtype=np.int32)
    return transfer_entropy(source, target, k=k)

# Create coupled signals
X = np.random.randint(0, 2, 1000)
Y = np.roll(X, 1)  # Y follows X with 1-step lag

print(f"TE(X→Y): {compute_transfer_entropy(X, Y):.4f}")  # High
print(f"TE(Y→X): {compute_transfer_entropy(Y, X):.4f}")  # Low/zero
```

---

## 4. Dynamical Systems Analysis

### Installation
```bash
pip install nolds pynamical scipy matplotlib
```

### Lyapunov Exponent Computation

```python
import numpy as np
import nolds

def logistic_map(r, x0, n):
    """Generate logistic map: x(n+1) = r*x(n)*(1-x(n))"""
    x = np.zeros(n)
    x[0] = x0
    for i in range(n-1):
        x[i+1] = r * x[i] * (1 - x[i])
    return x

# Chaotic regime (r=3.9)
chaos_data = logistic_map(r=3.9, x0=0.1, n=5000)
lyap = nolds.lyap_r(chaos_data, emb_dim=3, lag=1, min_tsep=10)
print(f"Lyapunov Exponent: {lyap:.4f}")

# Interpretation:
# λ > 0: CHAOTIC (sensitive to initial conditions)
# λ ≈ 0: EDGE OF CHAOS (optimal for computation)
# λ < 0: STABLE (converges to attractor)
```

### Takens Embedding (Attractor Reconstruction)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def delay_embedding(data, dim, tau):
    """
    Takens time-delay embedding.
    Reconstructs attractor from single variable.
    
    Parameters:
    - dim: Embedding dimension (use False Nearest Neighbors to choose)
    - tau: Time delay (use first minimum of mutual information)
    """
    N = len(data) - (dim - 1) * tau
    embedded = np.zeros((N, dim))
    for i in range(dim):
        embedded[:, i] = data[i * tau : i * tau + N]
    return embedded

# Generate Lorenz attractor (use x-component only)
def lorenz(t, xyz, sigma=10, rho=28, beta=8/3):
    x, y, z = xyz
    return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

sol = solve_ivp(lorenz, (0, 100), [1, 1, 1], t_eval=np.linspace(0, 100, 10000))
x_lorenz = sol.y[0, 1000:]  # Skip transient

# Reconstruct 3D attractor from x alone
tau = 15  # From mutual information analysis
embedded = delay_embedding(x_lorenz, dim=3, tau=tau)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
ax.plot(embedded[:, 0], embedded[:, 1], embedded[:, 2], lw=0.3, alpha=0.7)
ax.set_title(f"Reconstructed Attractor (τ={tau})")
plt.show()
```

### Bifurcation Diagram

```python
import numpy as np
import matplotlib.pyplot as plt

n_r = 10000
r = np.linspace(2.5, 4.0, n_r)
iterations, last = 1000, 100
x = 1e-5 * np.ones(n_r)
lyapunov = np.zeros(n_r)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

for i in range(iterations):
    x = r * x * (1 - x)  # Logistic map
    lyapunov += np.log(np.abs(r - 2 * r * x))  # Accumulate LE
    if i >= (iterations - last):
        ax1.plot(r, x, ',k', alpha=0.25)

ax1.set_ylabel('Attractor values')
ax1.set_title('Bifurcation Diagram: Logistic Map')

lyapunov /= iterations
ax2.plot(r[lyapunov < 0], lyapunov[lyapunov < 0], '.k', ms=0.5, alpha=0.5)
ax2.plot(r[lyapunov >= 0], lyapunov[lyapunov >= 0], '.r', ms=0.5, alpha=0.5)
ax2.axhline(0, color='gray', lw=0.5)
ax2.set_xlabel('Growth rate r')
ax2.set_ylabel('Lyapunov exponent')
ax2.set_title('Red = Chaos, Black = Periodic')
plt.tight_layout()
plt.show()
```

---

## 5. Self-Reference and Metacognition

### Installation
```bash
pip install torch bayesian-torch
```

### Confidence Calibration (Temperature Scaling)

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

class TemperatureScaling(nn.Module):
    """
    METACOGNITIVE CALIBRATION: Makes networks "know what they don't know."
    Adjusts confidence to match actual accuracy.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, x):
        logits = self.model(x)
        # Higher T = softer probabilities = less overconfident
        return logits / self.temperature
    
    def calibrate(self, val_loader, lr=0.01, max_iter=50):
        """Learn optimal temperature on validation set."""
        self.temperature.requires_grad = True
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        logits_list, labels_list = [], []
        with torch.no_grad():
            for x, y in val_loader:
                logits_list.append(self.model(x))
                labels_list.append(y)
        
        logits, labels = torch.cat(logits_list), torch.cat(labels_list)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / self.temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        print(f'Optimal temperature: {self.temperature.item():.3f}')
```

### Expected Calibration Error

```python
import numpy as np

def expected_calibration_error(confidences, predictions, labels, n_bins=10):
    """
    ECE: How well does confidence match accuracy?
    Perfect calibration: ECE = 0
    
    If model says 80% confident, it should be right 80% of time.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if in_bin.mean() > 0:
            accuracy = (predictions[in_bin] == labels[in_bin]).mean()
            avg_conf = confidences[in_bin].mean()
            ece += np.abs(accuracy - avg_conf) * in_bin.mean()
    
    return ece  # Lower = better calibrated
```

### Self-Modeling: Bayesian Neural Networks

```python
from bayesian_torch.layers import LinearReparameterization
import torch
import torch.nn as nn

class BayesianMLP(nn.Module):
    """
    Bayesian NN: Learns distributions over weights, not point estimates.
    Distinguishes epistemic (model) vs aleatoric (data) uncertainty.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = LinearReparameterization(input_dim, hidden_dim,
            prior_mean=0, prior_variance=1, posterior_mu_init=0, posterior_rho_init=-3)
        self.fc2 = LinearReparameterization(hidden_dim, output_dim,
            prior_mean=0, prior_variance=1, posterior_mu_init=0, posterior_rho_init=-3)
        self.relu = nn.ReLU()
    
    def predict_with_uncertainty(self, x, n_samples=50):
        """Returns prediction + epistemic + aleatoric uncertainty."""
        self.eval()
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                out, _ = self.fc1(x)
                out = self.relu(out)
                out, _ = self.fc2(out)
                predictions.append(torch.softmax(out, dim=-1))
        
        preds = torch.stack(predictions)
        mean_pred = preds.mean(dim=0)
        
        # Total uncertainty
        total = -(mean_pred * torch.log(mean_pred + 1e-10)).sum(dim=-1)
        # Aleatoric (inherent noise)
        aleatoric = -(preds * torch.log(preds + 1e-10)).sum(dim=-1).mean(dim=0)
        # Epistemic (model uncertainty)
        epistemic = total - aleatoric
        
        return mean_pred, epistemic, aleatoric
```

---

## 6. Game Theory Simulations

### Installation
```bash
pip install axelrod nashpy
```

### Iterated Prisoner's Dilemma Tournament

```python
import axelrod as axl
import matplotlib.pyplot as plt

# Create tournament with built-in strategies
players = [
    axl.Cooperator(),        # Always cooperates
    axl.Defector(),          # Always defects
    axl.TitForTat(),         # Mirror opponent's last move
    axl.Grudger(),           # Defect forever after first defection
    axl.Random(0.5),         # 50% random
    axl.WinStayLoseShift(),  # Pavlov: repeat if won, switch if lost
]

tournament = axl.Tournament(players, turns=200, repetitions=10, seed=42)
results = tournament.play()

print("Rankings:", results.ranked_names)
print("Cooperation rates:", dict(zip(results.ranked_names, results.cooperating_rating)))

# Visualize
plot = axl.Plot(results)
plot.boxplot()
plt.title("Tournament Scores")
plt.show()
```

### Custom Strategy

```python
from axelrod import Action, Player
C, D = Action.C, Action.D

class ForgivingTitForTat(Player):
    """TitForTat that occasionally forgives defections."""
    name = "Forgiving TFT"
    classifier = {'memory_depth': 1, 'stochastic': True}
    
    def __init__(self, forgiveness_prob=0.1):
        super().__init__()
        self.forgiveness_prob = forgiveness_prob
    
    def strategy(self, opponent):
        if len(self.history) == 0:
            return C
        if opponent.history[-1] == D:
            return C if self._random.random() < self.forgiveness_prob else D
        return C
```

### Moran Process (Evolutionary Dynamics)

```python
import axelrod as axl

players = [axl.Defector()]*3 + [axl.Cooperator()]*3 + [axl.TitForTat()]*3

mp = axl.MoranProcess(players, turns=200, seed=42)
populations = mp.play()

print(f"Winner: {mp.winning_strategy_name}")
print(f"Generations: {len(populations)}")

mp.populations_plot()
plt.title("Moran Process: Strategy Evolution")
plt.show()
```

### Replicator Dynamics

```python
import nashpy as nash
import numpy as np
import matplotlib.pyplot as plt

# Prisoner's Dilemma payoffs
A = np.array([[3, 0], [5, 1]])  # [CC, CD; DC, DD]
game = nash.Game(A)

y0 = np.array([0.9, 0.1])  # 90% cooperators
timepoints = np.linspace(0, 10, 1000)
trajectory = game.replicator_dynamics(y0=y0, timepoints=timepoints)

plt.plot(timepoints, trajectory[:, 0], label='Cooperators')
plt.plot(timepoints, trajectory[:, 1], label='Defectors')
plt.xlabel('Time'); plt.ylabel('Population')
plt.title('Replicator Dynamics (Defection Dominates)')
plt.legend(); plt.show()
```

---

## 7. Neural Complexity and Brain Network Metrics

### Installation
```bash
pip install networkx bctpy python-louvain leidenalg python-igraph
```

### Complete Brain Network Analysis

```python
import numpy as np
import networkx as nx
import bct  # Brain Connectivity Toolbox

def analyze_brain_network(matrix, threshold=0.1):
    """Complete analysis pipeline for connectivity matrix."""
    # Preprocess
    matrix = np.abs(matrix)
    np.fill_diagonal(matrix, 0)
    matrix[matrix < threshold] = 0
    
    G = nx.from_numpy_array(matrix)
    G.remove_edges_from(nx.selfloop_edges(G))
    
    results = {}
    
    # CLUSTERING: Local connectivity (segregation)
    results['mean_clustering'] = nx.average_clustering(G, weight='weight')
    
    # PATH LENGTH: Integration efficiency
    if nx.is_connected(G):
        results['path_length'] = nx.average_shortest_path_length(G)
    results['global_efficiency'] = nx.global_efficiency(G)
    
    # SMALL-WORLDNESS: Balance of integration/segregation
    # sigma > 1 indicates small-world (optimal for consciousness)
    try:
        results['sigma'] = nx.sigma(G, niter=5, nrand=5, seed=42)
    except:
        results['sigma'] = None
    
    # MODULARITY: Community structure
    C = bct.clustering_coef_wu(matrix)
    ci, Q = bct.community_louvain(matrix)
    results['modularity_Q'] = Q
    results['n_communities'] = len(np.unique(ci))
    
    # PARTICIPATION COEFFICIENT: Inter-modular connectivity
    P = bct.participation_coef(matrix, ci)
    results['mean_participation'] = np.mean(P)
    
    # RICH CLUB: Hub interconnectedness
    results['rich_club'] = nx.rich_club_coefficient(G, normalized=False)
    
    return results

# Interpretation Guide:
# - High clustering + short paths = SMALL-WORLD (optimal for consciousness)
# - High modularity (Q > 0.3) = strong functional specialization
# - High participation = connector hubs linking modules
# - Rich club = integrative backbone for global workspace
```

### Integration-Segregation Balance

```python
def integration_segregation_balance(matrix, communities):
    """
    Compute balance between within-module and between-module connectivity.
    
    IIT perspective: Optimal consciousness requires BOTH:
    - High integration (between-module)
    - High segregation (within-module specialization)
    """
    n = matrix.shape[0]
    within, between = 0, 0
    
    for i in range(n):
        for j in range(i+1, n):
            if matrix[i,j] > 0:
                if communities[i] == communities[j]:
                    within += matrix[i,j]
                else:
                    between += matrix[i,j]
    
    total = within + between
    return {
        'segregation_ratio': within / total if total else 0,
        'integration_ratio': between / total if total else 0,
        'balance': 1 - abs(within - between) / total if total else 0
    }
```

---

## Quick Reference: Installation Summary

```bash
# Core frameworks
pip install pyphi                    # IIT Phi computation
pip install inferactively-pymdp      # Active Inference

# Entropy metrics
pip install antropy nolds pyinform ordpy

# Dynamical systems
pip install pynamical scipy

# Self-reference/metacognition
pip install torch bayesian-torch

# Game theory
pip install axelrod nashpy

# Brain networks
pip install networkx bctpy python-louvain leidenalg python-igraph
```

## Minimum Data Requirements

| Metric | Minimum Samples | Recommended |
|--------|-----------------|-------------|
| Lempel-Ziv | 100 | 500+ |
| Sample Entropy (m=2) | 100 | 1000+ |
| Permutation Entropy (m=4) | 240 | 1000+ |
| Transfer Entropy | 100 | 500+ |
| Lyapunov Exponent | 1000 | 5000+ |

## Key Repositories

- **PyPhi**: github.com/wmayner/pyphi
- **pymdp**: github.com/infer-actively/pymdp
- **antropy**: github.com/raphaelvallat/antropy
- **nolds**: github.com/CSchoel/nolds
- **Axelrod**: github.com/Axelrod-Python/Axelrod
- **bctpy**: github.com/aestrivex/bctpy
- **pyinform**: github.com/elife-asu/PyInform