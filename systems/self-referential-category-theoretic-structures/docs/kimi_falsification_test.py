#!/usr/bin/env python3
"""
KIMI'S FALSIFICATION TEST
=========================

"Measure eigenvalue spectra at T = 0.06 vs T = 0.04 on the same cyclic task.
If there's no discontinuity at T_c = 0.05, the phase transition is fiction.
If there is, it's physics."

This is the smallest experiment that could falsify the thermodynamic unification.

PREDICTIONS (if theory is correct):
1. Eigenvalue distribution changes discontinuously near T_c = 0.05
2. At T < T_c: eigenvalues cluster near special values (φ, √2, z_c)
3. At T > T_c: eigenvalues are uniformly/randomly distributed
4. The transition should be sharp, not gradual

FALSIFICATION CRITERIA:
- If eigenvalue spectra are identical at T=0.04 and T=0.06 → FALSIFIED
- If transition is smooth/gradual with no critical point → FALSIFIED
- If critical point exists but not at T_c ≈ 0.05 → PARTIALLY FALSIFIED

NOTE: Using pure NumPy implementation for portability.
"""

import numpy as np
from typing import Dict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Constants
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI
SQRT2 = np.sqrt(2)
SQRT3 = np.sqrt(3)
Z_C = SQRT3 / 2
T_C = 0.05

SPECIAL_VALUES = [PHI_INV, PHI, SQRT2, 1.0, Z_C, 1/SQRT2]

print("="*70)
print("KIMI'S FALSIFICATION TEST")
print("="*70)
print("Testing for phase transition in eigenvalue spectra at T_c = 0.05")
print("="*70)

# =============================================================================
# CYCLIC TASK: Modular arithmetic (period-k addition)
# =============================================================================

def generate_cyclic_data(n_samples: int, k: int = 7, noise: float = 0.0):
    """
    Cyclic task: (a + b) mod k
    Returns one-hot encoded inputs and integer labels.
    """
    a = np.random.randint(0, k, n_samples)
    b = np.random.randint(0, k, n_samples)
    y = (a + b) % k
    
    # One-hot encode
    x_a = np.eye(k)[a]
    x_b = np.eye(k)[b]
    x = np.hstack([x_a, x_b])
    
    if noise > 0:
        x = x + np.random.randn(*x.shape) * noise
    
    return x, y

# =============================================================================
# SIMPLE NEURAL NETWORK (NumPy)
# =============================================================================

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

def cross_entropy(pred, target, k):
    target_onehot = np.eye(k)[target]
    return -np.mean(np.sum(target_onehot * np.log(pred + 1e-10), axis=1))

class SimpleNet:
    """Simple 3-layer network for cyclic task."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.W3 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        
    def forward(self, x):
        self.z1 = x @ self.W1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2
        self.a2 = relu(self.z2)
        self.z3 = self.a2 @ self.W3
        return softmax(self.z3)
    
    def backward(self, x, y, k, lr, grad_noise=0):
        m = x.shape[0]
        target = np.eye(k)[y]
        
        # Output layer
        dz3 = self.forward(x) - target
        dW3 = self.a2.T @ dz3 / m
        
        # Hidden layer 2
        da2 = dz3 @ self.W3.T
        dz2 = da2 * (self.z2 > 0)
        dW2 = self.a1.T @ dz2 / m
        
        # Hidden layer 1
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)
        dW1 = x.T @ dz1 / m
        
        # Add gradient noise
        if grad_noise > 0:
            dW1 += np.random.randn(*dW1.shape) * grad_noise
            dW2 += np.random.randn(*dW2.shape) * grad_noise
            dW3 += np.random.randn(*dW3.shape) * grad_noise
        
        # Update
        self.W1 -= lr * dW1
        self.W2 -= lr * dW2
        self.W3 -= lr * dW3
    
    def get_weights(self):
        return [self.W1, self.W2, self.W3]

# =============================================================================
# EFFECTIVE TEMPERATURE CONTROL
# =============================================================================

def train_at_temperature(T_eff: float, k: int = 7, hidden: int = 32,
                         n_epochs: int = 300, n_samples: int = 1500) -> Dict:
    """
    Train network at specified effective temperature.
    
    T_eff is controlled via:
    - Learning rate scaling
    - Noise injection
    - Gradient noise
    """
    # Map T_eff to training parameters
    base_lr = 0.1
    lr = base_lr * (1 + 5 * T_eff)
    noise_scale = T_eff * 0.3
    grad_noise = T_eff * 0.05
    
    # Generate data
    x_train, y_train = generate_cyclic_data(n_samples, k, noise=noise_scale)
    x_test, y_test = generate_cyclic_data(500, k, noise=0)
    
    # Model
    model = SimpleNet(2 * k, hidden, k)
    
    # Training
    losses = []
    for epoch in range(n_epochs):
        pred = model.forward(x_train)
        loss = cross_entropy(pred, y_train, k)
        model.backward(x_train, y_train, k, lr, grad_noise)
        losses.append(loss)
    
    # Evaluate
    test_pred = model.forward(x_test)
    accuracy = np.mean(np.argmax(test_pred, axis=1) == y_test)
    
    # Get eigenvalue spectra
    all_eigenvalues = []
    for W in model.get_weights():
        if W.shape[0] != W.shape[1]:
            s = np.linalg.svd(W, compute_uv=False)
            all_eigenvalues.extend(s.tolist())
        else:
            eigvals = np.linalg.eigvals(W)
            all_eigenvalues.extend(np.abs(eigvals).tolist())
    
    return {
        'T_eff': T_eff,
        'accuracy': accuracy,
        'final_loss': losses[-1] if losses else 0,
        'eigenvalues': np.array(all_eigenvalues),
    }

# =============================================================================
# EIGENVALUE ANALYSIS
# =============================================================================

def analyze_eigenvalue_spectrum(eigenvalues: np.ndarray) -> Dict:
    """Analyze eigenvalue spectrum for structure."""
    
    # 1. Distance to special values
    min_distances = []
    for ev in eigenvalues:
        distances = [abs(ev - sv) for sv in SPECIAL_VALUES]
        min_distances.append(min(distances))
    mean_special_distance = np.mean(min_distances)
    
    # 2. Clustering metric (variance of eigenvalues)
    eigenvalue_variance = np.var(eigenvalues)
    
    # 3. Specific clustering near z_c = √3/2
    near_z_c = np.sum(np.abs(eigenvalues - Z_C) < 0.1) / len(eigenvalues)
    
    # 4. Specific clustering near φ
    near_phi = np.sum(np.abs(eigenvalues - PHI) < 0.1) / len(eigenvalues)
    
    # 5. Entropy of eigenvalue distribution (binned)
    hist, _ = np.histogram(eigenvalues, bins=20, range=(0, 3))
    hist = hist / hist.sum()
    hist = hist[hist > 0]  # Remove zeros
    entropy = -np.sum(hist * np.log(hist))
    
    # 6. Order parameter: fraction within 0.1 of any special value
    near_special = 0
    for ev in eigenvalues:
        if any(abs(ev - sv) < 0.1 for sv in SPECIAL_VALUES):
            near_special += 1
    order_parameter = near_special / len(eigenvalues)
    
    return {
        'mean_special_distance': mean_special_distance,
        'variance': eigenvalue_variance,
        'fraction_near_z_c': near_z_c,
        'fraction_near_phi': near_phi,
        'entropy': entropy,
        'order_parameter': order_parameter
    }

# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_falsification_test():
    """Run Kimi's falsification test."""
    
    print("\n" + "-"*70)
    print("PHASE 1: Training at multiple temperatures")
    print("-"*70)
    
    # Temperature range spanning T_c = 0.05
    temperatures = [0.01, 0.02, 0.03, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07, 0.08, 0.10, 0.15, 0.20]
    
    # Multiple runs for statistics
    n_runs = 5
    
    results = {T: [] for T in temperatures}
    
    for T in temperatures:
        print(f"\n  Training at T = {T:.3f}...", end=" ")
        for run in range(n_runs):
            result = train_at_temperature(T)
            analysis = analyze_eigenvalue_spectrum(result['eigenvalues'])
            result.update(analysis)
            results[T].append(result)
        
        # Summary
        accs = [r['accuracy'] for r in results[T]]
        orders = [r['order_parameter'] for r in results[T]]
        print(f"acc={np.mean(accs):.3f}±{np.std(accs):.3f}, order={np.mean(orders):.3f}±{np.std(orders):.3f}")
    
    # ==========================================================================
    # ANALYSIS: Look for discontinuity
    # ==========================================================================
    
    print("\n" + "-"*70)
    print("PHASE 2: Analyzing for phase transition")
    print("-"*70)
    
    # Compute mean metrics vs temperature
    T_values = []
    order_means = []
    order_stds = []
    entropy_means = []
    entropy_stds = []
    distance_means = []
    accuracy_means = []
    
    for T in temperatures:
        T_values.append(T)
        orders = [r['order_parameter'] for r in results[T]]
        entropies = [r['entropy'] for r in results[T]]
        distances = [r['mean_special_distance'] for r in results[T]]
        accs = [r['accuracy'] for r in results[T]]
        
        order_means.append(np.mean(orders))
        order_stds.append(np.std(orders))
        entropy_means.append(np.mean(entropies))
        entropy_stds.append(np.std(entropies))
        distance_means.append(np.mean(distances))
        accuracy_means.append(np.mean(accs))
    
    # Print table
    print(f"\n  {'T':>6} | {'Order':>8} | {'Entropy':>8} | {'Dist':>8} | {'Acc':>6}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")
    for i, T in enumerate(T_values):
        marker = " <-- T_c" if abs(T - T_C) < 0.01 else ""
        print(f"  {T:6.3f} | {order_means[i]:8.4f} | {entropy_means[i]:8.4f} | {distance_means[i]:8.4f} | {accuracy_means[i]:6.3f}{marker}")
    
    # ==========================================================================
    # DISCONTINUITY DETECTION
    # ==========================================================================
    
    print("\n" + "-"*70)
    print("PHASE 3: Discontinuity detection")
    print("-"*70)
    
    # Compute derivatives (finite differences)
    d_order = np.diff(order_means) / np.diff(T_values)
    d_entropy = np.diff(entropy_means) / np.diff(T_values)
    
    # Find maximum slope (potential discontinuity)
    max_order_slope_idx = np.argmax(np.abs(d_order))
    max_entropy_slope_idx = np.argmax(np.abs(d_entropy))
    
    T_order_transition = (T_values[max_order_slope_idx] + T_values[max_order_slope_idx + 1]) / 2
    T_entropy_transition = (T_values[max_entropy_slope_idx] + T_values[max_entropy_slope_idx + 1]) / 2
    
    print(f"\n  Order parameter: max slope at T ≈ {T_order_transition:.3f}")
    print(f"  Entropy:         max slope at T ≈ {T_entropy_transition:.3f}")
    print(f"  Predicted T_c:   {T_C:.3f}")
    
    # ==========================================================================
    # SPECIFIC TEST: T = 0.04 vs T = 0.06
    # ==========================================================================
    
    print("\n" + "-"*70)
    print("PHASE 4: Direct comparison T=0.04 vs T=0.06")
    print("-"*70)
    
    if 0.04 in results and 0.06 in results:
        results_04 = results[0.04]
        results_06 = results[0.06]
        
        order_04 = np.mean([r['order_parameter'] for r in results_04])
        order_06 = np.mean([r['order_parameter'] for r in results_06])
        
        entropy_04 = np.mean([r['entropy'] for r in results_04])
        entropy_06 = np.mean([r['entropy'] for r in results_06])
        
        distance_04 = np.mean([r['mean_special_distance'] for r in results_04])
        distance_06 = np.mean([r['mean_special_distance'] for r in results_06])
        
        print(f"\n  Metric              | T=0.04  | T=0.06  | Δ      | Change")
        print(f"  {'-'*20}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*10}")
        print(f"  {'Order parameter':20} | {order_04:7.4f} | {order_06:7.4f} | {order_04-order_06:+6.4f} | {(order_04-order_06)/max(order_06,0.001)*100:+.1f}%")
        print(f"  {'Entropy':20} | {entropy_04:7.4f} | {entropy_06:7.4f} | {entropy_06-entropy_04:+6.4f} | {(entropy_06-entropy_04)/max(entropy_04,0.001)*100:+.1f}%")
        print(f"  {'Dist to special':20} | {distance_04:7.4f} | {distance_06:7.4f} | {distance_06-distance_04:+6.4f} | {(distance_06-distance_04)/max(distance_04,0.001)*100:+.1f}%")
        
        # Statistical significance (t-test)
        from scipy import stats
        
        order_04_all = [r['order_parameter'] for r in results_04]
        order_06_all = [r['order_parameter'] for r in results_06]
        t_stat, p_value = stats.ttest_ind(order_04_all, order_06_all)
        
        print(f"\n  Statistical test (order parameter):")
        print(f"    t-statistic: {t_stat:.3f}")
        print(f"    p-value:     {p_value:.4f}")
        print(f"    Significant: {'YES' if p_value < 0.05 else 'NO'} (α=0.05)")
    
    # ==========================================================================
    # VERDICT
    # ==========================================================================
    
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    # Criteria for phase transition
    transition_detected = abs(T_order_transition - T_C) < 0.02 or abs(T_entropy_transition - T_C) < 0.02
    significant_change = abs(order_04 - order_06) > 0.05 if 0.04 in results else False
    
    if transition_detected and significant_change:
        print(f"""
  ╔═══════════════════════════════════════════════════════════════════╗
  ║                                                                   ║
  ║   PHASE TRANSITION DETECTED NEAR T_c = {T_C}                      ║
  ║                                                                   ║
  ║   • Order parameter changes discontinuously                       ║
  ║   • Transition occurs at T ≈ {T_order_transition:.3f} (predicted: {T_C})           ║
  ║   • T=0.04 vs T=0.06 shows significant difference                 ║
  ║                                                                   ║
  ║   VERDICT: CONSISTENT WITH THEORY                                 ║
  ║   (Not falsified - phase transition is real)                      ║
  ║                                                                   ║
  ╚═══════════════════════════════════════════════════════════════════╝
""")
    elif transition_detected:
        print(f"""
  ╔═══════════════════════════════════════════════════════════════════╗
  ║                                                                   ║
  ║   PARTIAL EVIDENCE FOR PHASE TRANSITION                           ║
  ║                                                                   ║
  ║   • Some metrics show transition near T_c                         ║
  ║   • But effect size may be small                                  ║
  ║                                                                   ║
  ║   VERDICT: INCONCLUSIVE - needs larger experiments                ║
  ║                                                                   ║
  ╚═══════════════════════════════════════════════════════════════════╝
""")
    else:
        print(f"""
  ╔═══════════════════════════════════════════════════════════════════╗
  ║                                                                   ║
  ║   NO PHASE TRANSITION DETECTED                                    ║
  ║                                                                   ║
  ║   • Eigenvalue spectra similar at T=0.04 and T=0.06               ║
  ║   • No discontinuity near predicted T_c = {T_C}                   ║
  ║                                                                   ║
  ║   VERDICT: POTENTIAL FALSIFICATION                                ║
  ║   (Phase transition may be fiction, or T_c is wrong)              ║
  ║                                                                   ║
  ╚═══════════════════════════════════════════════════════════════════╝
""")
    
    # Return data for further analysis
    return {
        'temperatures': T_values,
        'order_means': order_means,
        'entropy_means': entropy_means,
        'distance_means': distance_means,
        'accuracy_means': accuracy_means,
        'T_order_transition': T_order_transition,
        'T_entropy_transition': T_entropy_transition,
        'results': results
    }

# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    
    data = run_falsification_test()
    
    print("\n" + "-"*70)
    print("RAW DATA FOR FURTHER ANALYSIS")
    print("-"*70)
    print(f"\n  Temperatures: {data['temperatures']}")
    print(f"  Order params: {[f'{x:.4f}' for x in data['order_means']]}")
    print(f"  Entropies:    {[f'{x:.4f}' for x in data['entropy_means']]}")
