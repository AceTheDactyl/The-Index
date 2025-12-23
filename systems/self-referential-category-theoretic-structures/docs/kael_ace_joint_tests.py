#!/usr/bin/env python3

# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: JUSTIFIED - Documentation file
# Severity: LOW RISK
# Risk Types: ['documentation']
# File: systems/self-referential-category-theoretic-structures/docs/kael_ace_joint_tests.py

"""
================================================================================
KAEL + ACE JOINT TEST SUITE
================================================================================

Tests derived from the convergence of:
  - Kael's neural network spin glass findings
  - Ace's consciousness physics (AT line, frustration geometry)

TESTS:
1. AT Line Test: T_c(h) = T_c(0) × √(1 - h²)
2. Overlap Distribution P(q) at different T
3. Ultrametricity Test
4. 120° Frustration in Cyclic Tasks

Expected runtime: ~3-4 hours total

================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
import time
import warnings

warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants
PHI = (1 + np.sqrt(5)) / 2
SQRT3 = np.sqrt(3)
SQRT3_2 = SQRT3 / 2  # = 0.866025... = z_c
T_C = 0.05

SPECIAL_VALUES = [PHI, 1/PHI, np.sqrt(2), 1/np.sqrt(2), SQRT3_2, 1.0, 0.5]

print(f"Device: {DEVICE}")
print(f"z_c = √3/2 = {SQRT3_2:.6f}")
print(f"T_c = {T_C}")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def make_modular_addition_data(n, mod=7):
    """Standard modular addition task (cyclic)"""
    a = torch.randint(0, mod, (n,))
    b = torch.randint(0, mod, (n,))
    x_a = torch.zeros(n, mod)
    x_b = torch.zeros(n, mod)
    x_a.scatter_(1, a.unsqueeze(1), 1)
    x_b.scatter_(1, b.unsqueeze(1), 1)
    x = torch.cat([x_a, x_b], dim=1)
    y = (a + b) % mod
    return x, y

def make_biased_data(n, mod=7, bias_strength=0.0):
    """
    Modular addition with output bias (effective "field" h)
    bias_strength ∈ [0, 1]: how much to bias toward output 0
    """
    a = torch.randint(0, mod, (n,))
    b = torch.randint(0, mod, (n,))
    x_a = torch.zeros(n, mod)
    x_b = torch.zeros(n, mod)
    x_a.scatter_(1, a.unsqueeze(1), 1)
    x_b.scatter_(1, b.unsqueeze(1), 1)
    x = torch.cat([x_a, x_b], dim=1)
    
    # True labels
    y = (a + b) % mod
    
    # Apply bias: with probability bias_strength, force y = 0
    if bias_strength > 0:
        mask = torch.rand(n) < bias_strength
        y[mask] = 0
    
    return x, y

class MLPNet(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, output_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False)
        )
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        return self.net(x)
    
    def get_weights_flat(self):
        """Get all weights as a flat vector"""
        return torch.cat([p.data.view(-1) for p in self.parameters()]).cpu().numpy()
    
    def get_square_weight(self):
        for m in self.net:
            if isinstance(m, nn.Linear) and m.weight.shape[0] == m.weight.shape[1]:
                return m.weight.data.cpu().numpy()
        return None

def order_param(W, tol=0.1):
    if W is None:
        return 0.0
    eigs = np.abs(np.linalg.eigvals(W))
    return sum(1 for ev in eigs if min(abs(ev - sv) for sv in SPECIAL_VALUES) < tol) / len(eigs)

def train_network(T, h=0.0, epochs=200, hidden_dim=64, seed=None):
    """Train with temperature T and effective field h (bias)"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    if h > 0:
        x_train, y_train = make_biased_data(5000, bias_strength=h)
    else:
        x_train, y_train = make_modular_addition_data(5000)
    
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=128, shuffle=True)
    
    model = MLPNet(hidden_dim=hidden_dim).to(DEVICE)
    lr = 0.01 * (1 + 2 * T)
    opt = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(model(xb), yb)
            loss.backward()
            if T > 0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad += torch.randn_like(p.grad) * (T * 0.05)
            opt.step()
    
    return model

# ==============================================================================
# TEST 1: AT LINE
# ==============================================================================

def test_at_line():
    """
    Test Almeida-Thouless line prediction:
    T_c(h) = T_c(0) × √(1 - h²)
    
    If neural networks are spin glasses, adding a "field" (output bias)
    should shift T_c according to the AT formula.
    """
    print("\n" + "="*70)
    print("TEST 1: ALMEIDA-THOULESS LINE")
    print("="*70)
    print("Prediction: T_c(h) = T_c(0) × √(1 - h²)")
    print("Adding output bias (field h) should shift T_c")
    
    # Test at different field strengths
    field_strengths = [0.0, 0.25, 0.5, 0.75]
    temps = np.linspace(0.02, 0.10, 9)
    n_runs = 20
    epochs = 150
    hidden_dim = 64
    
    results = {}
    start = time.time()
    
    for h in field_strengths:
        print(f"\nField strength h = {h}:")
        chi_values = []
        
        for T in temps:
            orders = []
            for run in range(n_runs):
                model = train_network(T, h=h, epochs=epochs, hidden_dim=hidden_dim, seed=run)
                W = model.get_square_weight()
                orders.append(order_param(W))
            
            chi = np.var(orders)
            chi_values.append(chi)
        
        # Find peak
        max_idx = np.argmax(chi_values)
        T_c_measured = temps[max_idx]
        
        results[h] = {
            'T_c': T_c_measured,
            'chi': chi_values
        }
        
        print(f"  T_c(h={h}) = {T_c_measured:.4f}")
    
    # Compare to AT line prediction
    print("\n" + "-"*50)
    print("AT LINE COMPARISON:")
    print("-"*50)
    
    T_c_0 = results[0.0]['T_c']
    print(f"\n{'h':>8} | {'T_c (meas)':>12} | {'T_c (AT)':>12} | {'Error':>10}")
    print("-"*50)
    
    for h in field_strengths:
        T_c_measured = results[h]['T_c']
        T_c_AT = T_c_0 * np.sqrt(1 - h**2) if h < 1 else 0
        error = abs(T_c_measured - T_c_AT)
        print(f"{h:>8.2f} | {T_c_measured:>12.4f} | {T_c_AT:>12.4f} | {error:>10.4f}")
    
    print(f"\nRuntime: {(time.time() - start)/60:.1f} minutes")
    
    return results

# ==============================================================================
# TEST 2: OVERLAP DISTRIBUTION P(q)
# ==============================================================================

def test_overlap_distribution():
    """
    Test overlap distribution P(q) at different temperatures.
    
    Spin glass prediction:
      T > T_c: P(q) peaked at 0 (replica symmetric)
      T < T_c: P(q) continuous (RSB)
      T ≈ T_c: P(q) bimodal (critical)
    """
    print("\n" + "="*70)
    print("TEST 2: OVERLAP DISTRIBUTION P(q)")
    print("="*70)
    print("Prediction: P(q) changes from peaked (T>T_c) to continuous (T<T_c)")
    
    temps = [0.02, 0.05, 0.08]  # Below, at, above T_c
    n_replicas = 30
    epochs = 200
    hidden_dim = 64
    
    results = {}
    start = time.time()
    
    for T in temps:
        print(f"\nTemperature T = {T}:")
        
        # Train replicas
        weights = []
        for run in range(n_replicas):
            model = train_network(T, epochs=epochs, hidden_dim=hidden_dim, seed=run*1000)
            weights.append(model.get_weights_flat())
        
        # Compute all pairwise overlaps
        overlaps = []
        for i in range(len(weights)):
            for j in range(i+1, len(weights)):
                q = np.dot(weights[i], weights[j]) / (np.linalg.norm(weights[i]) * np.linalg.norm(weights[j]))
                overlaps.append(q)
        
        overlaps = np.array(overlaps)
        
        # Statistics
        results[T] = {
            'overlaps': overlaps,
            'mean': np.mean(overlaps),
            'std': np.std(overlaps),
            'min': np.min(overlaps),
            'max': np.max(overlaps)
        }
        
        print(f"  q: {np.mean(overlaps):.4f} ± {np.std(overlaps):.4f}")
        print(f"  Range: [{np.min(overlaps):.4f}, {np.max(overlaps):.4f}]")
        
        # Simple bimodality test
        # For bimodal distribution, std should be larger relative to range
        bimodality = results[T]['std'] / (results[T]['max'] - results[T]['min'] + 1e-6)
        print(f"  Bimodality index: {bimodality:.4f}")
    
    print(f"\nRuntime: {(time.time() - start)/60:.1f} minutes")
    
    print("\nInterpretation:")
    print("  High bimodality at T_c → RSB transition")
    print("  Low bimodality above T_c → replica symmetric")
    
    return results

# ==============================================================================
# TEST 3: ULTRAMETRICITY
# ==============================================================================

def test_ultrametricity():
    """
    Test ultrametric structure in solution space.
    
    Ultrametric: d(α,γ) ≤ max(d(α,β), d(β,γ)) for all triples
    
    Spin glass prediction: ultrametricity increases below T_c
    """
    print("\n" + "="*70)
    print("TEST 3: ULTRAMETRIC STRUCTURE")
    print("="*70)
    print("Testing: d(α,γ) ≤ max(d(α,β), d(β,γ)) for all triples")
    
    temps = [0.02, 0.05, 0.08]
    n_networks = 20
    epochs = 200
    hidden_dim = 64
    
    results = {}
    start = time.time()
    
    for T in temps:
        print(f"\nTemperature T = {T}:")
        
        # Train networks
        weights = []
        for run in range(n_networks):
            model = train_network(T, epochs=epochs, hidden_dim=hidden_dim, seed=run)
            weights.append(model.get_weights_flat())
        
        # Compute distance matrix
        n = len(weights)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(weights[i] - weights[j])
                dist[i,j] = dist[j,i] = d
        
        # Check ultrametricity for all triples
        ultrametric_count = 0
        total_triples = 0
        
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    d_ij = dist[i,j]
                    d_jk = dist[j,k]
                    d_ik = dist[i,k]
                    
                    # Check strong triangle inequality
                    # In ultrametric: d_ik ≤ max(d_ij, d_jk) (for all permutations)
                    sides = sorted([d_ij, d_jk, d_ik])
                    if sides[2] <= sides[1] * 1.01:  # Allow 1% tolerance
                        ultrametric_count += 1
                    total_triples += 1
        
        ultrametricity = ultrametric_count / total_triples if total_triples > 0 else 0
        
        results[T] = {
            'ultrametricity': ultrametricity,
            'total_triples': total_triples
        }
        
        print(f"  Ultrametric triples: {ultrametric_count}/{total_triples} ({100*ultrametricity:.1f}%)")
    
    print(f"\nRuntime: {(time.time() - start)/60:.1f} minutes")
    
    print("\nInterpretation:")
    print("  High ultrametricity → hierarchical tree structure")
    print("  Should increase below T_c (spin glass phase)")
    
    return results

# ==============================================================================
# TEST 4: 120° FRUSTRATION IN CYCLIC TASKS
# ==============================================================================

def test_120_frustration():
    """
    Test for 120° frustration in cyclic (modular) tasks.
    
    Prediction: Cyclic tasks create triangular frustration,
    leading to 120° angles in weight space.
    """
    print("\n" + "="*70)
    print("TEST 4: 120° FRUSTRATION IN CYCLIC TASKS")
    print("="*70)
    print("Prediction: Cyclic tasks show more 120° angles than non-cyclic")
    
    def angle_distribution(model, n_samples=500):
        """Sample angles between weight row pairs"""
        W = model.get_square_weight()
        if W is None:
            return []
        
        n = W.shape[0]
        angles = []
        for _ in range(n_samples):
            i, j = np.random.choice(n, 2, replace=False)
            v1, v2 = W[i], W[j]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.degrees(np.arccos(cos_angle))
            angles.append(angle)
        return np.array(angles)
    
    n_networks = 20
    epochs = 300
    hidden_dim = 64
    T = 0.05  # At critical temperature
    
    results = {}
    start = time.time()
    
    # Cyclic task (modular addition)
    print("\n1. CYCLIC TASK (mod-7 addition):")
    angles_cyclic = []
    for run in range(n_networks):
        model = train_network(T, h=0, epochs=epochs, hidden_dim=hidden_dim, seed=run)
        angles_cyclic.extend(angle_distribution(model))
    
    angles_cyclic = np.array(angles_cyclic)
    results['cyclic'] = {
        'mean': np.mean(angles_cyclic),
        'near_90': np.mean(np.abs(angles_cyclic - 90) < 15),
        'near_120': np.mean(np.abs(angles_cyclic - 120) < 15)
    }
    print(f"  Mean angle: {np.mean(angles_cyclic):.1f}°")
    print(f"  Near 90°: {100*results['cyclic']['near_90']:.1f}%")
    print(f"  Near 120°: {100*results['cyclic']['near_120']:.1f}%")
    
    # Non-cyclic task (biased - breaks cyclic symmetry)
    print("\n2. NON-CYCLIC TASK (biased outputs):")
    angles_noncyclic = []
    for run in range(n_networks):
        model = train_network(T, h=0.5, epochs=epochs, hidden_dim=hidden_dim, seed=run)
        angles_noncyclic.extend(angle_distribution(model))
    
    angles_noncyclic = np.array(angles_noncyclic)
    results['noncyclic'] = {
        'mean': np.mean(angles_noncyclic),
        'near_90': np.mean(np.abs(angles_noncyclic - 90) < 15),
        'near_120': np.mean(np.abs(angles_noncyclic - 120) < 15)
    }
    print(f"  Mean angle: {np.mean(angles_noncyclic):.1f}°")
    print(f"  Near 90°: {100*results['noncyclic']['near_90']:.1f}%")
    print(f"  Near 120°: {100*results['noncyclic']['near_120']:.1f}%")
    
    print(f"\nRuntime: {(time.time() - start)/60:.1f} minutes")
    
    # Compare
    print("\n" + "-"*50)
    print("COMPARISON:")
    print("-"*50)
    print(f"\n{'Task':>15} | {'Mean Angle':>12} | {'Near 120°':>10}")
    print("-"*45)
    for task in ['cyclic', 'noncyclic']:
        print(f"{task:>15} | {results[task]['mean']:>10.1f}° | {100*results[task]['near_120']:>9.1f}%")
    
    diff = results['cyclic']['near_120'] - results['noncyclic']['near_120']
    print(f"\nDifference: {100*diff:+.1f}% more 120° angles in cyclic task")
    
    if diff > 0.05:
        print("✅ CYCLIC TASKS SHOW MORE FRUSTRATION (as predicted)")
    else:
        print("❌ No significant difference detected")
    
    return results

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "="*70)
    print("KAEL + ACE JOINT TEST SUITE")
    print("="*70)
    print(f"\nDevice: {DEVICE}")
    print("Testing predictions from spin glass + consciousness convergence")
    
    all_results = {}
    
    print("\n" + "#"*70)
    print("# TEST 1: AT LINE")
    print("#"*70)
    all_results['at_line'] = test_at_line()
    
    print("\n" + "#"*70)
    print("# TEST 2: OVERLAP DISTRIBUTION")
    print("#"*70)
    all_results['overlap'] = test_overlap_distribution()
    
    print("\n" + "#"*70)
    print("# TEST 3: ULTRAMETRICITY")
    print("#"*70)
    all_results['ultrametric'] = test_ultrametricity()
    
    print("\n" + "#"*70)
    print("# TEST 4: 120° FRUSTRATION")
    print("#"*70)
    all_results['frustration'] = test_120_frustration()
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print("""
    TEST 1 (AT Line): Does T_c shift with field according to √(1-h²)?
    TEST 2 (P(q)): Does overlap distribution show RSB below T_c?
    TEST 3 (Ultrametric): Is solution space ultrametric?
    TEST 4 (120°): Do cyclic tasks show more frustration angles?
    
    If all pass → Neural networks are spin glasses in full generality!
    """)
    
    return all_results

if __name__ == "__main__":
    results = main()
