#!/usr/bin/env python3
"""
================================================================================
KAEL + ACE COMPLETE TEST SUITE v2
================================================================================

Tests derived from spin glass physics (Ace) applied to neural networks (Kael).

NEW IN v2:
- Finite-size scaling test (γ/ν = 1/2 prediction)
- Better AT line implementation
- Edwards-Anderson order parameter
- Improved overlap distribution analysis

Expected runtime: ~4-6 hours on CPU, ~1-2 hours on GPU

================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from scipy import stats
from scipy.optimize import curve_fit
import time
import warnings

warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants from synthesis
PHI = (1 + np.sqrt(5)) / 2
SQRT3 = np.sqrt(3)
SQRT3_2 = SQRT3 / 2  # = 0.866025 = z_c
T_C = 0.05  # Our measured critical temperature

# SK model exponents (mean-field)
GAMMA = 1.0  # χ ~ |T - T_c|^{-γ}
NU = 2.0     # ξ ~ |T - T_c|^{-ν}
BETA = 0.5   # q_EA ~ (T_c - T)^β

SPECIAL_VALUES = [PHI, 1/PHI, np.sqrt(2), 1/np.sqrt(2), SQRT3_2, 1.0, 0.5]

print(f"Device: {DEVICE}")
print(f"Spin Glass Exponents: γ={GAMMA}, ν={NU}, β={BETA}")
print(f"Predicted: χ_max ~ n^{GAMMA/NU} = n^0.5 = √n")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def make_modular_addition_data(n, mod=7, bias=0.0):
    """Modular addition with optional output bias (effective field h)"""
    a = torch.randint(0, mod, (n,))
    b = torch.randint(0, mod, (n,))
    x_a = torch.zeros(n, mod)
    x_b = torch.zeros(n, mod)
    x_a.scatter_(1, a.unsqueeze(1), 1)
    x_b.scatter_(1, b.unsqueeze(1), 1)
    x = torch.cat([x_a, x_b], dim=1)
    y = (a + b) % mod
    
    # Apply bias toward class 0
    if bias > 0:
        mask = torch.rand(n) < bias
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
        return torch.cat([p.data.view(-1) for p in self.parameters()]).cpu().numpy()
    
    def get_square_weight(self):
        for m in self.net:
            if isinstance(m, nn.Linear) and m.weight.shape[0] == m.weight.shape[1]:
                return m.weight.data.cpu().numpy()
        return None

def order_param(W, tol=0.1):
    """Fraction of eigenvalues near special values (Edwards-Anderson analog)"""
    if W is None:
        return 0.0
    eigs = np.abs(np.linalg.eigvals(W))
    return sum(1 for ev in eigs if min(abs(ev - sv) for sv in SPECIAL_VALUES) < tol) / len(eigs)

def train_network(T, bias=0.0, epochs=200, hidden_dim=64, seed=None):
    """Train with effective temperature T and field h (bias)"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    x_train, y_train = make_modular_addition_data(5000, bias=bias)
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
# TEST 1: FINITE-SIZE SCALING (NEW - From Ace's SK theory)
# ==============================================================================

def test_finite_size_scaling():
    """
    KEY TEST from spin glass theory:
    
    χ_max ~ n^{γ/ν} = n^{1/2} = √n
    
    If true, neural networks are in SK universality class!
    """
    print("\n" + "="*70)
    print("TEST 1: FINITE-SIZE SCALING")
    print("="*70)
    print(f"Prediction: χ_max ~ n^{{γ/ν}} = n^{GAMMA/NU} = √n")
    print("This is THE critical test for SK universality class!")
    
    hidden_dims = [32, 64, 128]  # Could add 256 if time permits
    temps = np.linspace(0.02, 0.08, 13)
    n_runs = 25
    epochs = 180
    
    results = {}
    start = time.time()
    
    for hidden_dim in hidden_dims:
        print(f"\n--- Hidden dimension n = {hidden_dim} ---")
        chi_values = []
        o_means = []
        
        for T in temps:
            orders = []
            for run in range(n_runs):
                model = train_network(T, epochs=epochs, hidden_dim=hidden_dim, 
                                     seed=hash((hidden_dim, T, run)) % (2**32))
                W = model.get_square_weight()
                orders.append(order_param(W))
            
            chi = np.var(orders)
            chi_values.append(chi)
            o_means.append(np.mean(orders))
        
        # Find peak
        max_idx = np.argmax(chi_values)
        T_c_measured = temps[max_idx]
        chi_max = chi_values[max_idx]
        
        results[hidden_dim] = {
            'chi_max': chi_max,
            'T_c': T_c_measured,
            'chi_curve': chi_values,
            'temps': temps
        }
        
        print(f"  χ_max = {chi_max:.6f}")
        print(f"  T_c = {T_c_measured:.4f}")
    
    # Test scaling: χ_max ~ √n
    print("\n" + "-"*50)
    print("FINITE-SIZE SCALING ANALYSIS")
    print("-"*50)
    
    ns = list(results.keys())
    chi_maxs = [results[n]['chi_max'] for n in ns]
    
    # Log-log fit
    log_n = np.log(ns)
    log_chi = np.log(chi_maxs)
    
    slope, intercept, r, p, se = stats.linregress(log_n, log_chi)
    
    print(f"\nLog-log fit: log(χ_max) = {slope:.4f} × log(n) + {intercept:.4f}")
    print(f"  Measured exponent γ/ν = {slope:.4f}")
    print(f"  Predicted (SK): γ/ν = {GAMMA/NU:.4f}")
    print(f"  R² = {r**2:.4f}")
    print(f"  Error: {abs(slope - GAMMA/NU):.4f} ({100*abs(slope - GAMMA/NU)/(GAMMA/NU):.1f}%)")
    
    print(f"\nχ_max values:")
    print(f"  {'n':>6} | {'χ_max':>12} | {'√n':>8} | {'χ_max/√n':>12}")
    print("-"*50)
    for n in ns:
        chi = results[n]['chi_max']
        sqrt_n = np.sqrt(n)
        ratio = chi / sqrt_n
        print(f"  {n:>6} | {chi:>12.6f} | {sqrt_n:>8.2f} | {ratio:>12.6f}")
    
    # Check if ratio is constant (meaning χ ~ √n)
    ratios = [results[n]['chi_max'] / np.sqrt(n) for n in ns]
    ratio_std = np.std(ratios) / np.mean(ratios)  # Coefficient of variation
    
    print(f"\nχ_max/√n ratios: {[f'{r:.6f}' for r in ratios]}")
    print(f"Coefficient of variation: {100*ratio_std:.1f}%")
    
    if abs(slope - GAMMA/NU) < 0.2:
        print("\n✅ FINITE-SIZE SCALING CONSISTENT WITH SK MODEL (γ/ν ≈ 0.5)")
    else:
        print(f"\n❌ EXPONENT {slope:.2f} DIFFERS FROM SK PREDICTION 0.5")
    
    print(f"\nRuntime: {(time.time() - start)/60:.1f} minutes")
    
    return results

# ==============================================================================
# TEST 2: AT LINE (From Ace's theory)
# ==============================================================================

def test_at_line():
    """
    Test Almeida-Thouless line:
    T_c(h) = T_c(0) × √(1 - h²)
    
    where h is output bias (effective magnetic field)
    """
    print("\n" + "="*70)
    print("TEST 2: ALMEIDA-THOULESS LINE")
    print("="*70)
    print("Prediction: T_c(h) = T_c(0) × √(1 - h²)")
    
    biases = [0.0, 0.3, 0.5, 0.7]  # Effective field h
    temps = np.linspace(0.02, 0.10, 17)
    n_runs = 20
    epochs = 150
    hidden_dim = 64
    
    results = {}
    start = time.time()
    
    for h in biases:
        print(f"\nBias h = {h}:")
        chi_values = []
        
        for T in temps:
            orders = []
            for run in range(n_runs):
                model = train_network(T, bias=h, epochs=epochs, 
                                     hidden_dim=hidden_dim, seed=run)
                W = model.get_square_weight()
                orders.append(order_param(W))
            
            chi_values.append(np.var(orders))
        
        max_idx = np.argmax(chi_values)
        T_c_measured = temps[max_idx]
        
        results[h] = {
            'T_c': T_c_measured,
            'T_c_AT': T_C * np.sqrt(1 - h**2) if h < 1 else 0,
            'chi_curve': chi_values
        }
        
        print(f"  T_c(measured) = {T_c_measured:.4f}")
        print(f"  T_c(AT line)  = {results[h]['T_c_AT']:.4f}")
    
    # Compare to AT line
    print("\n" + "-"*50)
    print("AT LINE COMPARISON")
    print("-"*50)
    
    print(f"\n{'h':>8} | {'T_c (meas)':>12} | {'T_c (AT)':>12} | {'Error':>10}")
    print("-"*50)
    
    errors = []
    for h in biases:
        T_meas = results[h]['T_c']
        T_AT = results[h]['T_c_AT']
        error = abs(T_meas - T_AT) / T_AT if T_AT > 0 else 0
        errors.append(error)
        print(f"{h:>8.2f} | {T_meas:>12.4f} | {T_AT:>12.4f} | {100*error:>9.1f}%")
    
    mean_error = np.mean(errors)
    print(f"\nMean relative error: {100*mean_error:.1f}%")
    
    if mean_error < 0.3:
        print("\n✅ AT LINE BEHAVIOR CONFIRMED")
    else:
        print("\n❌ DEVIATES FROM AT LINE PREDICTION")
    
    print(f"\nRuntime: {(time.time() - start)/60:.1f} minutes")
    
    return results

# ==============================================================================
# TEST 3: OVERLAP DISTRIBUTION P(q)
# ==============================================================================

def test_overlap_distribution():
    """
    Test P(q) distribution at different temperatures.
    
    T > T_c: P(q) peaked at ~0 (replica symmetric)
    T < T_c: P(q) broad/continuous (RSB)
    """
    print("\n" + "="*70)
    print("TEST 3: OVERLAP DISTRIBUTION P(q)")
    print("="*70)
    
    temps = [0.02, 0.05, 0.08]
    n_replicas = 30
    epochs = 200
    hidden_dim = 64
    
    results = {}
    start = time.time()
    
    for T in temps:
        print(f"\nTemperature T = {T}:")
        
        weights = []
        for run in range(n_replicas):
            model = train_network(T, epochs=epochs, hidden_dim=hidden_dim, seed=run*1000)
            weights.append(model.get_weights_flat())
        
        # Compute overlaps
        overlaps = []
        for i in range(len(weights)):
            for j in range(i+1, len(weights)):
                q = np.dot(weights[i], weights[j]) / (np.linalg.norm(weights[i]) * np.linalg.norm(weights[j]))
                overlaps.append(q)
        
        overlaps = np.array(overlaps)
        
        results[T] = {
            'overlaps': overlaps,
            'mean': np.mean(overlaps),
            'std': np.std(overlaps),
            'q_EA': np.percentile(overlaps, 95)  # Edwards-Anderson proxy
        }
        
        print(f"  <q> = {np.mean(overlaps):.4f} ± {np.std(overlaps):.4f}")
        print(f"  q_EA (95th percentile) = {results[T]['q_EA']:.4f}")
        
        # Bimodality test
        # For bimodal, expect high kurtosis (>3) or clear two peaks
        kurtosis = stats.kurtosis(overlaps)
        print(f"  Kurtosis: {kurtosis:.2f} (bimodal if >> 0)")
    
    print(f"\nRuntime: {(time.time() - start)/60:.1f} minutes")
    
    # Check for RSB signature
    print("\nRSB Signature Check:")
    print("  P(q) should be broader below T_c")
    
    if results[0.02]['std'] > results[0.08]['std'] * 1.2:
        print("  ✅ Broader P(q) below T_c (RSB signature)")
    else:
        print("  ❌ No clear broadening below T_c")
    
    return results

# ==============================================================================
# TEST 4: ULTRAMETRICITY
# ==============================================================================

def test_ultrametricity():
    """
    Test ultrametric structure: d(α,γ) ≤ max(d(α,β), d(β,γ))
    """
    print("\n" + "="*70)
    print("TEST 4: ULTRAMETRIC STRUCTURE")
    print("="*70)
    
    temps = [0.02, 0.05, 0.08]
    n_networks = 20
    epochs = 200
    hidden_dim = 64
    
    results = {}
    start = time.time()
    
    for T in temps:
        print(f"\nTemperature T = {T}:")
        
        weights = []
        for run in range(n_networks):
            model = train_network(T, epochs=epochs, hidden_dim=hidden_dim, seed=run)
            weights.append(model.get_weights_flat())
        
        # Distance matrix
        n = len(weights)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(weights[i] - weights[j])
                dist[i,j] = dist[j,i] = d
        
        # Check ultrametricity
        ultra_count = 0
        total = 0
        
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    sides = sorted([dist[i,j], dist[j,k], dist[i,k]])
                    # Ultrametric: two longest sides are equal (within tolerance)
                    if sides[2] <= sides[1] * 1.05:
                        ultra_count += 1
                    total += 1
        
        ultrametricity = ultra_count / total if total > 0 else 0
        
        results[T] = {'ultrametricity': ultrametricity}
        print(f"  Ultrametric triangles: {100*ultrametricity:.1f}%")
    
    print(f"\nRuntime: {(time.time() - start)/60:.1f} minutes")
    
    if results[0.02]['ultrametricity'] > results[0.08]['ultrametricity']:
        print("\n✅ Higher ultrametricity below T_c (as predicted)")
    else:
        print("\n❌ No ultrametric signature found")
    
    return results

# ==============================================================================
# TEST 5: CRITICAL EXPONENT β
# ==============================================================================

def test_beta_exponent():
    """
    Test q_EA ~ (T_c - T)^β with β = 1/2
    
    Use order parameter O as proxy for q_EA
    """
    print("\n" + "="*70)
    print("TEST 5: CRITICAL EXPONENT β")
    print("="*70)
    print("Prediction: O - O_c ~ (T_c - T)^{1/2} below T_c")
    
    temps = np.linspace(0.01, 0.045, 8)  # Below T_c
    n_runs = 30
    epochs = 200
    hidden_dim = 64
    
    results = {'T': [], 'O': [], 'O_std': []}
    start = time.time()
    
    for T in temps:
        orders = []
        for run in range(n_runs):
            model = train_network(T, epochs=epochs, hidden_dim=hidden_dim, seed=run)
            W = model.get_square_weight()
            orders.append(order_param(W))
        
        results['T'].append(T)
        results['O'].append(np.mean(orders))
        results['O_std'].append(np.std(orders))
        print(f"  T = {T:.3f}: O = {np.mean(orders):.4f} ± {np.std(orders):.4f}")
    
    # Fit O = A + B*(T_c - T)^β
    T_c_fit = T_C
    
    def power_law(T, A, B, beta):
        return A + B * (T_c_fit - T)**beta
    
    try:
        popt, pcov = curve_fit(power_law, results['T'], results['O'], 
                               p0=[0.5, 0.5, 0.5], bounds=([0, 0, 0.1], [1, 2, 1]))
        A, B, beta_fit = popt
        
        print(f"\nFit: O = {A:.3f} + {B:.3f} × (T_c - T)^{beta_fit:.3f}")
        print(f"  Measured β = {beta_fit:.3f}")
        print(f"  Predicted β = {BETA:.3f}")
        print(f"  Error: {abs(beta_fit - BETA):.3f}")
        
        if abs(beta_fit - BETA) < 0.2:
            print("\n✅ EXPONENT β CONSISTENT WITH SK MODEL")
        else:
            print(f"\n⚠️ β = {beta_fit:.2f} differs from SK prediction 0.5")
    except:
        print("\nCould not fit power law (may need more data)")
    
    print(f"\nRuntime: {(time.time() - start)/60:.1f} minutes")
    
    return results

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "="*70)
    print("KAEL + ACE COMPLETE TEST SUITE v2")
    print("="*70)
    print(f"Device: {DEVICE}")
    print("\nTesting spin glass predictions from Ace's framework:")
    print("  - Finite-size scaling (γ/ν = 0.5)")
    print("  - AT line: T_c(h) = T_c(0)√(1-h²)")
    print("  - P(q) distribution changes at T_c")
    print("  - Ultrametric structure")
    print("  - Critical exponent β = 0.5")
    
    all_results = {}
    
    # Run tests
    print("\n" + "#"*70)
    print("# TEST 1: FINITE-SIZE SCALING (Critical for SK universality)")
    print("#"*70)
    all_results['fss'] = test_finite_size_scaling()
    
    print("\n" + "#"*70)
    print("# TEST 2: AT LINE")
    print("#"*70)
    all_results['at_line'] = test_at_line()
    
    print("\n" + "#"*70)
    print("# TEST 3: OVERLAP DISTRIBUTION")
    print("#"*70)
    all_results['overlap'] = test_overlap_distribution()
    
    print("\n" + "#"*70)
    print("# TEST 4: ULTRAMETRICITY")
    print("#"*70)
    all_results['ultra'] = test_ultrametricity()
    
    print("\n" + "#"*70)
    print("# TEST 5: CRITICAL EXPONENT β")
    print("#"*70)
    all_results['beta'] = test_beta_exponent()
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY: SK UNIVERSALITY CLASS TEST")
    print("="*70)
    
    print("""
    If neural networks are in the SK universality class:
    
    1. χ_max ~ √n (finite-size scaling)
    2. T_c(h) follows AT line
    3. P(q) broadens below T_c
    4. Ultrametricity increases below T_c  
    5. β ≈ 0.5
    
    Each confirmed prediction strengthens the spin glass interpretation.
    """)
    
    return all_results

if __name__ == "__main__":
    results = main()
