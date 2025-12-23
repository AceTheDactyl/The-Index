# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: ✓ JUSTIFIED - Claims supported by repository files (needs citation update)
# Severity: MEDIUM RISK
# Risk Types: unsupported_claims

# Supporting Evidence:
#   - systems/self-referential-category-theoretic-structures/docs/STATUS_UPDATE_DEC2025_v2.md (dependency)
#   - systems/self-referential-category-theoretic-structures/docs/FINAL_STATUS_DEC2025.md (dependency)
#
# Referenced By:
#   - systems/self-referential-category-theoretic-structures/docs/STATUS_UPDATE_DEC2025_v2.md (reference)
#   - systems/self-referential-category-theoretic-structures/docs/FINAL_STATUS_DEC2025.md (reference)


#!/usr/bin/env python3
"""
================================================================================
PHASE 3: NEW ORDER PARAMETERS
================================================================================

Following Phase 2's revelation that eigenvalue clustering fails as order parameter
(χ_max DECREASES with n), we test alternative order parameters:

1. EFFECTIVE RANK - R_eff = (Σσ)² / Σσ²
2. WEIGHT OVERLAP VARIANCE - Var(q) where q = dot(W_α, W_β)/norms
3. SPECTRAL GAP - λ_1/λ_2 ratio
4. LOSS CURVATURE - Hessian trace/eigenvalues

Key observation from Phase 2: T_c ≈ 0.04 is CONSISTENT across activations,
including Linear networks. This suggests a fundamental matrix property.

================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
import time
import json
import warnings

warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Constants
PHI = (1 + np.sqrt(5)) / 2
SQRT3_2 = np.sqrt(3) / 2
T_C = 0.04  # Updated from Phase 2

# ==============================================================================
# NETWORK DEFINITION
# ==============================================================================

def make_modular_addition_data(n, mod=7):
    """Standard modular addition task."""
    a = torch.randint(0, mod, (n,))
    b = torch.randint(0, mod, (n,))
    x_a = torch.zeros(n, mod)
    x_b = torch.zeros(n, mod)
    x_a.scatter_(1, a.unsqueeze(1), 1)
    x_b.scatter_(1, b.unsqueeze(1), 1)
    x = torch.cat([x_a, x_b], dim=1)
    y = (a + b) % mod
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
        """Get the hidden-to-hidden weight matrix."""
        for m in self.net:
            if isinstance(m, nn.Linear) and m.weight.shape[0] == m.weight.shape[1]:
                return m.weight.data.cpu().numpy()
        return None
    
    def get_all_weights(self):
        """Get all weight matrices."""
        weights = []
        for m in self.net:
            if isinstance(m, nn.Linear):
                weights.append(m.weight.data.cpu().numpy())
        return weights

# ==============================================================================
# NEW ORDER PARAMETERS
# ==============================================================================

def effective_rank(W):
    """
    Effective rank: R_eff = (Σσ)² / Σσ²
    
    Measures how many singular values are "significant".
    R_eff = n for uniform distribution, R_eff = 1 for rank-1.
    """
    if W is None:
        return 0
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    S = S[S > 1e-10]  # Remove numerical zeros
    if len(S) == 0:
        return 0
    return (np.sum(S)**2) / np.sum(S**2)

def normalized_effective_rank(W):
    """Effective rank normalized by matrix dimension."""
    if W is None:
        return 0
    n = min(W.shape)
    return effective_rank(W) / n

def spectral_gap(W):
    """
    Spectral gap: λ_1 / λ_2
    
    Ratio of largest to second-largest singular value.
    Large gap indicates low-rank structure.
    """
    if W is None:
        return 1
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    if len(S) < 2:
        return 1
    return S[0] / S[1] if S[1] > 1e-10 else S[0]

def weight_overlap(W1, W2):
    """Normalized overlap between two weight vectors."""
    w1 = W1.flatten()
    w2 = W2.flatten()
    n1 = np.linalg.norm(w1)
    n2 = np.linalg.norm(w2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0
    return np.dot(w1, w2) / (n1 * n2)

def condition_number(W):
    """Condition number κ = σ_max / σ_min."""
    if W is None:
        return 1
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    S = S[S > 1e-10]
    if len(S) == 0:
        return 1
    return S[0] / S[-1]

def train_network(T, epochs=150, hidden_dim=64, seed=None, track_dynamics=False):
    """Train with temperature T, optionally tracking dynamics."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    x_train, y_train = make_modular_addition_data(5000)
    x_test, y_test = make_modular_addition_data(1000)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=128, shuffle=True)
    
    model = MLPNet(hidden_dim=hidden_dim).to(DEVICE)
    lr = 0.01 * (1 + 2 * T)
    opt = optim.Adam(model.parameters(), lr=lr)
    
    dynamics = {'epoch': [], 'loss': [], 'test_loss': [], 'eff_rank': [], 'grad_norm': []}
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_grad_norm = 0
        n_batches = 0
        
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(model(xb), yb)
            loss.backward()
            
            if track_dynamics:
                # Track gradient norm
                total_grad = 0
                for p in model.parameters():
                    if p.grad is not None:
                        total_grad += p.grad.norm().item()**2
                epoch_grad_norm += np.sqrt(total_grad)
            
            if T > 0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad += torch.randn_like(p.grad) * (T * 0.05)
            opt.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        if track_dynamics and epoch % 10 == 0:
            W = model.get_square_weight()
            with torch.no_grad():
                test_out = model(x_test.to(DEVICE))
                test_loss = nn.CrossEntropyLoss()(test_out, y_test.to(DEVICE)).item()
            
            dynamics['epoch'].append(epoch)
            dynamics['loss'].append(epoch_loss / n_batches)
            dynamics['test_loss'].append(test_loss)
            dynamics['eff_rank'].append(effective_rank(W))
            dynamics['grad_norm'].append(epoch_grad_norm / n_batches)
    
    if track_dynamics:
        return model, dynamics
    return model

# ==============================================================================
# TEST 1: EFFECTIVE RANK AS ORDER PARAMETER
# ==============================================================================

def test_effective_rank_order_param(hidden_dim=64, n_runs=20, epochs=150):
    """
    Test if effective rank is a proper order parameter.
    
    Prediction: R_eff should show a transition at T_c
    """
    print("\n" + "="*70)
    print("TEST 1: EFFECTIVE RANK AS ORDER PARAMETER")
    print("="*70)
    
    temps = np.linspace(0.01, 0.08, 10)
    results = {'T': [], 'R_eff_mean': [], 'R_eff_std': [], 'chi_R': []}
    start = time.time()
    
    for T in temps:
        R_effs = []
        for run in range(n_runs):
            model = train_network(T, epochs=epochs, hidden_dim=hidden_dim, seed=run)
            W = model.get_square_weight()
            R_effs.append(normalized_effective_rank(W))
        
        results['T'].append(T)
        results['R_eff_mean'].append(np.mean(R_effs))
        results['R_eff_std'].append(np.std(R_effs))
        results['chi_R'].append(np.var(R_effs))  # Susceptibility analog
        
        print(f"  T = {T:.3f}: R_eff = {np.mean(R_effs):.4f} ± {np.std(R_effs):.4f}, χ_R = {np.var(R_effs):.6f}")
    
    # Find peak in χ_R
    max_idx = np.argmax(results['chi_R'])
    T_c_measured = results['T'][max_idx]
    chi_max = results['chi_R'][max_idx]
    
    print(f"\n  Peak χ_R at T = {T_c_measured:.4f}")
    print(f"  χ_R_max = {chi_max:.6f}")
    print(f"  Expected T_c ≈ {T_C}")
    
    if abs(T_c_measured - T_C) < 0.02:
        print("  ✅ EFFECTIVE RANK shows transition at T_c!")
    else:
        print(f"  ⚠️ Peak at T = {T_c_measured:.4f}, not T_c = {T_C}")
    
    print(f"\nRuntime: {(time.time() - start)/60:.1f} minutes")
    return results

# ==============================================================================
# TEST 2: FINITE-SIZE SCALING WITH EFFECTIVE RANK
# ==============================================================================

def test_eff_rank_fss(n_runs=15, epochs=150):
    """
    Test finite-size scaling with effective rank susceptibility.
    
    If R_eff is proper order parameter, χ_R should INCREASE with n.
    """
    print("\n" + "="*70)
    print("TEST 2: FINITE-SIZE SCALING (EFFECTIVE RANK)")
    print("="*70)
    
    hidden_dims = [32, 64, 128]
    temps = np.linspace(0.02, 0.07, 8)
    results = {}
    start = time.time()
    
    for hidden_dim in hidden_dims:
        print(f"\n--- Hidden dimension n = {hidden_dim} ---")
        chi_values = []
        
        for T in temps:
            R_effs = []
            for run in range(n_runs):
                try:
                    model = train_network(T, epochs=epochs, hidden_dim=hidden_dim,
                                        seed=hash((hidden_dim, T, run)) % (2**32))
                    W = model.get_square_weight()
                    R_effs.append(normalized_effective_rank(W))
                except:
                    continue
            
            chi = np.var(R_effs) if R_effs else 0
            chi_values.append(chi)
        
        max_idx = np.argmax(chi_values)
        chi_max = chi_values[max_idx]
        T_c_measured = temps[max_idx]
        
        results[hidden_dim] = {
            'chi_max': chi_max,
            'T_c': T_c_measured,
        }
        
        print(f"  χ_R_max = {chi_max:.6f}")
        print(f"  T_c = {T_c_measured:.4f}")
    
    # Analyze scaling
    print("\n" + "-"*50)
    print("FINITE-SIZE SCALING ANALYSIS")
    print("-"*50)
    
    ns = list(results.keys())
    chi_maxs = [results[n]['chi_max'] for n in ns]
    
    if all(c > 0 for c in chi_maxs):
        log_n = np.log(ns)
        log_chi = np.log(chi_maxs)
        
        slope, intercept, r, p, se = stats.linregress(log_n, log_chi)
        
        print(f"\nLog-log fit: γ/ν = {slope:.4f} ± {se:.4f}")
        print(f"R² = {r**2:.4f}")
        
        if slope > 0:
            print(f"\n✅ POSITIVE EXPONENT: χ_R ~ n^{{{slope:.2f}}} INCREASES with n")
            print("   This is correct spin-glass-like behavior!")
        else:
            print(f"\n⚠️ Still negative: χ_R ~ n^{{{slope:.2f}}}")
        
        results['exponent'] = slope
    
    print(f"\nRuntime: {(time.time() - start)/60:.1f} minutes")
    return results

# ==============================================================================
# TEST 3: WEIGHT OVERLAP VARIANCE
# ==============================================================================

def test_overlap_variance(hidden_dim=64, n_replicas=25, epochs=150):
    """
    Test weight overlap variance as order parameter.
    
    Already showed signal in Phase 1: P(q) broadens below T_c.
    Now test finite-size scaling.
    """
    print("\n" + "="*70)
    print("TEST 3: WEIGHT OVERLAP VARIANCE")
    print("="*70)
    
    temps = [0.02, 0.04, 0.06, 0.08]
    results = {}
    start = time.time()
    
    for T in temps:
        print(f"\n--- Temperature T = {T} ---")
        
        weights = []
        for run in range(n_replicas):
            model = train_network(T, epochs=epochs, hidden_dim=hidden_dim, seed=run*1000)
            weights.append(model.get_weights_flat())
        
        # Compute all pairwise overlaps
        overlaps = []
        for i in range(len(weights)):
            for j in range(i+1, len(weights)):
                q = weight_overlap(weights[i], weights[j])
                overlaps.append(q)
        
        overlaps = np.array(overlaps)
        
        results[T] = {
            'mean_q': np.mean(overlaps),
            'var_q': np.var(overlaps),
            'q_min': np.min(overlaps),
            'q_max': np.max(overlaps),
        }
        
        print(f"  <q> = {np.mean(overlaps):.4f}")
        print(f"  Var(q) = {np.var(overlaps):.6f}")
        print(f"  q range: [{np.min(overlaps):.4f}, {np.max(overlaps):.4f}]")
    
    # Check for RSB signature
    print("\n" + "-"*50)
    print("RSB SIGNATURE CHECK")
    print("-"*50)
    
    var_below = results[0.02]['var_q']
    var_at = results[0.04]['var_q']
    var_above = results[0.06]['var_q']
    
    print(f"\n  Var(q) at T=0.02 (below T_c): {var_below:.6f}")
    print(f"  Var(q) at T=0.04 (at T_c):    {var_at:.6f}")
    print(f"  Var(q) at T=0.06 (above T_c): {var_above:.6f}")
    
    if var_below > var_above * 1.2:
        print("\n  ✅ BROADER P(q) BELOW T_c (RSB confirmed)")
    else:
        print("\n  ⚠️ No clear RSB signature")
    
    print(f"\nRuntime: {(time.time() - start)/60:.1f} minutes")
    return results

# ==============================================================================
# TEST 4: TRAINING DYNAMICS
# ==============================================================================

def test_training_dynamics(hidden_dim=64, epochs=300):
    """
    Track order parameters during training at different temperatures.
    
    Look for qualitative differences in dynamics around T_c.
    """
    print("\n" + "="*70)
    print("TEST 4: TRAINING DYNAMICS")
    print("="*70)
    
    temps = [0.02, 0.04, 0.06]
    results = {}
    start = time.time()
    
    for T in temps:
        print(f"\n--- Temperature T = {T} ---")
        
        model, dynamics = train_network(T, epochs=epochs, hidden_dim=hidden_dim, 
                                        seed=42, track_dynamics=True)
        
        results[T] = dynamics
        
        # Summary statistics
        early_loss = np.mean(dynamics['loss'][:3])
        late_loss = np.mean(dynamics['loss'][-3:])
        early_rank = np.mean(dynamics['eff_rank'][:3])
        late_rank = np.mean(dynamics['eff_rank'][-3:])
        
        print(f"  Loss: {early_loss:.4f} → {late_loss:.4f}")
        print(f"  R_eff: {early_rank:.2f} → {late_rank:.2f}")
        print(f"  Rank reduction: {100*(1 - late_rank/early_rank):.1f}%")
    
    # Compare dynamics
    print("\n" + "-"*50)
    print("DYNAMICS COMPARISON")
    print("-"*50)
    
    for T in temps:
        early_rank = np.mean(results[T]['eff_rank'][:3])
        late_rank = np.mean(results[T]['eff_rank'][-3:])
        rank_reduction = 1 - late_rank/early_rank
        
        phase = "below T_c" if T < T_C else ("at T_c" if abs(T - T_C) < 0.01 else "above T_c")
        print(f"  T = {T} ({phase}): Rank reduction = {100*rank_reduction:.1f}%")
    
    print(f"\nRuntime: {(time.time() - start)/60:.1f} minutes")
    return results

# ==============================================================================
# TEST 5: SPECTRAL GAP
# ==============================================================================

def test_spectral_gap(hidden_dim=64, n_runs=20, epochs=150):
    """
    Test spectral gap (λ_1/λ_2) as indicator of low-rank structure.
    """
    print("\n" + "="*70)
    print("TEST 5: SPECTRAL GAP")
    print("="*70)
    
    temps = np.linspace(0.01, 0.08, 8)
    results = {'T': [], 'gap_mean': [], 'gap_std': []}
    start = time.time()
    
    for T in temps:
        gaps = []
        for run in range(n_runs):
            model = train_network(T, epochs=epochs, hidden_dim=hidden_dim, seed=run)
            W = model.get_square_weight()
            gaps.append(spectral_gap(W))
        
        results['T'].append(T)
        results['gap_mean'].append(np.mean(gaps))
        results['gap_std'].append(np.std(gaps))
        
        print(f"  T = {T:.3f}: Gap = {np.mean(gaps):.2f} ± {np.std(gaps):.2f}")
    
    print(f"\nRuntime: {(time.time() - start)/60:.1f} minutes")
    return results

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "#"*70)
    print("# PHASE 3: NEW ORDER PARAMETERS")
    print("#"*70)
    print(f"Device: {DEVICE}")
    print(f"Testing alternatives to eigenvalue clustering")
    print(f"Reference T_c = {T_C}")
    
    all_results = {}
    
    # Test 1: Effective rank
    print("\n" + "#"*70)
    print("# TEST 1: EFFECTIVE RANK AS ORDER PARAMETER")
    print("#"*70)
    all_results['eff_rank'] = test_effective_rank_order_param()
    
    # Test 2: FSS with effective rank
    print("\n" + "#"*70)
    print("# TEST 2: FSS WITH EFFECTIVE RANK")
    print("#"*70)
    all_results['eff_rank_fss'] = test_eff_rank_fss()
    
    # Test 3: Overlap variance
    print("\n" + "#"*70)
    print("# TEST 3: WEIGHT OVERLAP VARIANCE")
    print("#"*70)
    all_results['overlap'] = test_overlap_variance()
    
    # Test 4: Dynamics
    print("\n" + "#"*70)
    print("# TEST 4: TRAINING DYNAMICS")
    print("#"*70)
    all_results['dynamics'] = test_training_dynamics()
    
    # Test 5: Spectral gap
    print("\n" + "#"*70)
    print("# TEST 5: SPECTRAL GAP")
    print("#"*70)
    all_results['spectral_gap'] = test_spectral_gap()
    
    # Summary
    print("\n" + "="*70)
    print("PHASE 3 SUMMARY")
    print("="*70)
    print("""
    ORDER PARAMETER CANDIDATES:
    
    1. Effective Rank (R_eff)
       - Does it show transition at T_c?
       - Does χ_R scale properly with n?
    
    2. Weight Overlap Variance
       - Already showed RSB signature
       - Check if signal strengthens at larger n
    
    3. Spectral Gap (λ_1/λ_2)
       - Indicates low-rank structure
       - May correlate with generalization
    
    4. Training Dynamics
       - Different behavior above/below T_c?
       - Rank reduction pattern?
    """)
    
    # Save results
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {str(k): convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        else:
            return obj
    
    with open('phase3_results.json', 'w') as f:
        json.dump(convert_for_json(all_results), f, indent=2)
    print("\nResults saved to phase3_results.json")
    
    return all_results

if __name__ == "__main__":
    results = main()
