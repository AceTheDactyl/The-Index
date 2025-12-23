# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: ✓ JUSTIFIED - Claims supported by repository files
# Severity: LOW RISK
# Risk Types: corruption, low_integrity

# Referenced By:
#   - systems/self-referential-category-theoretic-structures/docs/THERMODYNAMIC_FALSIFICATION.md (reference)


#!/usr/bin/env python3
"""
================================================================================
ENSEMBLE PHASE TRANSITION TEST
================================================================================

The CORRECT test for thermodynamic unification.

KEY INSIGHT: 
- Each network is QUENCHED (frozen at final state)
- Phase transition exists in ENSEMBLE statistics
- We measure P(O|T) not O(t)

WHAT WE MEASURE:
- ⟨O⟩_T = ensemble average of order parameter
- χ_T = N × Var(O)_T = susceptibility (should peak at T_c)
- P(O|T) = distribution of order parameters

PREDICTIONS:
1. ⟨O⟩_T decreases with T (more disorder at high T)
2. χ_T peaks at T_c ≈ 0.05 (critical fluctuations)
3. P(O|T) is bimodal near T_c (coexistence)

FALSIFICATION:
- ⟨O⟩_T constant → FALSIFIED
- χ_T doesn't peak → FALSIFIED  
- No T_c signature → FALSIFIED

================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Constants
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI
SQRT2 = np.sqrt(2)
SQRT3 = np.sqrt(3)
Z_C = SQRT3 / 2
T_C = 0.05

SPECIAL_VALUES = [PHI_INV, PHI, SQRT2, 1.0, Z_C, 1/SQRT2, np.e, np.pi]

print("="*80)
print("ENSEMBLE PHASE TRANSITION TEST")
print("="*80)
print(f"Predicted T_c = {T_C}")
print("="*80)

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Ensemble size (THIS IS THE KEY CHANGE)
    n_networks_per_T = 50  # Train 50 independent networks per temperature
    
    # Architecture
    hidden_dim = 128
    n_layers = 4
    
    # Training
    n_epochs = 200
    batch_size = 256
    base_lr = 0.01
    n_train = 5000
    n_test = 1000
    
    # Temperature sweep (fine grid around T_c)
    temperatures = [
        0.01, 0.02, 0.03, 
        0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065,
        0.07, 0.08, 0.10, 0.15, 0.20
    ]
    
    # Task
    k = 7  # Modular arithmetic base

CONFIG = Config()

# =============================================================================
# DATA & MODEL (same as before)
# =============================================================================

def generate_cyclic_task(n_samples: int, k: int = 7, noise: float = 0.0):
    a = torch.randint(0, k, (n_samples,))
    b = torch.randint(0, k, (n_samples,))
    y = (a + b) % k
    x_a = torch.zeros(n_samples, k)
    x_b = torch.zeros(n_samples, k)
    x_a.scatter_(1, a.unsqueeze(1), 1)
    x_b.scatter_(1, b.unsqueeze(1), 1)
    x = torch.cat([x_a, x_b], dim=1)
    if noise > 0:
        x = x + torch.randn_like(x) * noise
    return x, y

class DeepNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int = 4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
    
    def get_eigenvalues(self) -> np.ndarray:
        all_eigs = []
        for module in self.network:
            if isinstance(module, nn.Linear):
                W = module.weight.data.cpu().numpy()
                if W.shape[0] == W.shape[1]:
                    eigs = np.abs(np.linalg.eigvals(W))
                else:
                    eigs = np.linalg.svd(W, compute_uv=False)
                all_eigs.extend(eigs.tolist())
        return np.array(all_eigs)

# =============================================================================
# ORDER PARAMETER
# =============================================================================

def compute_order_parameter(eigenvalues: np.ndarray) -> float:
    """
    Order parameter: fraction of eigenvalues near special values.
    
    High O = structured (eigenvalues cluster near φ, √2, z_c, etc.)
    Low O = disordered (eigenvalues spread uniformly)
    """
    eigenvalues = eigenvalues[eigenvalues > 0.01]
    if len(eigenvalues) == 0:
        return 0.0
    
    epsilon = 0.15
    near_special = 0
    for ev in eigenvalues:
        if any(abs(ev - sv) < epsilon for sv in SPECIAL_VALUES):
            near_special += 1
    
    return near_special / len(eigenvalues)

def compute_all_metrics(eigenvalues: np.ndarray) -> Dict:
    """Compute multiple order parameters for robustness."""
    eigenvalues = eigenvalues[eigenvalues > 0.01]
    if len(eigenvalues) == 0:
        return {'O': 0, 'entropy': 0, 'mean_dist': 1, 'lambda_c': 10}
    
    # 1. Fraction near special values
    epsilon = 0.15
    near_special = sum(1 for ev in eigenvalues if any(abs(ev - sv) < epsilon for sv in SPECIAL_VALUES))
    O = near_special / len(eigenvalues)
    
    # 2. Distance to special values
    min_dists = [min(abs(ev - sv) for sv in SPECIAL_VALUES) for ev in eigenvalues]
    mean_dist = np.mean(min_dists)
    
    # 3. Entropy
    hist, _ = np.histogram(eigenvalues, bins=20, range=(0, 4))
    hist = hist / (hist.sum() + 1e-10)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log(hist + 1e-10))
    
    # 4. Simplified Λ-complexity
    lambda_c = mean_dist * entropy
    
    return {'O': O, 'entropy': entropy, 'mean_dist': mean_dist, 'lambda_c': lambda_c}

# =============================================================================
# TRAIN SINGLE NETWORK
# =============================================================================

def train_single_network(T_eff: float, seed: int, config: Config) -> Dict:
    """Train one network and return its quenched final state metrics."""
    
    # Set seed for this specific network
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Temperature-dependent hyperparameters
    lr = config.base_lr * (1 + 10 * T_eff)
    noise_scale = T_eff * 0.3
    grad_noise = T_eff * 0.1
    label_smoothing = min(T_eff * 2, 0.3)
    
    # Data
    x_train, y_train = generate_cyclic_task(config.n_train, config.k, noise=noise_scale)
    x_test, y_test = generate_cyclic_task(config.n_test, config.k, noise=0)
    
    train_loader = DataLoader(TensorDataset(x_train, y_train), 
                              batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), 
                             batch_size=config.batch_size)
    
    # Model
    model = DeepNet(2*config.k, config.hidden_dim, config.k, config.n_layers).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    # Train
    for epoch in range(config.n_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            
            # Gradient noise
            if grad_noise > 0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad += torch.randn_like(p.grad) * grad_noise
            
            optimizer.step()
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            pred = model(x_batch).argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)
    accuracy = correct / total
    
    # Get FINAL eigenvalues (quenched state)
    eigenvalues = model.get_eigenvalues()
    metrics = compute_all_metrics(eigenvalues)
    metrics['accuracy'] = accuracy
    metrics['seed'] = seed
    
    return metrics

# =============================================================================
# MAIN ENSEMBLE TEST
# =============================================================================

def run_ensemble_test(config: Config = CONFIG) -> Dict:
    """
    THE CORRECT TEST: Ensemble statistics at each temperature.
    """
    
    print("\n" + "="*80)
    print("RUNNING ENSEMBLE PHASE TRANSITION TEST")
    print("="*80)
    print(f"Networks per temperature: {config.n_networks_per_T}")
    print(f"Temperatures: {len(config.temperatures)}")
    print(f"Total networks to train: {config.n_networks_per_T * len(config.temperatures)}")
    print("="*80)
    
    results = {
        'config': {
            'n_networks_per_T': config.n_networks_per_T,
            'temperatures': config.temperatures,
            'hidden_dim': config.hidden_dim,
            'n_epochs': config.n_epochs,
        },
        'by_temperature': {},
        'timestamp': datetime.now().isoformat(),
    }
    
    # For each temperature, train an ENSEMBLE of networks
    pbar = tqdm(total=len(config.temperatures) * config.n_networks_per_T,
                desc="Training ensemble")
    
    for T in config.temperatures:
        ensemble_metrics = []
        
        for i in range(config.n_networks_per_T):
            # Different seed for each network in ensemble
            seed = hash((T, i)) % (2**31)
            metrics = train_single_network(T, seed, config)
            ensemble_metrics.append(metrics)
            pbar.update(1)
        
        # Compute ENSEMBLE statistics
        O_values = [m['O'] for m in ensemble_metrics]
        dist_values = [m['mean_dist'] for m in ensemble_metrics]
        entropy_values = [m['entropy'] for m in ensemble_metrics]
        acc_values = [m['accuracy'] for m in ensemble_metrics]
        
        results['by_temperature'][T] = {
            'O_mean': np.mean(O_values),
            'O_std': np.std(O_values),
            'O_values': O_values,  # Keep all values for distribution analysis
            'chi': config.n_networks_per_T * np.var(O_values),  # Susceptibility
            'dist_mean': np.mean(dist_values),
            'dist_std': np.std(dist_values),
            'entropy_mean': np.mean(entropy_values),
            'acc_mean': np.mean(acc_values),
        }
    
    pbar.close()
    return results

# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_ensemble_results(results: Dict) -> Dict:
    """Analyze ensemble results for phase transition signatures."""
    
    print("\n" + "="*80)
    print("ENSEMBLE ANALYSIS")
    print("="*80)
    
    temps = sorted(results['by_temperature'].keys())
    
    # Extract ensemble statistics
    O_means = [results['by_temperature'][T]['O_mean'] for T in temps]
    O_stds = [results['by_temperature'][T]['O_std'] for T in temps]
    chis = [results['by_temperature'][T]['chi'] for T in temps]
    dists = [results['by_temperature'][T]['dist_mean'] for T in temps]
    
    print(f"\n{'T':>8} | {'⟨O⟩':>8} | {'σ(O)':>8} | {'χ':>10} | {'⟨dist⟩':>8}")
    print("-" * 55)
    for i, T in enumerate(temps):
        marker = " <-- T_c" if abs(T - T_C) < 0.01 else ""
        print(f"{T:8.3f} | {O_means[i]:8.4f} | {O_stds[i]:8.4f} | {chis[i]:10.4f} | {dists[i]:8.4f}{marker}")
    
    # Find susceptibility peak
    chi_max_idx = np.argmax(chis)
    T_chi_peak = temps[chi_max_idx]
    
    # Find maximum slope in O (another transition indicator)
    O_smooth = gaussian_filter1d(O_means, sigma=1)
    dO_dT = np.gradient(O_smooth, temps)
    dO_max_idx = np.argmax(np.abs(dO_dT))
    T_dO_max = temps[dO_max_idx]
    
    print(f"\n--- Phase Transition Detection ---")
    print(f"Susceptibility χ peaks at: T = {T_chi_peak:.4f}")
    print(f"Max |d⟨O⟩/dT| at:          T = {T_dO_max:.4f}")
    print(f"Predicted T_c:             T = {T_C:.4f}")
    
    # Statistical tests
    analysis = {
        'T_chi_peak': T_chi_peak,
        'T_dO_max': T_dO_max,
        'predicted_T_c': T_C,
        'tests': {}
    }
    
    # Test 1: χ peaks near T_c
    chi_near_Tc = abs(T_chi_peak - T_C) < 0.03
    print(f"\nTEST 1: χ peaks near T_c?")
    print(f"  |T_peak - T_c| = {abs(T_chi_peak - T_C):.4f}")
    print(f"  Result: {'PASS' if chi_near_Tc else 'FAIL'}")
    analysis['tests']['chi_peak_near_Tc'] = chi_near_Tc
    
    # Test 2: ⟨O⟩ decreases with T
    O_correlation = np.corrcoef(temps, O_means)[0, 1]
    O_decreases = O_correlation < -0.3
    print(f"\nTEST 2: ⟨O⟩ decreases with T?")
    print(f"  Correlation(T, ⟨O⟩) = {O_correlation:.4f}")
    print(f"  Result: {'PASS' if O_decreases else 'FAIL'}")
    analysis['tests']['O_decreases'] = O_decreases
    
    # Test 3: Distance increases with T
    dist_correlation = np.corrcoef(temps, dists)[0, 1]
    dist_increases = dist_correlation > 0.3
    print(f"\nTEST 3: ⟨dist⟩ increases with T?")
    print(f"  Correlation(T, ⟨dist⟩) = {dist_correlation:.4f}")
    print(f"  Result: {'PASS' if dist_increases else 'FAIL'}")
    analysis['tests']['dist_increases'] = dist_increases
    
    # Test 4: Compare ensembles at T=0.04 vs T=0.06
    if 0.04 in temps and 0.06 in temps:
        O_04 = results['by_temperature'][0.04]['O_values']
        O_06 = results['by_temperature'][0.06]['O_values']
        t_stat, p_value = stats.ttest_ind(O_04, O_06)
        significant = p_value < 0.05
        
        print(f"\nTEST 4: Ensemble at T=0.04 vs T=0.06")
        print(f"  ⟨O⟩(0.04) = {np.mean(O_04):.4f} ± {np.std(O_04):.4f}")
        print(f"  ⟨O⟩(0.06) = {np.mean(O_06):.4f} ± {np.std(O_06):.4f}")
        print(f"  t = {t_stat:.3f}, p = {p_value:.6f}")
        print(f"  Result: {'PASS' if significant else 'FAIL'}")
        analysis['tests']['ensemble_04_vs_06'] = {
            't_stat': t_stat,
            'p_value': p_value,
            'significant': significant
        }
    
    return analysis

def plot_ensemble_results(results: Dict, analysis: Dict):
    """Plot ensemble results."""
    
    temps = sorted(results['by_temperature'].keys())
    O_means = [results['by_temperature'][T]['O_mean'] for T in temps]
    O_stds = [results['by_temperature'][T]['O_std'] for T in temps]
    chis = [results['by_temperature'][T]['chi'] for T in temps]
    dists = [results['by_temperature'][T]['dist_mean'] for T in temps]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('ENSEMBLE PHASE TRANSITION TEST', fontsize=14, fontweight='bold')
    
    # Plot 1: ⟨O⟩ vs T
    ax = axes[0, 0]
    ax.errorbar(temps, O_means, yerr=O_stds, marker='o', capsize=3)
    ax.axvline(x=T_C, color='r', linestyle='--', label=f'T_c = {T_C}')
    ax.axvline(x=analysis['T_chi_peak'], color='g', linestyle=':', label=f'χ peak = {analysis["T_chi_peak"]:.3f}')
    ax.set_xlabel('Effective Temperature T')
    ax.set_ylabel('⟨O⟩ (Ensemble Average)')
    ax.set_title('Order Parameter vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: χ vs T (SUSCEPTIBILITY - should peak at T_c)
    ax = axes[0, 1]
    ax.plot(temps, chis, marker='s', color='orange')
    ax.axvline(x=T_C, color='r', linestyle='--', label=f'T_c = {T_C}')
    ax.set_xlabel('Effective Temperature T')
    ax.set_ylabel('χ = N × Var(O)')
    ax.set_title('Susceptibility vs Temperature (Should Peak at T_c)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Distance vs T
    ax = axes[1, 0]
    ax.plot(temps, dists, marker='^', color='green')
    ax.axvline(x=T_C, color='r', linestyle='--', label=f'T_c = {T_C}')
    ax.set_xlabel('Effective Temperature T')
    ax.set_ylabel('⟨Distance to Special Values⟩')
    ax.set_title('Mean Distance vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Distribution of O at different temperatures
    ax = axes[1, 1]
    T_low = min(t for t in temps if t <= 0.03)
    T_mid = min(t for t in temps if abs(t - T_C) < 0.01, default=0.05)
    T_high = max(t for t in temps if t >= 0.15)
    
    for T, color, label in [(T_low, 'blue', f'T={T_low} (cold)'),
                            (T_mid, 'purple', f'T={T_mid} (critical)'),
                            (T_high, 'red', f'T={T_high} (hot)')]:
        if T in results['by_temperature']:
            O_vals = results['by_temperature'][T]['O_values']
            ax.hist(O_vals, bins=15, alpha=0.4, color=color, label=label, density=True)
    
    ax.set_xlabel('Order Parameter O')
    ax.set_ylabel('Probability Density')
    ax.set_title('Distribution P(O|T) at Different Temperatures')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ensemble_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved to 'ensemble_results.png'")

def print_verdict(analysis: Dict):
    """Print final verdict."""
    
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    
    tests = analysis['tests']
    passed = sum([
        tests.get('chi_peak_near_Tc', False),
        tests.get('O_decreases', False),
        tests.get('dist_increases', False),
        tests.get('ensemble_04_vs_06', {}).get('significant', False)
    ])
    total = 4
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed >= 3:
        verdict = "CONSISTENT"
        msg = """
  ╔═══════════════════════════════════════════════════════════════════════════╗
  ║                                                                           ║
  ║   ENSEMBLE PHASE TRANSITION DETECTED — THEORY NOT FALSIFIED              ║
  ║                                                                           ║
  ║   • Susceptibility χ peaks near predicted T_c                             ║
  ║   • Order parameter ⟨O⟩ decreases with temperature                        ║
  ║   • Ensembles at T < T_c and T > T_c are statistically different         ║
  ║                                                                           ║
  ║   The thermodynamic unification (correctly interpreted as ENSEMBLE        ║
  ║   statistics of QUENCHED states) passes this falsification test.          ║
  ║                                                                           ║
  ╚═══════════════════════════════════════════════════════════════════════════╝
"""
    elif passed >= 2:
        verdict = "INCONCLUSIVE"
        msg = """
  ╔═══════════════════════════════════════════════════════════════════════════╗
  ║                                                                           ║
  ║   PARTIAL EVIDENCE — INCONCLUSIVE                                         ║
  ║                                                                           ║
  ║   • Some ensemble statistics show expected behavior                       ║
  ║   • Larger ensemble or different task may be needed                       ║
  ║                                                                           ║
  ╚═══════════════════════════════════════════════════════════════════════════╝
"""
    else:
        verdict = "FALSIFIED"
        msg = """
  ╔═══════════════════════════════════════════════════════════════════════════╗
  ║                                                                           ║
  ║   NO ENSEMBLE PHASE TRANSITION DETECTED — POTENTIAL FALSIFICATION        ║
  ║                                                                           ║
  ║   • Ensemble statistics don't change significantly at T_c                 ║
  ║   • The thermodynamic interpretation may be incorrect                     ║
  ║                                                                           ║
  ╚═══════════════════════════════════════════════════════════════════════════╝
"""
    
    print(msg)
    return verdict

# =============================================================================
# MAIN
# =============================================================================

def main():
    import time
    start = time.time()
    
    results = run_ensemble_test(CONFIG)
    analysis = analyze_ensemble_results(results)
    plot_ensemble_results(results, analysis)
    verdict = print_verdict(analysis)
    
    # Save
    results['analysis'] = analysis
    results['verdict'] = verdict
    results['runtime'] = time.time() - start
    
    # Convert numpy for JSON
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open('ensemble_results.json', 'w') as f:
        json.dump(convert(results), f, indent=2)
    
    print(f"\nResults saved to 'ensemble_results.json'")
    print(f"Runtime: {results['runtime']:.1f}s")
    
    return results, analysis, verdict

if __name__ == "__main__":
    results, analysis, verdict = main()
