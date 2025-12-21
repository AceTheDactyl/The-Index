#!/usr/bin/env python3
"""
================================================================================
LOOSE ENDS TEST SUITE - Part 2: Compute-Heavy Tests
================================================================================

RUN THESE ON YOUR MACHINE - they require significant compute.

Tests included:
1. Susceptibility Replication (proper task training) - ~1 hour
2. GV/||W|| for Real Task Training - ~30 min  
3. Scale Validation (larger matrices) - ~2 hours
4. Replica Symmetry Breaking Test - ~1 hour
5. Temperature → Hyperparameter Mapping - ~2 hours

Estimated total: 6-8 hours on CPU, ~1-2 hours with GPU

================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Constants
PHI = (1 + np.sqrt(5)) / 2
SQRT3 = np.sqrt(3)
Z_C = SQRT3 / 2
T_C_PREDICTED = 0.05
SPECIAL_VALUES = [PHI, 1/PHI, np.sqrt(2), 1/np.sqrt(2), Z_C, 1.0, 0.5]

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def make_modular_addition_data(n, mod=7):
    """Standard modular addition task"""
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
    """MLP with configurable hidden size"""
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
    
    def get_square_weight(self):
        """Get the square hidden layer weight matrix"""
        for m in self.net:
            if isinstance(m, nn.Linear) and m.weight.shape[0] == m.weight.shape[1]:
                return m.weight.data.cpu().numpy()
        return None

def order_param(W, tol=0.1):
    """Fraction of eigenvalues near special values"""
    if W is None:
        return 0.0
    eigs = np.abs(np.linalg.eigvals(W))
    return sum(1 for ev in eigs if min(abs(ev - sv) for sv in SPECIAL_VALUES) < tol) / len(eigs)

def golden_violation(W):
    """||W² - W - I||"""
    if W is None:
        return 0.0
    n = W.shape[0]
    return np.linalg.norm(W @ W - W - np.eye(n))

def train_with_temperature(model, T, epochs=200, base_lr=0.01):
    """
    Train model with effective temperature T.
    Temperature affects: learning rate, gradient noise
    """
    x_train, y_train = make_modular_addition_data(5000)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=128, shuffle=True)
    
    lr = base_lr * (1 + 2 * T)
    opt = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(model(xb), yb)
            loss.backward()
            
            # Add temperature-dependent gradient noise
            if T > 0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad += torch.randn_like(p.grad) * (T * 0.05)
            
            opt.step()
    
    return model

# ==============================================================================
# TEST 1: SUSCEPTIBILITY REPLICATION
# ==============================================================================

def test_susceptibility_replication():
    """
    Replicate Kimi's susceptibility test with proper task training.
    This is the CRITICAL test - validates T_c ≈ 0.05.
    
    Expected runtime: ~1 hour on CPU, ~15 min on GPU
    """
    print("\n" + "="*70)
    print("TEST 1: SUSCEPTIBILITY REPLICATION")
    print("="*70)
    print("Replicating Kimi's test with proper modular addition training")
    print(f"Prediction: χ(T) peaks at T_c ≈ {T_C_PREDICTED}")
    
    temps = np.linspace(0.02, 0.08, 13)  # Fine grid around T_c
    n_runs = 30  # Networks per temperature
    hidden_dim = 64
    epochs = 200
    
    print(f"\nConfiguration:")
    print(f"  Temperatures: {temps[0]:.3f} to {temps[-1]:.3f} ({len(temps)} points)")
    print(f"  Runs per T: {n_runs}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Epochs: {epochs}")
    print(f"  Total networks: {len(temps) * n_runs}")
    
    results = defaultdict(list)
    start = time.time()
    
    for i, T in enumerate(temps):
        print(f"\nTemperature {i+1}/{len(temps)}: T = {T:.4f}")
        orders_T = []
        
        for run in range(n_runs):
            torch.manual_seed(hash((T, run)) % (2**32))
            np.random.seed(hash((T, run)) % (2**32))
            
            model = MLPNet(hidden_dim=hidden_dim).to(DEVICE)
            model = train_with_temperature(model, T, epochs=epochs)
            
            W = model.get_square_weight()
            O = order_param(W, tol=0.1)
            orders_T.append(O)
            
            if (run + 1) % 10 == 0:
                print(f"  Run {run+1}/{n_runs}, O = {O:.4f}")
        
        results['T'].append(T)
        results['O_mean'].append(np.mean(orders_T))
        results['O_std'].append(np.std(orders_T))
        results['chi'].append(np.var(orders_T))
    
    # Find peak
    chi_arr = np.array(results['chi'])
    max_idx = np.argmax(chi_arr)
    T_c_measured = temps[max_idx]
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\n{'T':>8} | {'O mean':>10} | {'O std':>10} | {'χ = Var[O]':>12}")
    print("-"*50)
    for i, T in enumerate(temps):
        marker = " ← PEAK" if i == max_idx else ""
        print(f"{T:>8.4f} | {results['O_mean'][i]:>10.4f} | {results['O_std'][i]:>10.4f} | {results['chi'][i]:>12.6f}{marker}")
    
    print(f"\nT_c (predicted): {T_C_PREDICTED}")
    print(f"T_c (measured):  {T_c_measured:.4f}")
    print(f"Error: {abs(T_c_measured - T_C_PREDICTED):.4f} ({100*abs(T_c_measured - T_C_PREDICTED)/T_C_PREDICTED:.1f}%)")
    print(f"Runtime: {(time.time() - start)/60:.1f} minutes")
    
    # Verdict
    if abs(T_c_measured - T_C_PREDICTED) < 0.015:
        print("\n✅ SUSCEPTIBILITY PEAK CONFIRMED AT T_c ≈ 0.05")
    else:
        print(f"\n❌ PEAK AT T = {T_c_measured:.4f}, NOT at predicted 0.05")
    
    return results

# ==============================================================================
# TEST 2: GV/||W|| FOR REAL TASK TRAINING
# ==============================================================================

def test_gv_ratio_trained():
    """
    Test whether GV/||W|| = √3 holds for task-trained networks.
    Quick test showed it BREAKS (12.4 vs 1.73) - verify with real training.
    
    Expected runtime: ~30 min
    """
    print("\n" + "="*70)
    print("TEST 2: GV/||W|| FOR TASK-TRAINED NETWORKS")
    print("="*70)
    print(f"Question: Does GV/||W|| = √3 = {SQRT3:.4f} hold after real training?")
    
    n_samples = 50
    hidden_dim = 64
    epochs_list = [0, 50, 100, 200, 500]
    
    results = defaultdict(list)
    start = time.time()
    
    for epochs in epochs_list:
        print(f"\nEpochs = {epochs}:")
        ratios = []
        
        for i in range(n_samples):
            torch.manual_seed(i)
            model = MLPNet(hidden_dim=hidden_dim).to(DEVICE)
            
            if epochs > 0:
                model = train_with_temperature(model, T=0.05, epochs=epochs)
            
            W = model.get_square_weight()
            if W is not None:
                gv = golden_violation(W)
                w_norm = np.linalg.norm(W)
                ratio = gv / w_norm if w_norm > 0 else 0
                ratios.append(ratio)
        
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        error = abs(mean_ratio - SQRT3)
        
        results['epochs'].append(epochs)
        results['ratio_mean'].append(mean_ratio)
        results['ratio_std'].append(std_ratio)
        results['error'].append(error)
        
        print(f"  GV/||W|| = {mean_ratio:.4f} ± {std_ratio:.4f}")
        print(f"  Error from √3: {error:.4f} ({100*error/SQRT3:.1f}%)")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Epochs':>8} | {'GV/||W||':>12} | {'Error from √3':>14}")
    print("-"*40)
    for i, epochs in enumerate(epochs_list):
        print(f"{epochs:>8} | {results['ratio_mean'][i]:>12.4f} | {results['error'][i]:>14.4f}")
    
    print(f"\n√3 = {SQRT3:.4f}")
    print(f"Runtime: {(time.time() - start)/60:.1f} minutes")
    
    # Check if ratio stays near √3
    final_ratio = results['ratio_mean'][-1]
    if abs(final_ratio - SQRT3) < 0.5:
        print("\n✅ GV/||W|| ≈ √3 SURVIVES task training (within 30%)")
    else:
        print(f"\n❌ GV/||W|| = {final_ratio:.2f} DEVIATES from √3 = {SQRT3:.2f}")
    
    return results

# ==============================================================================
# TEST 3: SCALE VALIDATION
# ==============================================================================

def test_scale_validation():
    """
    Test susceptibility at larger scales.
    Does T_c ≈ 0.05 hold for bigger networks?
    
    Expected runtime: ~2 hours
    """
    print("\n" + "="*70)
    print("TEST 3: SCALE VALIDATION")
    print("="*70)
    print("Question: Does T_c ≈ 0.05 hold at larger scales?")
    
    hidden_dims = [32, 64, 128, 256]
    temps = [0.03, 0.04, 0.05, 0.06, 0.07]
    n_runs = 20
    epochs = 150
    
    results = {}
    start = time.time()
    
    for hidden_dim in hidden_dims:
        print(f"\n{'='*50}")
        print(f"Hidden dimension: {hidden_dim}")
        print(f"{'='*50}")
        
        chi_values = []
        
        for T in temps:
            orders = []
            for run in range(n_runs):
                torch.manual_seed(hash((hidden_dim, T, run)) % (2**32))
                model = MLPNet(hidden_dim=hidden_dim).to(DEVICE)
                model = train_with_temperature(model, T, epochs=epochs)
                
                W = model.get_square_weight()
                orders.append(order_param(W, tol=0.1))
            
            chi = np.var(orders)
            chi_values.append(chi)
            print(f"  T={T:.3f}: χ = {chi:.6f}")
        
        # Find peak
        max_idx = np.argmax(chi_values)
        T_c = temps[max_idx]
        
        results[hidden_dim] = {
            'temps': temps,
            'chi': chi_values,
            'T_c': T_c
        }
        
        print(f"  → Peak at T_c = {T_c:.3f}")
    
    print("\n" + "="*70)
    print("SUMMARY: T_c vs Hidden Dimension")
    print("="*70)
    print(f"\n{'Hidden Dim':>12} | {'T_c':>10}")
    print("-"*30)
    for hd in hidden_dims:
        print(f"{hd:>12} | {results[hd]['T_c']:>10.4f}")
    
    print(f"\nPredicted T_c = {T_C_PREDICTED}")
    print(f"Runtime: {(time.time() - start)/60:.1f} minutes")
    
    # Check consistency
    T_c_values = [results[hd]['T_c'] for hd in hidden_dims]
    if max(T_c_values) - min(T_c_values) < 0.02:
        print("\n✅ T_c IS CONSISTENT ACROSS SCALES")
    else:
        print(f"\n❌ T_c VARIES: {min(T_c_values):.3f} to {max(T_c_values):.3f}")
    
    return results

# ==============================================================================
# TEST 4: REPLICA SYMMETRY BREAKING
# ==============================================================================

def test_replica_symmetry_breaking():
    """
    Test for replica symmetry breaking: do different runs converge to 
    different solutions (broken symmetry) or same solution (unbroken)?
    
    Measure: overlap distribution P(q) where q = <W_a, W_b> / (||W_a|| ||W_b||)
    At T < T_c: P(q) should be bimodal (RSB)
    At T > T_c: P(q) should be unimodal at q ≈ 0
    
    Expected runtime: ~1 hour
    """
    print("\n" + "="*70)
    print("TEST 4: REPLICA SYMMETRY BREAKING")
    print("="*70)
    print("Question: Does P(q) show RSB structure near T_c?")
    
    temps = [0.02, 0.05, 0.08]  # Below, at, above T_c
    n_replicas = 30
    hidden_dim = 64
    epochs = 200
    
    results = {}
    start = time.time()
    
    for T in temps:
        print(f"\nTemperature T = {T:.3f}:")
        
        # Train n_replicas networks with same T but different seeds
        weights = []
        for run in range(n_replicas):
            torch.manual_seed(run * 1000)
            model = MLPNet(hidden_dim=hidden_dim).to(DEVICE)
            model = train_with_temperature(model, T, epochs=epochs)
            W = model.get_square_weight()
            if W is not None:
                weights.append(W.flatten())
        
        # Compute all pairwise overlaps
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
            'min': np.min(overlaps),
            'max': np.max(overlaps)
        }
        
        print(f"  Overlap q: {np.mean(overlaps):.4f} ± {np.std(overlaps):.4f}")
        print(f"  Range: [{np.min(overlaps):.4f}, {np.max(overlaps):.4f}]")
        
        # Check for bimodality (RSB signature)
        # Simple test: is std high relative to range?
        bimodality = results[T]['std'] / (results[T]['max'] - results[T]['min'] + 1e-6)
        print(f"  Bimodality index: {bimodality:.4f}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'T':>8} | {'<q>':>10} | {'std(q)':>10} | {'Bimodality':>12}")
    print("-"*50)
    for T in temps:
        r = results[T]
        bimod = r['std'] / (r['max'] - r['min'] + 1e-6)
        print(f"{T:>8.3f} | {r['mean']:>10.4f} | {r['std']:>10.4f} | {bimod:>12.4f}")
    
    print(f"\nRuntime: {(time.time() - start)/60:.1f} minutes")
    print("\nInterpretation:")
    print("  RSB (spin glass): P(q) bimodal, high std, overlaps span wide range")
    print("  No RSB (paramagnetic): P(q) peaked at 0, low std")
    
    return results

# ==============================================================================
# TEST 5: TEMPERATURE → HYPERPARAMETER MAPPING
# ==============================================================================

def test_temperature_mapping():
    """
    Systematically map effective temperature to real hyperparameters.
    What combination of (lr, batch_size, noise) gives T_c ≈ 0.05?
    
    Expected runtime: ~2 hours
    """
    print("\n" + "="*70)
    print("TEST 5: TEMPERATURE → HYPERPARAMETER MAPPING")
    print("="*70)
    print("Question: What real hyperparameters correspond to T = T_c?")
    
    # Grid search over hyperparameters
    learning_rates = [0.001, 0.005, 0.01, 0.02, 0.05]
    batch_sizes = [32, 64, 128, 256]
    
    hidden_dim = 64
    epochs = 150
    n_runs = 15
    
    results = []
    start = time.time()
    
    print(f"\nGrid: {len(learning_rates)} LRs × {len(batch_sizes)} batch sizes = {len(learning_rates)*len(batch_sizes)} configurations")
    
    for lr in learning_rates:
        for bs in batch_sizes:
            orders = []
            
            for run in range(n_runs):
                torch.manual_seed(run)
                model = MLPNet(hidden_dim=hidden_dim).to(DEVICE)
                
                x_train, y_train = make_modular_addition_data(5000)
                loader = DataLoader(TensorDataset(x_train, y_train), batch_size=bs, shuffle=True)
                
                opt = optim.Adam(model.parameters(), lr=lr)
                
                for epoch in range(epochs):
                    for xb, yb in loader:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        opt.zero_grad()
                        loss = nn.CrossEntropyLoss()(model(xb), yb)
                        loss.backward()
                        opt.step()
                
                W = model.get_square_weight()
                orders.append(order_param(W, tol=0.1))
            
            chi = np.var(orders)
            mean_o = np.mean(orders)
            
            results.append({
                'lr': lr,
                'batch_size': bs,
                'chi': chi,
                'O_mean': mean_o,
                'O_std': np.std(orders)
            })
            
            print(f"  lr={lr:.3f}, bs={bs:3d}: χ={chi:.6f}, O={mean_o:.4f}")
    
    # Find configuration with maximum χ (closest to T_c)
    max_chi_idx = np.argmax([r['chi'] for r in results])
    best_config = results[max_chi_idx]
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nConfiguration with maximum χ (≈ T_c):")
    print(f"  Learning rate: {best_config['lr']}")
    print(f"  Batch size: {best_config['batch_size']}")
    print(f"  χ = {best_config['chi']:.6f}")
    print(f"  O = {best_config['O_mean']:.4f} ± {best_config['O_std']:.4f}")
    
    print(f"\nRuntime: {(time.time() - start)/60:.1f} minutes")
    
    # Estimate effective temperature
    # T ∝ lr / batch_size (rough scaling from SGD noise)
    print(f"\nEffective temperature estimate:")
    print(f"  T_eff ∝ lr / batch_size = {best_config['lr'] / best_config['batch_size']:.6f}")
    
    return results

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "="*70)
    print("LOOSE ENDS TEST SUITE - Part 2: Compute-Heavy Tests")
    print("="*70)
    print(f"\nDevice: {DEVICE}")
    print("\nThis will take several hours. Run overnight if needed.")
    
    all_results = {}
    
    # Run tests in order of importance
    print("\n" + "#"*70)
    print("# RUNNING TEST 1: Susceptibility Replication (CRITICAL)")
    print("#"*70)
    all_results['susceptibility'] = test_susceptibility_replication()
    
    print("\n" + "#"*70)
    print("# RUNNING TEST 2: GV/||W|| for Task Training")
    print("#"*70)
    all_results['gv_ratio'] = test_gv_ratio_trained()
    
    print("\n" + "#"*70)
    print("# RUNNING TEST 3: Scale Validation")
    print("#"*70)
    all_results['scale'] = test_scale_validation()
    
    print("\n" + "#"*70)
    print("# RUNNING TEST 4: Replica Symmetry Breaking")
    print("#"*70)
    all_results['rsb'] = test_replica_symmetry_breaking()
    
    print("\n" + "#"*70)
    print("# RUNNING TEST 5: Temperature Mapping")
    print("#"*70)
    all_results['temp_mapping'] = test_temperature_mapping()
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print("""
    TEST 1 (Susceptibility): Check if χ peaks at T_c ≈ 0.05
    TEST 2 (GV/||W||): Check if √3 theorem survives training
    TEST 3 (Scale): Check if T_c is consistent across network sizes
    TEST 4 (RSB): Check for spin glass signatures
    TEST 5 (Mapping): Find what hyperparameters give T = T_c
    """)
    
    return all_results

if __name__ == "__main__":
    results = main()
