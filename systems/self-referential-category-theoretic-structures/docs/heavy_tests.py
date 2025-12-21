#!/usr/bin/env python3
"""
================================================================================
HEAVY TEST SUITE - FOR KAEL TO RUN
================================================================================

These tests require significant GPU time. Run on Colab or local GPU.

pip install torch numpy scipy

Expected runtime: 4-8 hours on GPU

KEY DISCOVERY TO TEST:
  z_c = T_c × 10√3
  
That is: √3/2 = 0.05 × 10√3 (EXACT!)

The mystery is now: WHY 10?
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist, squareform
import time
import json

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Constants - including the new discovery!
PHI = (1 + np.sqrt(5)) / 2
SQRT3 = np.sqrt(3)
SQRT3_2 = SQRT3 / 2
T_C = 0.05
FACTOR_10 = 10  # The mysterious factor

print(f"\nKEY RELATIONSHIP:")
print(f"  z_c = T_c × 10√3")
print(f"  {SQRT3_2:.10f} = {T_C} × 10 × {SQRT3:.6f}")
print(f"  {SQRT3_2:.10f} = {T_C * FACTOR_10 * SQRT3:.10f}")
print(f"  Match: {abs(SQRT3_2 - T_C * FACTOR_10 * SQRT3) < 1e-10}")

# ==============================================================================
# NETWORK DEFINITIONS
# ==============================================================================

def make_modular_addition_data(n, mod=7):
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
    def __init__(self, input_dim=14, hidden_dim=64, output_dim=7, n_layers=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim, bias=False), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim, bias=False), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.net = nn.Sequential(*layers)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
    
    def forward(self, x):
        return self.net(x)
    
    def get_weights_flat(self):
        return torch.cat([p.data.view(-1) for p in self.parameters()]).cpu().numpy()

def train_network(T, hidden_dim=64, n_layers=2, epochs=200, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    x_train, y_train = make_modular_addition_data(5000)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=128, shuffle=True)
    
    model = MLPNet(hidden_dim=hidden_dim, n_layers=n_layers).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=0.01 * (1 + 2*T))
    
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
# TEST 1: TESTING THE FACTOR OF 10
# ==============================================================================

def test_factor_of_10():
    """
    Test if the factor 10 relates to network architecture.
    
    Hypothesis: Maybe 10 = 2 × n_layers + something?
    Or related to hidden_dim somehow?
    """
    print("\n" + "="*70)
    print("TEST 1: INVESTIGATING THE FACTOR OF 10")
    print("="*70)
    
    # Test different architectures
    configs = [
        {'n_layers': 1, 'hidden_dim': 32},
        {'n_layers': 2, 'hidden_dim': 32},
        {'n_layers': 2, 'hidden_dim': 64},
        {'n_layers': 3, 'hidden_dim': 64},
        {'n_layers': 4, 'hidden_dim': 64},
        {'n_layers': 2, 'hidden_dim': 128},
    ]
    
    temps = np.linspace(0.02, 0.08, 7)
    n_runs = 15
    epochs = 150
    
    results = {}
    
    for config in configs:
        config_name = f"L{config['n_layers']}_H{config['hidden_dim']}"
        print(f"\n--- {config_name} ---")
        
        chi_values = []
        for T in temps:
            overlaps = []
            for run in range(n_runs):
                m1 = train_network(T, **config, epochs=epochs, seed=run)
                m2 = train_network(T, **config, epochs=epochs, seed=run+1000)
                w1 = m1.get_weights_flat()
                w2 = m2.get_weights_flat()
                q = np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2))
                overlaps.append(q)
            chi_values.append(np.var(overlaps))
        
        max_idx = np.argmax(chi_values)
        T_c_measured = temps[max_idx]
        
        # Compute predicted factor
        # If z_c = T_c × factor × √3, then factor = z_c / (T_c × √3)
        factor = SQRT3_2 / (T_c_measured * SQRT3)
        
        results[config_name] = {
            'T_c': T_c_measured,
            'factor': factor,
            'n_layers': config['n_layers'],
            'hidden_dim': config['hidden_dim'],
        }
        
        print(f"  T_c = {T_c_measured:.4f}")
        print(f"  Factor = z_c / (T_c × √3) = {factor:.2f}")
    
    print("\n" + "-"*50)
    print("FACTOR ANALYSIS")
    print("-"*50)
    print(f"{'Config':<15} | {'T_c':>6} | {'Factor':>8} | {'n_layers':>8} | {'hidden':>8}")
    print("-"*60)
    for name, r in results.items():
        print(f"{name:<15} | {r['T_c']:>6.4f} | {r['factor']:>8.2f} | {r['n_layers']:>8} | {r['hidden_dim']:>8}")
    
    # Check if factor correlates with architecture
    n_layers_list = [r['n_layers'] for r in results.values()]
    factors = [r['factor'] for r in results.values()]
    
    corr, p = stats.pearsonr(n_layers_list, factors)
    print(f"\nCorrelation(n_layers, factor) = {corr:.3f} (p = {p:.3f})")
    
    return results

# ==============================================================================
# TEST 2: COPHENETIC CORRELATION
# ==============================================================================

def test_cophenetic_correlation():
    """
    Test if trained networks have higher tree-like structure than random.
    
    Cophenetic correlation measures how well hierarchical clustering
    represents the original distances.
    """
    print("\n" + "="*70)
    print("TEST 2: COPHENETIC CORRELATION (Tree Structure)")
    print("="*70)
    
    temps = [0.02, 0.04, 0.06]
    n_networks = 20
    
    results = {}
    
    for T in temps:
        print(f"\n--- Temperature T = {T} ---")
        
        # Train networks
        weights = []
        for run in range(n_networks):
            model = train_network(T, epochs=200, seed=run)
            weights.append(model.get_weights_flat())
        
        weights = np.array(weights)
        
        # Compute cophenetic correlation
        dists = pdist(weights)
        Z = linkage(dists, method='average')
        c, _ = cophenet(Z, dists)
        
        results[T] = {'cophenetic': c}
        print(f"  Cophenetic correlation: {c:.4f}")
    
    # Compare to random
    print("\n--- Random high-D points ---")
    random_weights = np.random.randn(n_networks, weights.shape[1])
    dists_rand = pdist(random_weights)
    Z_rand = linkage(dists_rand, method='average')
    c_rand, _ = cophenet(Z_rand, dists_rand)
    print(f"  Cophenetic correlation: {c_rand:.4f}")
    
    results['random'] = {'cophenetic': c_rand}
    
    print("\n" + "-"*50)
    print("COMPARISON")
    print("-"*50)
    
    trained_coph = np.mean([results[T]['cophenetic'] for T in temps])
    print(f"  Trained networks: {trained_coph:.4f}")
    print(f"  Random points:    {c_rand:.4f}")
    
    if trained_coph > c_rand * 1.05:
        print("  ✅ Trained networks have STRONGER tree structure")
    else:
        print("  ⚠️ No significant difference in tree structure")
    
    return results

# ==============================================================================
# TEST 3: INTRINSIC DIMENSION
# ==============================================================================

def test_intrinsic_dimension():
    """
    Estimate intrinsic dimension of trained network manifold.
    
    If trained networks lie on low-D manifold, this would distinguish
    them from random high-D points.
    """
    print("\n" + "="*70)
    print("TEST 3: INTRINSIC DIMENSION ESTIMATION")
    print("="*70)
    
    def estimate_intrinsic_dim(points, k=5):
        """MLE estimator for intrinsic dimension."""
        from scipy.spatial import distance_matrix
        
        n = len(points)
        dists = distance_matrix(points, points)
        
        # For each point, get k nearest neighbors
        dims = []
        for i in range(n):
            sorted_dists = np.sort(dists[i])[1:k+2]  # Exclude self
            if sorted_dists[-1] > 0 and sorted_dists[0] > 0:
                # MLE estimator
                log_ratios = np.log(sorted_dists[-1] / sorted_dists[:-1])
                dim_est = (k - 1) / np.sum(log_ratios)
                dims.append(dim_est)
        
        return np.mean(dims), np.std(dims)
    
    temps = [0.02, 0.04, 0.06]
    n_networks = 30
    
    results = {}
    
    for T in temps:
        print(f"\n--- Temperature T = {T} ---")
        
        weights = []
        for run in range(n_networks):
            model = train_network(T, epochs=200, seed=run)
            weights.append(model.get_weights_flat())
        
        weights = np.array(weights)
        dim_mean, dim_std = estimate_intrinsic_dim(weights)
        
        results[T] = {'dim': dim_mean, 'dim_std': dim_std}
        print(f"  Intrinsic dimension: {dim_mean:.1f} ± {dim_std:.1f}")
        print(f"  Ambient dimension: {weights.shape[1]}")
    
    # Compare to random
    print("\n--- Random high-D points ---")
    random_weights = np.random.randn(n_networks, weights.shape[1])
    dim_rand, dim_rand_std = estimate_intrinsic_dim(random_weights)
    print(f"  Intrinsic dimension: {dim_rand:.1f} ± {dim_rand_std:.1f}")
    
    results['random'] = {'dim': dim_rand, 'dim_std': dim_rand_std}
    
    return results

# ==============================================================================
# TEST 4: SOLUTION DIVERSITY
# ==============================================================================

def test_solution_diversity():
    """
    Do different seeds find genuinely different solutions?
    
    Measure: accuracy on same test set, weight similarity
    """
    print("\n" + "="*70)
    print("TEST 4: SOLUTION DIVERSITY")
    print("="*70)
    
    x_test, y_test = make_modular_addition_data(2000)
    x_test, y_test = x_test.to(DEVICE), y_test.to(DEVICE)
    
    temps = [0.02, 0.04, 0.06]
    n_networks = 20
    
    results = {}
    
    for T in temps:
        print(f"\n--- Temperature T = {T} ---")
        
        weights = []
        accuracies = []
        predictions = []
        
        for run in range(n_networks):
            model = train_network(T, epochs=200, seed=run)
            model.eval()
            
            with torch.no_grad():
                out = model(x_test)
                pred = out.argmax(dim=1)
                acc = (pred == y_test).float().mean().item()
            
            weights.append(model.get_weights_flat())
            accuracies.append(acc)
            predictions.append(pred.cpu().numpy())
        
        # Weight diversity
        weights = np.array(weights)
        weight_dists = pdist(weights)
        
        # Prediction diversity: how often do networks disagree?
        predictions = np.array(predictions)
        disagreement = []
        for i in range(n_networks):
            for j in range(i+1, n_networks):
                disagreement.append(np.mean(predictions[i] != predictions[j]))
        
        results[T] = {
            'mean_acc': np.mean(accuracies),
            'std_acc': np.std(accuracies),
            'mean_weight_dist': np.mean(weight_dists),
            'mean_disagreement': np.mean(disagreement),
        }
        
        print(f"  Accuracy: {100*np.mean(accuracies):.1f}% ± {100*np.std(accuracies):.1f}%")
        print(f"  Mean weight distance: {np.mean(weight_dists):.2f}")
        print(f"  Mean prediction disagreement: {100*np.mean(disagreement):.1f}%")
    
    return results

# ==============================================================================
# TEST 5: GROKKING CHECK
# ==============================================================================

def test_grokking():
    """
    Check if grokking occurs and if T_c relates to it.
    """
    print("\n" + "="*70)
    print("TEST 5: GROKKING INVESTIGATION")
    print("="*70)
    
    temps = [0.02, 0.04, 0.06]
    epochs = 1000  # Long training for grokking
    
    results = {}
    
    for T in temps:
        print(f"\n--- Temperature T = {T} ---")
        
        torch.manual_seed(42)
        np.random.seed(42)
        
        x_train, y_train = make_modular_addition_data(500)  # Small dataset
        x_test, y_test = make_modular_addition_data(1000)
        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=128, shuffle=True)
        
        model = MLPNet(hidden_dim=64).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.1)  # Regularization
        
        train_accs = []
        test_accs = []
        
        for epoch in range(epochs):
            model.train()
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
            
            if epoch % 50 == 0:
                model.eval()
                with torch.no_grad():
                    train_out = model(x_train.to(DEVICE))
                    train_acc = (train_out.argmax(1) == y_train.to(DEVICE)).float().mean().item()
                    test_out = model(x_test.to(DEVICE))
                    test_acc = (test_out.argmax(1) == y_test.to(DEVICE)).float().mean().item()
                
                train_accs.append(train_acc)
                test_accs.append(test_acc)
                
                if epoch % 200 == 0:
                    print(f"  Epoch {epoch}: train={100*train_acc:.1f}%, test={100*test_acc:.1f}%")
        
        # Check for grokking (train acc saturates before test)
        train_saturated = np.argmax(np.array(train_accs) > 0.95) if max(train_accs) > 0.95 else -1
        test_saturated = np.argmax(np.array(test_accs) > 0.95) if max(test_accs) > 0.95 else -1
        
        results[T] = {
            'train_accs': train_accs,
            'test_accs': test_accs,
            'train_saturated_epoch': train_saturated * 50 if train_saturated >= 0 else -1,
            'test_saturated_epoch': test_saturated * 50 if test_saturated >= 0 else -1,
            'grokking_gap': (test_saturated - train_saturated) * 50 if train_saturated >= 0 and test_saturated >= 0 else None,
        }
        
        print(f"  Final: train={100*train_accs[-1]:.1f}%, test={100*test_accs[-1]:.1f}%")
    
    return results

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "#"*70)
    print("# HEAVY TEST SUITE")
    print("#"*70)
    print(f"Device: {DEVICE}")
    print(f"\nKey relationship: z_c = T_c × 10√3")
    
    all_results = {}
    start = time.time()
    
    # Run tests
    tests = [
        ("Factor of 10", test_factor_of_10),
        ("Cophenetic Correlation", test_cophenetic_correlation),
        ("Intrinsic Dimension", test_intrinsic_dimension),
        ("Solution Diversity", test_solution_diversity),
        ("Grokking", test_grokking),
    ]
    
    for name, test_fn in tests:
        print(f"\n{'#'*70}")
        print(f"# {name}")
        print(f"{'#'*70}")
        try:
            result = test_fn()
            all_results[name] = result
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = {'error': str(e)}
    
    # Save results
    def convert(obj):
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(i) for i in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float64, np.float32, np.int64, np.int32)):
            return float(obj) if isinstance(obj, (np.float64, np.float32)) else int(obj)
        else:
            return obj
    
    with open('heavy_test_results.json', 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"COMPLETE - Total runtime: {(time.time()-start)/60:.1f} minutes")
    print(f"Results saved to heavy_test_results.json")
    print(f"{'='*70}")
    
    return all_results

if __name__ == "__main__":
    results = main()
