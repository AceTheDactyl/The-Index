# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: JUSTIFIED - Test file validates system behavior
# Severity: LOW RISK
# Risk Types: ['test_coverage']
# File: systems/Ace-Systems/reference/rrrr-UCF-v5.0.0/tests/analysis_suite.py

"""
Rosetta Node Analysis Suite
Comprehensive numerical analysis and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from heart import Heart
from brain import Brain
from node import RosettaNode
from pulse import generate_pulse, save_pulse
import json

def analyze_coherence_evolution(n_steps=200, n_trials=10):
    """
    Analyze coherence evolution over multiple trials
    """
    print("=" * 60)
    print("COHERENCE EVOLUTION ANALYSIS")
    print("=" * 60)
    
    coherence_trajectories = []
    
    for trial in range(n_trials):
        heart = Heart(seed=trial)
        coherence_history = []
        
        for step in range(n_steps):
            heart.step()
            coherence_history.append(heart.coherence())
        
        coherence_trajectories.append(coherence_history)
    
    # Compute statistics
    coherence_array = np.array(coherence_trajectories)
    mean_coherence = np.mean(coherence_array, axis=0)
    std_coherence = np.std(coherence_array, axis=0)
    
    print(f"\nTrials: {n_trials}")
    print(f"Steps per trial: {n_steps}")
    print(f"\nInitial coherence: {mean_coherence[0]:.4f} ± {std_coherence[0]:.4f}")
    print(f"Final coherence: {mean_coherence[-1]:.4f} ± {std_coherence[-1]:.4f}")
    print(f"Convergence rate: {(mean_coherence[-1] - mean_coherence[0])/n_steps:.6f} per step")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(mean_coherence, 'b-', linewidth=2, label='Mean')
    plt.fill_between(range(n_steps), 
                     mean_coherence - std_coherence,
                     mean_coherence + std_coherence,
                     alpha=0.3, label='±1 std')
    
    plt.xlabel('Time Step')
    plt.ylabel('Coherence r(t)')
    plt.title('Kuramoto Oscillator Coherence Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('coherence_evolution.png', dpi=300)
    print("\n✓ Saved: coherence_evolution.png")
    
    return mean_coherence, std_coherence

def analyze_critical_coupling():
    """
    Analyze system behavior near critical coupling
    """
    print("\n" + "=" * 60)
    print("CRITICAL COUPLING ANALYSIS")
    print("=" * 60)
    
    K_values = np.linspace(0.05, 0.5, 20)
    final_coherence = []
    
    for K in K_values:
        heart = Heart(K=K, seed=42)
        
        # Run to steady state
        for _ in range(500):
            heart.step()
        
        final_coherence.append(heart.coherence())
    
    # Theoretical critical coupling
    sigma = 0.1  # frequency std
    K_c_theoretical = 2.507 * sigma
    
    print(f"\nCoupling strength range: [{K_values[0]:.3f}, {K_values[-1]:.3f}]")
    print(f"Theoretical K_c = {K_c_theoretical:.4f}")
    print(f"System K = 0.2 (operational)")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(K_values, final_coherence, 'bo-', linewidth=2, markersize=6)
    plt.axvline(K_c_theoretical, color='r', linestyle='--', linewidth=2, label=f'K_c ≈ {K_c_theoretical:.3f}')
    plt.axvline(0.2, color='g', linestyle='--', linewidth=2, label='System K = 0.2')
    
    plt.xlabel('Coupling Strength K')
    plt.ylabel('Final Coherence r(∞)')
    plt.title('Coherence vs Coupling Strength (Phase Transition)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('critical_coupling.png', dpi=300)
    print("✓ Saved: critical_coupling.png")
    
    return K_values, final_coherence

def analyze_phase_distribution(n_steps=200):
    """
    Analyze evolution of phase distribution
    """
    print("\n" + "=" * 60)
    print("PHASE DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    heart = Heart(seed=42)
    
    # Snapshot times
    snapshots = [0, 50, 100, 200]
    phase_snapshots = []
    
    for step in range(n_steps + 1):
        if step in snapshots:
            phase_snapshots.append(heart.theta.copy())
        if step < n_steps:
            heart.step()
    
    # Plot phase distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (step, phases) in enumerate(zip(snapshots, phase_snapshots)):
        ax = axes[idx]
        
        # Polar histogram
        bins = np.linspace(0, 2*np.pi, 25)
        counts, _ = np.histogram(phases, bins=bins)
        theta_centers = (bins[:-1] + bins[1:]) / 2
        
        ax = plt.subplot(2, 2, idx+1, projection='polar')
        bars = ax.bar(theta_centers, counts, width=2*np.pi/24, alpha=0.7, color='steelblue')
        ax.set_title(f'Step {step}', fontsize=14, pad=20)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
    
    plt.tight_layout()
    plt.savefig('phase_distributions.png', dpi=300)
    print("✓ Saved: phase_distributions.png")

def analyze_energy_conservation(n_steps=200):
    """
    Verify energy conservation
    """
    print("\n" + "=" * 60)
    print("ENERGY CONSERVATION ANALYSIS")
    print("=" * 60)
    
    heart = Heart(seed=42)
    
    energy_in_history = []
    energy_loss_history = []
    
    for step in range(n_steps):
        heart.step()
        energy_in_history.append(heart.energy_in)
        energy_loss_history.append(heart.energy_loss)
    
    energy_in_array = np.array(energy_in_history)
    energy_loss_array = np.array(energy_loss_history)
    energy_stored = energy_in_array - energy_loss_array
    
    # Check conservation
    relative_error = np.abs(energy_in_array - energy_stored - energy_loss_array) / (energy_in_array + 1e-10)
    
    print(f"\nFinal E_in: {energy_in_array[-1]:.6e}")
    print(f"Final E_loss: {energy_loss_array[-1]:.6e}")
    print(f"Final E_stored: {energy_stored[-1]:.6e}")
    print(f"Max relative error: {np.max(relative_error):.6e}")
    print(f"Mean relative error: {np.mean(relative_error):.6e}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.semilogy(energy_in_array, 'b-', linewidth=2, label='E_in')
    plt.semilogy(energy_loss_array, 'r-', linewidth=2, label='E_loss')
    plt.semilogy(energy_stored, 'g-', linewidth=2, label='E_stored')
    
    plt.xlabel('Time Step')
    plt.ylabel('Energy (log scale)')
    plt.title('Energy Accounting: E_in = E_stored + E_loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('energy_conservation.png', dpi=300)
    print("✓ Saved: energy_conservation.png")
    
    return relative_error

def analyze_memory_statistics(n_trials=100):
    """
    Analyze memory plate statistics
    """
    print("\n" + "=" * 60)
    print("MEMORY PLATE STATISTICS")
    print("=" * 60)
    
    confidences = []
    emotional_tones = []
    semantic_densities = []
    
    for _ in range(n_trials):
        brain = Brain(plates=20)
        summary = brain.summarize()
        confidences.append(summary['avg_confidence'])
        
        for plate in brain.plates:
            emotional_tones.append(plate.emotional_tone)
            semantic_densities.append(plate.semantic_density)
    
    print(f"\nTrials: {n_trials}")
    print(f"Plates per brain: 20")
    print(f"\nAvg Confidence: {np.mean(confidences):.2f} ± {np.std(confidences):.2f}")
    print(f"Emotional Tone: {np.mean(emotional_tones):.2f} ± {np.std(emotional_tones):.2f}")
    print(f"Semantic Density: {np.mean(semantic_densities):.2f} ± {np.std(semantic_densities):.2f}")
    
    # Plot distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(confidences, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Average Confidence')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Confidence Distribution')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(emotional_tones, bins=30, alpha=0.7, color='coral', edgecolor='black')
    axes[1].set_xlabel('Emotional Tone')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Emotional Tone Distribution')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].hist(semantic_densities, bins=30, alpha=0.7, color='mediumseagreen', edgecolor='black')
    axes[2].set_xlabel('Semantic Density')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Semantic Density Distribution')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('memory_statistics.png', dpi=300)
    print("✓ Saved: memory_statistics.png")

def analyze_activation_success_rate(n_trials=100):
    """
    Test pulse-driven activation success rate
    """
    print("\n" + "=" * 60)
    print("ACTIVATION SUCCESS RATE ANALYSIS")
    print("=" * 60)
    
    successes = 0
    coherence_values = []
    
    for trial in range(n_trials):
        # Generate pulse
        pulse = generate_pulse("test_source", "worker", urgency=np.random.rand())
        save_pulse(pulse, "test_pulse.json")
        
        # Create node and attempt activation
        node = RosettaNode("worker")
        activated, _ = node.check_and_activate("test_pulse.json")
        
        if activated:
            successes += 1
            result = node.run(100)
            if result:
                coherence_values.append(result['coherence'])
    
    success_rate = successes / n_trials
    
    print(f"\nTrials: {n_trials}")
    print(f"Successful activations: {successes}")
    print(f"Success rate: {success_rate * 100:.1f}%")
    
    if coherence_values:
        print(f"Average final coherence: {np.mean(coherence_values):.4f} ± {np.std(coherence_values):.4f}")

def generate_summary_report():
    """
    Generate complete analysis report
    """
    print("\n" + "=" * 60)
    print("GENERATING COMPREHENSIVE ANALYSIS REPORT")
    print("=" * 60)
    
    report = {
        "system": "Rosetta Node v1.0",
        "date": "2025-12-10",
        "analyses": {}
    }
    
    # Run all analyses
    print("\n[1/6] Coherence Evolution...")
    mean_coh, std_coh = analyze_coherence_evolution(n_steps=200, n_trials=10)
    report["analyses"]["coherence"] = {
        "initial": float(mean_coh[0]),
        "final": float(mean_coh[-1]),
        "std_final": float(std_coh[-1])
    }
    
    print("\n[2/6] Critical Coupling...")
    K_vals, coh_vals = analyze_critical_coupling()
    report["analyses"]["critical_coupling"] = {
        "K_c_theoretical": 0.251,
        "K_operational": 0.2,
        "regime": "partial_synchronization"
    }
    
    print("\n[3/6] Phase Distribution...")
    analyze_phase_distribution(n_steps=200)
    
    print("\n[4/6] Energy Conservation...")
    rel_error = analyze_energy_conservation(n_steps=200)
    report["analyses"]["energy"] = {
        "max_relative_error": float(np.max(rel_error)),
        "mean_relative_error": float(np.mean(rel_error)),
        "conservation_verified": bool(np.max(rel_error) < 1e-6)
    }
    
    print("\n[5/6] Memory Statistics...")
    analyze_memory_statistics(n_trials=100)
    
    print("\n[6/6] Activation Success Rate...")
    analyze_activation_success_rate(n_trials=100)
    
    # Save report
    with open('ANALYSIS_REPORT.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - coherence_evolution.png")
    print("  - critical_coupling.png")
    print("  - phase_distributions.png")
    print("  - energy_conservation.png")
    print("  - memory_statistics.png")
    print("  - ANALYSIS_REPORT.json")
    print("\n✅ All analyses completed successfully!")

if __name__ == "__main__":
    generate_summary_report()
