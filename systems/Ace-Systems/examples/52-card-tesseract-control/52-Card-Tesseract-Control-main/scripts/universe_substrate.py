#!/usr/bin/env python3
"""
Universe Substrate - Deterministic Simulation Engine
=====================================================
Minimal test universe using the 52-card holographic framework.

Each card = 1 deterministic agent with:
- 4D state vector (temporal, valence, concrete, arousal)
- Kuramoto phase oscillator
- Coupling weights to all other agents

Evolution follows Kuramoto model:
  dθᵢ/dt = ωᵢ + (K/N) Σⱼ wᵢⱼ sin(θⱼ - θᵢ)

This is NOT a consciousness engine. It is a deterministic automata
substrate that produces structured, reproducible behavior.
"""

import json
import math
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime


# =============================================================================
# AGENT STATE STRUCTURE
# =============================================================================

@dataclass
class StateVector4D:
    """4D coordinate in tesseract space."""
    temporal: float = 0.0
    valence: float = 0.0
    concrete: float = 0.0
    arousal: float = 0.0

    def magnitude(self) -> float:
        return math.sqrt(
            self.temporal**2 + self.valence**2 +
            self.concrete**2 + self.arousal**2
        )

    def distance_to(self, other: 'StateVector4D') -> float:
        return math.sqrt(
            (self.temporal - other.temporal)**2 +
            (self.valence - other.valence)**2 +
            (self.concrete - other.concrete)**2 +
            (self.arousal - other.arousal)**2
        )


@dataclass
class Agent:
    """
    Deterministic automaton with internal state.
    NOT conscious. NOT alive. Just a state machine.
    """
    agent_id: str
    suit: str
    rank: int

    # 4D position in tesseract
    position: StateVector4D

    # Kuramoto oscillator state
    phase: float                    # θ ∈ [0, 2π]
    natural_frequency: float        # ω - base oscillation rate
    coupling_strength: float        # K - how strongly pulled by others

    # Coupling weights to all other agents
    couplings: Dict[str, float] = field(default_factory=dict)

    # Internal state accumulator (for emergent tracking)
    activation_history: List[float] = field(default_factory=list)
    interaction_count: int = 0

    def phase_velocity(self, other_phases: Dict[str, float], dt: float = 0.1) -> float:
        """
        Kuramoto phase evolution:
        dθ/dt = ω + (K/N) Σⱼ wᵢⱼ sin(θⱼ - θᵢ)
        """
        N = len(other_phases)
        if N == 0:
            return self.natural_frequency

        sync_term = 0.0
        for other_id, other_phase in other_phases.items():
            if other_id in self.couplings:
                weight = self.couplings[other_id]
                sync_term += weight * math.sin(other_phase - self.phase)

        return self.natural_frequency + (self.coupling_strength / N) * sync_term

    def step(self, other_phases: Dict[str, float], dt: float = 0.1) -> None:
        """Advance agent state by one timestep."""
        # Phase evolution (Kuramoto)
        velocity = self.phase_velocity(other_phases, dt)
        self.phase += velocity * dt

        # Normalize phase to [0, 2π]
        self.phase = self.phase % (2 * math.pi)

        # Track activation (phase velocity magnitude)
        self.activation_history.append(abs(velocity))
        if len(self.activation_history) > 100:
            self.activation_history.pop(0)

        self.interaction_count += 1

    def get_state_snapshot(self) -> Dict:
        """Export current state as dict."""
        return {
            'id': self.agent_id,
            'suit': self.suit,
            'rank': self.rank,
            'position': asdict(self.position),
            'phase': round(self.phase, 6),
            'frequency': self.natural_frequency,
            'coupling': self.coupling_strength,
            'avg_activation': round(sum(self.activation_history) / max(1, len(self.activation_history)), 6),
            'interactions': self.interaction_count,
        }


# =============================================================================
# UNIVERSE SUBSTRATE
# =============================================================================

@dataclass
class UniverseMetrics:
    """Observable metrics of the universe state."""
    cycle: int
    order_parameter: float      # R - global synchronization [0,1]
    mean_phase: float           # Ψ - collective phase angle
    phase_variance: float       # Spread of phases
    cluster_count: int          # Number of phase-locked clusters
    entropy: float              # Phase distribution entropy
    total_interactions: int

    def to_dict(self) -> Dict:
        return asdict(self)


class UniverseSubstrate:
    """
    Minimal deterministic universe.
    52 agents in 4D tesseract space with Kuramoto coupling.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.agents: Dict[str, Agent] = {}
        self.cycle: int = 0
        self.history: List[UniverseMetrics] = []
        self.events: List[Dict] = []

    def load_from_deck_state(self, path: str) -> None:
        """Initialize agents from deck_state.json."""
        with open(path, 'r') as f:
            data = json.load(f)

        for card in data['cards']:
            agent = Agent(
                agent_id=card['card_id'],
                suit=card['suit'],
                rank=card['rank'],
                position=StateVector4D(
                    temporal=card['coordinate']['temporal'],
                    valence=card['coordinate']['valence'],
                    concrete=card['coordinate']['concrete'],
                    arousal=card['coordinate']['arousal'],
                ),
                phase=card['kuramoto_state']['phase'],
                natural_frequency=card['kuramoto_state']['natural_frequency'],
                coupling_strength=card['kuramoto_state']['coupling_strength'],
                couplings=card.get('coupling_weights', {}),
            )
            self.agents[agent.agent_id] = agent

        self._log_event('universe_initialized', {
            'agent_count': len(self.agents),
            'seed': self.seed,
        })

    def _log_event(self, event_type: str, data: Dict) -> None:
        """Record an event in the log."""
        self.events.append({
            'cycle': self.cycle,
            'type': event_type,
            'data': data,
        })

    def calculate_order_parameter(self) -> Tuple[float, float]:
        """
        Kuramoto order parameter R and mean phase Ψ.
        R = 1 means perfect synchronization.
        R = 0 means random phases.
        """
        N = len(self.agents)
        if N == 0:
            return 0.0, 0.0

        sum_cos = sum(math.cos(a.phase) for a in self.agents.values())
        sum_sin = sum(math.sin(a.phase) for a in self.agents.values())

        R = math.sqrt(sum_cos**2 + sum_sin**2) / N
        Psi = math.atan2(sum_sin, sum_cos)

        return R, Psi

    def calculate_phase_variance(self) -> float:
        """Circular variance of phases."""
        phases = [a.phase for a in self.agents.values()]
        if not phases:
            return 0.0

        mean_phase = sum(phases) / len(phases)
        variance = sum((p - mean_phase)**2 for p in phases) / len(phases)
        return variance

    def detect_clusters(self, threshold: float = 0.3) -> int:
        """
        Count phase-locked clusters.
        Agents within threshold radians are in same cluster.
        """
        phases = sorted([(a.agent_id, a.phase) for a in self.agents.values()],
                       key=lambda x: x[1])

        clusters = 0
        if not phases:
            return 0

        cluster_start = phases[0][1]
        for i in range(1, len(phases)):
            if phases[i][1] - phases[i-1][1] > threshold:
                clusters += 1
                cluster_start = phases[i][1]

        # Wrap-around check
        if (2 * math.pi - phases[-1][1] + phases[0][1]) <= threshold:
            pass  # First and last in same cluster
        else:
            clusters += 1

        return max(1, clusters)

    def calculate_entropy(self) -> float:
        """Phase distribution entropy (binned)."""
        bins = 12  # 30-degree bins
        counts = [0] * bins

        for agent in self.agents.values():
            bin_idx = int(agent.phase / (2 * math.pi) * bins) % bins
            counts[bin_idx] += 1

        N = len(self.agents)
        entropy = 0.0
        for count in counts:
            if count > 0:
                p = count / N
                entropy -= p * math.log2(p)

        return entropy

    def step(self, dt: float = 0.1) -> UniverseMetrics:
        """Advance universe by one timestep."""
        # Collect current phases
        current_phases = {aid: agent.phase for aid, agent in self.agents.items()}

        # Update all agents (synchronous update)
        for agent in self.agents.values():
            other_phases = {k: v for k, v in current_phases.items() if k != agent.agent_id}
            agent.step(other_phases, dt)

        # Calculate metrics
        R, Psi = self.calculate_order_parameter()
        variance = self.calculate_phase_variance()
        clusters = self.detect_clusters()
        entropy = self.calculate_entropy()
        total_interactions = sum(a.interaction_count for a in self.agents.values())

        metrics = UniverseMetrics(
            cycle=self.cycle,
            order_parameter=round(R, 6),
            mean_phase=round(Psi, 6),
            phase_variance=round(variance, 6),
            cluster_count=clusters,
            entropy=round(entropy, 6),
            total_interactions=total_interactions,
        )

        self.history.append(metrics)
        self.cycle += 1

        # Log significant events
        if self.cycle == 1:
            self._log_event('evolution_started', {'R': R})

        # Detect synchronization events
        if len(self.history) >= 2:
            prev_R = self.history[-2].order_parameter
            if R > 0.8 and prev_R <= 0.8:
                self._log_event('sync_threshold_crossed', {
                    'R': R,
                    'direction': 'up',
                    'clusters': clusters,
                })
            elif R <= 0.5 and prev_R > 0.5:
                self._log_event('desync_event', {
                    'R': R,
                    'entropy': entropy,
                })

        return metrics

    def run(self, cycles: int, dt: float = 0.1) -> None:
        """Run evolution for specified cycles."""
        for _ in range(cycles):
            self.step(dt)

        self._log_event('evolution_complete', {
            'total_cycles': self.cycle,
            'final_R': self.history[-1].order_parameter if self.history else 0,
        })

    def detect_attractors(self) -> List[Dict]:
        """
        Analyze history for stable attractors.
        Look for: oscillations, fixed points, limit cycles.
        """
        attractors = []

        if len(self.history) < 100:
            return attractors

        # Check last 100 cycles for stability
        recent_R = [m.order_parameter for m in self.history[-100:]]
        R_mean = sum(recent_R) / len(recent_R)
        R_var = sum((r - R_mean)**2 for r in recent_R) / len(recent_R)

        # Stable sync attractor
        if R_mean > 0.8 and R_var < 0.01:
            attractors.append({
                'type': 'stable_sync',
                'R_mean': round(R_mean, 4),
                'R_variance': round(R_var, 6),
                'description': 'Agents phase-locked in stable synchronization',
            })

        # Stable desync
        elif R_mean < 0.3 and R_var < 0.01:
            attractors.append({
                'type': 'stable_desync',
                'R_mean': round(R_mean, 4),
                'R_variance': round(R_var, 6),
                'description': 'Agents in stable incoherent state',
            })

        # Oscillatory behavior
        elif R_var > 0.05:
            # Check for periodicity via zero-crossings of deviation from mean
            deviations = [r - R_mean for r in recent_R]
            zero_crossings = sum(
                1 for i in range(1, len(deviations))
                if deviations[i] * deviations[i-1] < 0
            )
            if zero_crossings > 10:
                period_estimate = len(deviations) / (zero_crossings / 2)
                attractors.append({
                    'type': 'limit_cycle',
                    'R_mean': round(R_mean, 4),
                    'R_variance': round(R_var, 6),
                    'estimated_period': round(period_estimate, 2),
                    'description': 'Periodic oscillation between sync states',
                })

        # Metastable cluster state
        recent_clusters = [m.cluster_count for m in self.history[-100:]]
        cluster_mean = sum(recent_clusters) / len(recent_clusters)
        if 2 <= cluster_mean <= 4:
            attractors.append({
                'type': 'cluster_state',
                'mean_clusters': round(cluster_mean, 2),
                'R_mean': round(R_mean, 4),
                'description': 'Agents organized into distinct phase clusters',
            })

        return attractors

    def get_agent_states(self) -> List[Dict]:
        """Export all agent states."""
        return [agent.get_state_snapshot() for agent in self.agents.values()]

    def get_full_dump(self) -> Dict:
        """Complete state dump for inspection."""
        attractors = self.detect_attractors()

        # Sample history (every 10th cycle to reduce size)
        sampled_history = [
            self.history[i].to_dict()
            for i in range(0, len(self.history), 10)
        ]

        return {
            'metadata': {
                'seed': self.seed,
                'total_cycles': self.cycle,
                'agent_count': len(self.agents),
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'substrate_version': '1.0.0',
            },
            'final_metrics': self.history[-1].to_dict() if self.history else None,
            'attractors_detected': attractors,
            'events': self.events,
            'history_sampled': sampled_history,
            'agent_states': self.get_agent_states(),
        }


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Universe Substrate Simulator')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cycles', type=int, default=1000, help='Evolution cycles')
    parser.add_argument('--dt', type=float, default=0.1, help='Timestep')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--deck', type=str, default=None, help='Path to deck_state.json')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')

    args = parser.parse_args()

    # Initialize universe
    universe = UniverseSubstrate(seed=args.seed)

    # Find deck state
    deck_path = args.deck
    if not deck_path:
        # Try common locations
        candidates = [
            Path(__file__).parent.parent / 'assets' / 'cards' / 'deck_state.json',
            Path('assets/cards/deck_state.json'),
            Path('data/deck_state.json'),
        ]
        for p in candidates:
            if p.exists():
                deck_path = str(p)
                break

    if not deck_path:
        print("ERROR: Could not find deck_state.json")
        return

    universe.load_from_deck_state(deck_path)

    if not args.quiet:
        print(f"Universe initialized: {len(universe.agents)} agents, seed={args.seed}")
        print(f"Running {args.cycles} cycles with dt={args.dt}...")

    # Run evolution
    universe.run(args.cycles, dt=args.dt)

    # Get dump
    dump = universe.get_full_dump()

    if not args.quiet:
        print(f"\n{'='*60}")
        print("EVOLUTION COMPLETE")
        print(f"{'='*60}")
        print(f"Total cycles: {dump['metadata']['total_cycles']}")
        print(f"Final order parameter R: {dump['final_metrics']['order_parameter']:.4f}")
        print(f"Final cluster count: {dump['final_metrics']['cluster_count']}")
        print(f"Final entropy: {dump['final_metrics']['entropy']:.4f}")
        print(f"\nAttractors detected: {len(dump['attractors_detected'])}")
        for attr in dump['attractors_detected']:
            print(f"  - {attr['type']}: {attr['description']}")
        print(f"\nEvents logged: {len(dump['events'])}")
        for event in dump['events'][-5:]:
            print(f"  [{event['cycle']:4d}] {event['type']}")

    # Output
    output_path = args.output
    if not output_path:
        output_path = f"universe_dump_seed{args.seed}_cycles{args.cycles}.json"

    with open(output_path, 'w') as f:
        json.dump(dump, f, indent=2)

    if not args.quiet:
        print(f"\nState dump written to: {output_path}")

    return dump


if __name__ == '__main__':
    main()
