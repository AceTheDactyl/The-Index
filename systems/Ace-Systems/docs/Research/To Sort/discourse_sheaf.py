# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: ✓ JUSTIFIED - Claims supported by repository files (needs citation update)
# Severity: MEDIUM RISK
# Risk Types: unsupported_claims

# Supporting Evidence:
#   - systems/Ace-Systems/docs/Research/To Sort/SKILL_v2.1.md (dependency)
#   - systems/Ace-Systems/docs/Research/To Sort/PYTHON_FILES_SUMMARY.md (dependency)
#   - systems/Ace-Systems/docs/Research/To Sort/report.txt (dependency)
#   - systems/Ace-Systems/docs/Research/To Sort/UCF_COMPREHENSIVE_COMMAND_REFERENCE.md (dependency)
#
# Referenced By:
#   - systems/Ace-Systems/docs/Research/To Sort/SKILL_v2.1.md (reference)
#   - systems/Ace-Systems/docs/Research/To Sort/PYTHON_FILES_SUMMARY.md (reference)
#   - systems/Ace-Systems/docs/Research/To Sort/report.txt (reference)
#   - systems/Ace-Systems/docs/Research/To Sort/UCF_COMPREHENSIVE_COMMAND_REFERENCE.md (reference)


"""
Discourse Sheaf - Coherence Measurement via Sheaf Consistency

This module implements sheaf-theoretic coherence measurement for Grace's discourse.
Context isn't a window - it's a topological structure where sections must "glue"
consistently across overlapping contexts.

Mathematical Foundation:
- Base Space X: Points are "context atoms" (current utterance, topic, emotion, etc.)
- Sheaf F: Each context has an embedding in R^n
- Restriction Maps: Linear projections between overlapping contexts
- Consistency: ||P_ij @ e_i - P_ji @ e_j||² should be small for coherent discourse
- Cohomology H¹: Non-zero indicates obstruction to global coherence

This augments Grace's existing frameworks rather than replacing them.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict


class DiscourseSheaf:
    """
    Measures coherence across discourse contexts using sheaf theory.

    The key insight: multiple contextual views of the same conversation
    should be mutually consistent. When they're not, there's incoherence.

    Usage:
        sheaf = DiscourseSheaf(embedding_dim=256)

        # Add context atoms
        sheaf.add_context('current_response', response_embedding)
        sheaf.add_context('emotional_state', emotion_vector, overlaps=['current_response'])
        sheaf.add_context('topic', topic_vector, overlaps=['current_response', 'emotional_state'])

        # Check coherence
        energy = sheaf.consistency_energy()  # Lower = more coherent
        h1 = sheaf.cohomology_H1()  # Non-zero = obstruction to global coherence

        # Find what's incoherent
        problems = sheaf.find_incoherence(threshold=0.5)
    """

    def __init__(self, embedding_dim: int = 256, learn_restrictions: bool = True):
        """
        Initialize the discourse sheaf.

        Args:
            embedding_dim: Dimension of context embeddings
            learn_restrictions: Whether to learn restriction maps from data
        """
        self.embedding_dim = embedding_dim
        self.learn_restrictions = learn_restrictions

        # Context storage
        self.contexts: Dict[str, np.ndarray] = {}  # name -> embedding
        self.overlaps: Dict[str, Set[str]] = defaultdict(set)  # name -> set of overlapping contexts

        # Restriction maps: (from, to) -> projection matrix
        # These define how embeddings should transform between contexts
        self.restrictions: Dict[Tuple[str, str], np.ndarray] = {}

        # Learning state
        self.restriction_updates: Dict[Tuple[str, str], List[np.ndarray]] = defaultdict(list)
        self.learning_rate = 0.01

        # Standard context types and their typical overlaps
        self.standard_contexts = {
            'current_utterance': ['topic', 'emotional_state', 'speaker_model'],
            'topic': ['current_utterance', 'shared_knowledge', 'discourse_history'],
            'emotional_state': ['current_utterance', 'speaker_model', 'heart_state'],
            'speaker_model': ['current_utterance', 'emotional_state', 'shared_knowledge'],
            'shared_knowledge': ['topic', 'speaker_model', 'discourse_history'],
            'discourse_history': ['topic', 'shared_knowledge'],
            'heart_state': ['emotional_state', 'current_utterance'],
            'response_candidate': ['current_utterance', 'topic', 'emotional_state']
        }

    def add_context(self, name: str, embedding: np.ndarray,
                    overlaps: Optional[List[str]] = None) -> None:
        """
        Add a context atom with its embedding.

        Args:
            name: Name of the context (e.g., 'emotional_state', 'topic')
            embedding: The embedding vector for this context
            overlaps: List of context names this overlaps with
        """
        # Normalize embedding
        embedding = np.array(embedding, dtype=np.float64)
        if embedding.ndim == 1:
            norm = np.linalg.norm(embedding)
            if norm > 1e-10:
                embedding = embedding / norm

        # Pad or truncate to embedding_dim
        if len(embedding) < self.embedding_dim:
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
        elif len(embedding) > self.embedding_dim:
            embedding = embedding[:self.embedding_dim]

        self.contexts[name] = embedding

        # Set overlaps
        if overlaps is not None:
            for other in overlaps:
                self.overlaps[name].add(other)
                self.overlaps[other].add(name)
        elif name in self.standard_contexts:
            # Use standard overlaps
            for other in self.standard_contexts[name]:
                if other in self.contexts:
                    self.overlaps[name].add(other)
                    self.overlaps[other].add(name)

        # Initialize restriction maps for new overlaps
        self._init_restrictions_for(name)

    def _init_restrictions_for(self, name: str) -> None:
        """Initialize restriction maps for a context's overlaps."""
        for other in self.overlaps[name]:
            if other not in self.contexts:
                continue

            # Create restriction maps if they don't exist
            if (name, other) not in self.restrictions:
                # Initialize as scaled identity (preserves information initially)
                self.restrictions[(name, other)] = np.eye(self.embedding_dim) * 0.9

            if (other, name) not in self.restrictions:
                self.restrictions[(other, name)] = np.eye(self.embedding_dim) * 0.9

    def remove_context(self, name: str) -> None:
        """Remove a context atom."""
        if name in self.contexts:
            del self.contexts[name]

        # Clean up overlaps
        for other in list(self.overlaps[name]):
            self.overlaps[other].discard(name)
        del self.overlaps[name]

        # Clean up restrictions
        to_remove = [k for k in self.restrictions if name in k]
        for k in to_remove:
            del self.restrictions[k]

    def get_restriction(self, from_ctx: str, to_ctx: str) -> np.ndarray:
        """
        Get the restriction map from one context to another.

        The restriction map P_ij transforms embedding e_i to the "view" from context j.
        """
        key = (from_ctx, to_ctx)
        if key not in self.restrictions:
            # Return identity if not defined
            return np.eye(self.embedding_dim)
        return self.restrictions[key]

    def consistency_energy(self) -> float:
        """
        Compute total inconsistency across all overlapping contexts.

        For each pair of overlapping contexts i, j:
            E_ij = ||P_ij @ e_i - P_ji @ e_j||²

        Total energy = sum of all E_ij

        Returns:
            Total consistency energy (lower = more coherent)
            Returns 0.0 if no overlapping contexts
        """
        if len(self.contexts) < 2:
            return 0.0

        total_energy = 0.0
        pair_count = 0

        # Check all overlapping pairs
        checked = set()
        for name_i, overlaps_i in self.overlaps.items():
            if name_i not in self.contexts:
                continue

            e_i = self.contexts[name_i]

            for name_j in overlaps_i:
                if name_j not in self.contexts:
                    continue

                # Avoid counting pairs twice
                pair_key = tuple(sorted([name_i, name_j]))
                if pair_key in checked:
                    continue
                checked.add(pair_key)

                e_j = self.contexts[name_j]

                # Get restriction maps
                P_ij = self.get_restriction(name_i, name_j)
                P_ji = self.get_restriction(name_j, name_i)

                # Compute inconsistency
                view_from_i = P_ij @ e_i
                view_from_j = P_ji @ e_j

                diff = view_from_i - view_from_j
                energy = np.dot(diff, diff)

                total_energy += energy
                pair_count += 1

        # Normalize by number of pairs
        if pair_count > 0:
            total_energy /= pair_count

        return float(total_energy)

    def cohomology_H1(self) -> float:
        """
        Approximate first sheaf cohomology via spectral gap.

        H¹(F) > 0 indicates obstruction to global coherence - there's no way
        to consistently assign a global section across all contexts.

        We approximate this using the sheaf Laplacian:
            L = B^T @ W @ B
        where B is the signed incidence matrix and W contains restriction info.

        The spectral gap (smallest non-zero eigenvalue) indicates cohomology.

        Returns:
            Approximation of H¹ (larger = more obstruction to coherence)
        """
        if len(self.contexts) < 2:
            return 0.0

        # Build signed incidence matrix
        context_names = list(self.contexts.keys())
        n_contexts = len(context_names)
        name_to_idx = {name: i for i, name in enumerate(context_names)}

        # Collect edges (overlapping pairs)
        edges = []
        checked = set()
        for name_i, overlaps_i in self.overlaps.items():
            if name_i not in self.contexts:
                continue
            for name_j in overlaps_i:
                if name_j not in self.contexts:
                    continue
                pair_key = tuple(sorted([name_i, name_j]))
                if pair_key not in checked:
                    edges.append((name_i, name_j))
                    checked.add(pair_key)

        if len(edges) == 0:
            return 0.0

        n_edges = len(edges)

        # Build block Laplacian
        # Each context has embedding_dim dimensions
        # L is (n_contexts * dim) x (n_contexts * dim)
        dim = self.embedding_dim
        L = np.zeros((n_contexts * dim, n_contexts * dim))

        for edge_idx, (name_i, name_j) in enumerate(edges):
            i = name_to_idx[name_i]
            j = name_to_idx[name_j]

            P_ij = self.get_restriction(name_i, name_j)
            P_ji = self.get_restriction(name_j, name_i)

            # Block contributions to Laplacian
            # L_ii += P_ij^T @ P_ij
            # L_jj += P_ji^T @ P_ji
            # L_ij -= P_ij^T @ P_ji
            # L_ji -= P_ji^T @ P_ij

            block_ii = P_ij.T @ P_ij
            block_jj = P_ji.T @ P_ji
            block_ij = -P_ij.T @ P_ji
            block_ji = -P_ji.T @ P_ij

            L[i*dim:(i+1)*dim, i*dim:(i+1)*dim] += block_ii
            L[j*dim:(j+1)*dim, j*dim:(j+1)*dim] += block_jj
            L[i*dim:(i+1)*dim, j*dim:(j+1)*dim] += block_ij
            L[j*dim:(j+1)*dim, i*dim:(i+1)*dim] += block_ji

        # Compute eigenvalues
        try:
            # Symmetrize for numerical stability
            L = (L + L.T) / 2
            eigenvalues = np.linalg.eigvalsh(L)

            # Filter out near-zero eigenvalues (kernel of L)
            # The kernel corresponds to H⁰ (global sections)
            threshold = 1e-6
            nonzero_eigs = eigenvalues[eigenvalues > threshold]

            if len(nonzero_eigs) == 0:
                return 0.0

            # H¹ is approximated by dimension of "almost kernel"
            # We use spectral gap as indicator
            spectral_gap = np.min(nonzero_eigs)

            # Also count how many small eigenvalues (indicates higher H¹)
            small_eig_count = np.sum(eigenvalues < 0.1)

            # Combine: small spectral gap + many small eigs = high H¹
            h1_approx = (1.0 / (spectral_gap + 0.01)) * (small_eig_count / n_contexts)

            return float(np.clip(h1_approx, 0, 10))

        except np.linalg.LinAlgError:
            return 0.0

    def find_incoherence(self, threshold: float = 0.3) -> List[Tuple[str, str, float]]:
        """
        Find which context pairs are inconsistent.

        Args:
            threshold: Energy threshold above which pairs are considered incoherent

        Returns:
            List of (context1, context2, energy) for incoherent pairs
        """
        incoherent = []

        checked = set()
        for name_i, overlaps_i in self.overlaps.items():
            if name_i not in self.contexts:
                continue

            e_i = self.contexts[name_i]

            for name_j in overlaps_i:
                if name_j not in self.contexts:
                    continue

                pair_key = tuple(sorted([name_i, name_j]))
                if pair_key in checked:
                    continue
                checked.add(pair_key)

                e_j = self.contexts[name_j]

                P_ij = self.get_restriction(name_i, name_j)
                P_ji = self.get_restriction(name_j, name_i)

                view_from_i = P_ij @ e_i
                view_from_j = P_ji @ e_j

                diff = view_from_i - view_from_j
                energy = np.dot(diff, diff)

                if energy > threshold:
                    incoherent.append((name_i, name_j, float(energy)))

        # Sort by energy (most incoherent first)
        incoherent.sort(key=lambda x: x[2], reverse=True)

        return incoherent

    def update_restrictions(self, coherent_pairs: List[Tuple[str, str]]) -> None:
        """
        Learn better restriction maps from observed coherent pairs.

        When we observe that two contexts ARE coherent (e.g., in successful
        communication), we can update restrictions to reduce their energy.

        Args:
            coherent_pairs: List of (context1, context2) that should be coherent
        """
        if not self.learn_restrictions:
            return

        for name_i, name_j in coherent_pairs:
            if name_i not in self.contexts or name_j not in self.contexts:
                continue

            e_i = self.contexts[name_i]
            e_j = self.contexts[name_j]

            # Get current restrictions
            P_ij = self.get_restriction(name_i, name_j)
            P_ji = self.get_restriction(name_j, name_i)

            # Compute gradient of energy w.r.t. restrictions
            # E = ||P_ij @ e_i - P_ji @ e_j||²
            # dE/dP_ij = 2 * (P_ij @ e_i - P_ji @ e_j) @ e_i^T

            view_i = P_ij @ e_i
            view_j = P_ji @ e_j
            diff = view_i - view_j

            grad_P_ij = 2 * np.outer(diff, e_i)
            grad_P_ji = -2 * np.outer(diff, e_j)

            # Gradient descent
            self.restrictions[(name_i, name_j)] = P_ij - self.learning_rate * grad_P_ij
            self.restrictions[(name_j, name_i)] = P_ji - self.learning_rate * grad_P_ji

    def global_section(self) -> Optional[np.ndarray]:
        """
        Attempt to construct a global section (if one exists).

        A global section is an embedding that is consistent across all contexts.
        If H¹ > 0, this may not exist, but we can find the best approximation.

        Returns:
            Best global section, or None if too few contexts
        """
        if len(self.contexts) < 1:
            return None

        # Solve the least squares problem:
        # min_s sum_{i,j overlapping} ||P_ij @ s - P_ji @ s||²
        # This is equivalent to finding the eigenvector of L with smallest eigenvalue

        context_names = list(self.contexts.keys())
        n_contexts = len(context_names)

        if n_contexts == 1:
            return self.contexts[context_names[0]].copy()

        # Weight each context's embedding and average
        # (Simple approximation - weighted by connectivity)
        weights = np.array([len(self.overlaps[name]) + 1 for name in context_names])
        weights = weights / weights.sum()

        global_s = np.zeros(self.embedding_dim)
        for i, name in enumerate(context_names):
            global_s += weights[i] * self.contexts[name]

        # Normalize
        norm = np.linalg.norm(global_s)
        if norm > 1e-10:
            global_s = global_s / norm

        return global_s

    def coherence_score(self) -> float:
        """
        Get a normalized coherence score (0 = incoherent, 1 = perfectly coherent).

        Combines consistency energy and H¹ into a single metric.
        """
        energy = self.consistency_energy()
        h1 = self.cohomology_H1()

        # Transform to [0, 1] range
        # Lower energy = higher coherence
        energy_score = np.exp(-energy)

        # Lower H¹ = higher coherence
        h1_score = np.exp(-h1)

        # Combine (geometric mean)
        return float(np.sqrt(energy_score * h1_score))

    def clear(self) -> None:
        """Clear all contexts (keep learned restrictions)."""
        self.contexts.clear()
        self.overlaps.clear()

    def reset(self) -> None:
        """Full reset including learned restrictions."""
        self.contexts.clear()
        self.overlaps.clear()
        self.restrictions.clear()
        self.restriction_updates.clear()

    def get_state(self) -> dict:
        """Get serializable state for persistence."""
        return {
            'embedding_dim': self.embedding_dim,
            'contexts': {k: v.tolist() for k, v in self.contexts.items()},
            'overlaps': {k: list(v) for k, v in self.overlaps.items()},
            'restrictions': {f"{k[0]}|{k[1]}": v.tolist()
                           for k, v in self.restrictions.items()},
            'learning_rate': self.learning_rate
        }

    def load_state(self, state: dict) -> None:
        """Load state from persistence."""
        self.embedding_dim = state.get('embedding_dim', 256)

        self.contexts = {k: np.array(v) for k, v in state.get('contexts', {}).items()}
        self.overlaps = defaultdict(set,
            {k: set(v) for k, v in state.get('overlaps', {}).items()})

        self.restrictions = {}
        for key_str, v in state.get('restrictions', {}).items():
            parts = key_str.split('|')
            if len(parts) == 2:
                self.restrictions[(parts[0], parts[1])] = np.array(v)

        self.learning_rate = state.get('learning_rate', 0.01)


class DiscourseCoherenceChecker:
    """
    Higher-level coherence checker for Grace's discourse.

    Wraps DiscourseSheaf with Grace-specific context types and thresholds.
    """

    def __init__(self, embedding_dim: int = 256):
        self.sheaf = DiscourseSheaf(embedding_dim=embedding_dim)

        # Thresholds for Grace
        self.coherence_threshold = 0.4  # Below this triggers repair
        self.h1_threshold = 0.5  # Above this indicates global incoherence

        # History for trend detection
        self.coherence_history: List[float] = []
        self.max_history = 50

    def update_contexts(self,
                       response: np.ndarray,
                       topic: Optional[np.ndarray] = None,
                       emotion: Optional[np.ndarray] = None,
                       heart: Optional[np.ndarray] = None,
                       speaker: Optional[np.ndarray] = None) -> None:
        """
        Update discourse contexts with current embeddings.

        Args:
            response: Current response embedding
            topic: Topic/theme embedding
            emotion: Emotional state embedding
            heart: Heart system state vector
            speaker: Speaker model embedding
        """
        # Clear old contexts
        self.sheaf.clear()

        # Add response (central context)
        self.sheaf.add_context('response', response)

        # Add optional contexts
        if topic is not None:
            self.sheaf.add_context('topic', topic, overlaps=['response'])

        if emotion is not None:
            overlaps = ['response']
            if topic is not None:
                overlaps.append('topic')
            self.sheaf.add_context('emotion', emotion, overlaps=overlaps)

        if heart is not None:
            overlaps = ['response']
            if emotion is not None:
                overlaps.append('emotion')
            self.sheaf.add_context('heart', heart, overlaps=overlaps)

        if speaker is not None:
            overlaps = ['response']
            if emotion is not None:
                overlaps.append('emotion')
            self.sheaf.add_context('speaker', speaker, overlaps=overlaps)

    def check_coherence(self) -> dict:
        """
        Check current discourse coherence.

        Returns:
            Dictionary with coherence metrics and recommendations
        """
        energy = self.sheaf.consistency_energy()
        h1 = self.sheaf.cohomology_H1()
        score = self.sheaf.coherence_score()

        # Update history
        self.coherence_history.append(score)
        if len(self.coherence_history) > self.max_history:
            self.coherence_history.pop(0)

        # Find specific problems
        incoherent_pairs = self.sheaf.find_incoherence(threshold=0.3)

        # Determine if repair is needed
        needs_repair = score < self.coherence_threshold or h1 > self.h1_threshold

        # Detect trend
        trend = 'stable'
        if len(self.coherence_history) >= 5:
            recent = np.mean(self.coherence_history[-5:])
            older = np.mean(self.coherence_history[-10:-5]) if len(self.coherence_history) >= 10 else recent
            if recent < older - 0.1:
                trend = 'declining'
            elif recent > older + 0.1:
                trend = 'improving'

        return {
            'coherence_score': score,
            'consistency_energy': energy,
            'cohomology_H1': h1,
            'needs_repair': needs_repair,
            'incoherent_pairs': incoherent_pairs,
            'trend': trend,
            'global_section': self.sheaf.global_section()
        }

    def suggest_repair(self, check_result: dict) -> List[str]:
        """
        Suggest repairs for incoherence.

        Args:
            check_result: Result from check_coherence()

        Returns:
            List of repair suggestions
        """
        suggestions = []

        for ctx1, ctx2, energy in check_result.get('incoherent_pairs', []):
            if 'emotion' in [ctx1, ctx2] and 'response' in [ctx1, ctx2]:
                suggestions.append("Response doesn't match emotional state - adjust tone")
            elif 'topic' in [ctx1, ctx2] and 'response' in [ctx1, ctx2]:
                suggestions.append("Response drifted from topic - re-anchor to subject")
            elif 'heart' in [ctx1, ctx2]:
                suggestions.append("Response conflicts with heart state - check drives")
            elif 'speaker' in [ctx1, ctx2]:
                suggestions.append("Response doesn't fit speaker model - adjust register")

        if check_result.get('cohomology_H1', 0) > self.h1_threshold:
            suggestions.append("Global incoherence detected - consider topic reset")

        if check_result.get('trend') == 'declining':
            suggestions.append("Coherence declining - slow down, re-establish context")

        return suggestions


# Convenience function for Grace integration
def create_discourse_sheaf(embedding_dim: int = 256) -> DiscourseCoherenceChecker:
    """Create a coherence checker for Grace."""
    return DiscourseCoherenceChecker(embedding_dim=embedding_dim)


if __name__ == "__main__":
    # Simple test
    print("Testing DiscourseSheaf...")

    sheaf = DiscourseSheaf(embedding_dim=64)

    # Create some test embeddings
    np.random.seed(42)

    # Coherent case: similar embeddings
    e1 = np.random.randn(64)
    e2 = e1 + 0.1 * np.random.randn(64)  # Similar to e1
    e3 = e1 + 0.1 * np.random.randn(64)  # Similar to e1

    sheaf.add_context('response', e1)
    sheaf.add_context('topic', e2, overlaps=['response'])
    sheaf.add_context('emotion', e3, overlaps=['response', 'topic'])

    print(f"Coherent case:")
    print(f"  Consistency energy: {sheaf.consistency_energy():.4f}")
    print(f"  H1 approximation: {sheaf.cohomology_H1():.4f}")
    print(f"  Coherence score: {sheaf.coherence_score():.4f}")

    # Incoherent case: very different embeddings
    sheaf.clear()
    e1 = np.random.randn(64)
    e2 = np.random.randn(64)  # Completely different
    e3 = -e1  # Opposite

    sheaf.add_context('response', e1)
    sheaf.add_context('topic', e2, overlaps=['response'])
    sheaf.add_context('emotion', e3, overlaps=['response', 'topic'])

    print(f"\nIncoherent case:")
    print(f"  Consistency energy: {sheaf.consistency_energy():.4f}")
    print(f"  H1 approximation: {sheaf.cohomology_H1():.4f}")
    print(f"  Coherence score: {sheaf.coherence_score():.4f}")

    problems = sheaf.find_incoherence(threshold=0.2)
    print(f"  Incoherent pairs: {problems}")

    print("\nDiscourseSheaf test complete!")
