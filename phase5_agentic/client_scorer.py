"""
client_scorer.py -- Per-Round Client Scoring for Agentic Selection
=================================================================
Scores each client based on:
  - Recent loss (30%):      Lower loss = higher score
  - Update norm (20%):      Moderate norms preferred; extremes penalized
  - Staleness (20%):        Rounds since last participation; slight boost
  - Resource state (15%):   Battery + CPU availability composite
  - Data quality (15%):     Local validation accuracy

Also includes cosine-similarity-based anomaly detection to flag
potentially poisoned or straggling client updates.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class ClientScorer:
    """Scores and selects clients for each FL round."""

    def __init__(
        self,
        num_clients: int,
        fraction_fit: float = 0.6,
        cosine_sim_threshold: float = -0.2,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            num_clients:          Total number of clients.
            fraction_fit:         Fraction of clients to select each round.
            cosine_sim_threshold: Threshold for anomaly detection.
            weights:              Scoring weights for each feature.
        """
        self.num_clients = num_clients
        self.fraction_fit = fraction_fit
        self.top_k = max(1, int(num_clients * fraction_fit))
        self.cosine_sim_threshold = cosine_sim_threshold

        self.weights = weights or {
            "loss": 0.30,
            "update_norm": 0.20,
            "staleness": 0.20,
            "resource": 0.15,
            "data_quality": 0.15,
        }

        # Client history tracking
        self.last_loss: Dict[int, float] = {}
        self.last_update_norm: Dict[int, float] = {}
        self.last_participation_round: Dict[int, int] = {}
        self.last_accuracy: Dict[int, float] = {}
        self.resource_states: Dict[int, Dict] = {}

        # Running average update for cosine similarity check
        self.running_avg_update: Optional[List[np.ndarray]] = None
        self.update_count = 0

    def update_client_history(
        self,
        client_id: int,
        current_round: int,
        loss: float,
        update_norm: float,
        accuracy: float = 0.0,
        resource_state: Optional[Dict] = None,
    ):
        """Record a client's metrics after participation."""
        self.last_loss[client_id] = loss
        self.last_update_norm[client_id] = update_norm
        self.last_participation_round[client_id] = current_round
        self.last_accuracy[client_id] = accuracy
        if resource_state:
            self.resource_states[client_id] = resource_state

    def score_client(self, client_id: int, current_round: int) -> float:
        """
        Compute composite score for a single client.
        Higher score = more desirable for selection.
        """
        scores = {}

        # 1. Loss score: lower loss -> higher score (inverse)
        loss = self.last_loss.get(client_id, 2.0)  # Default high loss for new clients
        # Normalize: map [0, 5] -> [1, 0]
        scores["loss"] = max(0, 1.0 - loss / 5.0)

        # 2. Update norm score: prefer moderate norms
        norm = self.last_update_norm.get(client_id, 1.0)
        # Bell curve around ideal norm of 1.0
        scores["update_norm"] = np.exp(-0.5 * ((norm - 1.0) / 2.0) ** 2)

        # 3. Staleness score: boost stale clients (haven't participated recently)
        last_round = self.last_participation_round.get(client_id, 0)
        staleness = current_round - last_round
        # Moderate staleness is good (diversity), extreme is risky
        scores["staleness"] = min(1.0, staleness / 5.0) * 0.7 + 0.3

        # 4. Resource score: battery and CPU availability
        resource = self.resource_states.get(client_id, {})
        battery = resource.get("battery_pct", 50) / 100.0
        cpu_avail = resource.get("device_cpu_ghz", 1.5) / 2.5
        scores["resource"] = 0.6 * battery + 0.4 * cpu_avail

        # 5. Data quality: local validation accuracy
        acc = self.last_accuracy.get(client_id, 0.5)
        scores["data_quality"] = acc

        # Weighted composite
        composite = sum(self.weights[k] * scores[k] for k in self.weights)
        return composite

    def select_clients(
        self,
        current_round: int,
        available_clients: Optional[List[int]] = None,
    ) -> Tuple[List[int], List[Tuple[int, float]]]:
        """
        Select top-K clients by composite score.

        Returns:
            (selected_ids, all_scores): List of selected client IDs and
            all (client_id, score) tuples for logging.
        """
        if available_clients is None:
            available_clients = list(range(self.num_clients))

        # Score all available clients
        client_scores = [
            (cid, self.score_client(cid, current_round))
            for cid in available_clients
        ]

        # Sort by score (descending)
        client_scores.sort(key=lambda x: x[1], reverse=True)

        # Select top-K
        selected = [cid for cid, _ in client_scores[:self.top_k]]

        return selected, client_scores

    def check_anomaly(
        self,
        client_update: List[np.ndarray],
    ) -> Tuple[bool, float]:
        """
        Check if a client's update is anomalous compared to the running average.

        Returns:
            (is_anomalous, cosine_similarity)
        """
        if self.running_avg_update is None:
            return False, 1.0

        # Use only the last layer (dense classifier output) for cosine similarity
        # to prevent sparse embedding layers (e.g. in TextLSTM) from skewing the similarity to 0.
        flat_client = client_update[-1].flatten()
        flat_avg = self.running_avg_update[-1].flatten()

        # Cosine similarity
        dot = np.dot(flat_client, flat_avg)
        norm_c = np.linalg.norm(flat_client)
        norm_a = np.linalg.norm(flat_avg)

        if norm_c < 1e-10 or norm_a < 1e-10:
            return True, 0.0

        cos_sim = float(dot / (norm_c * norm_a))
        is_anomalous = cos_sim < self.cosine_sim_threshold

        return is_anomalous, cos_sim

    def update_running_average(self, client_update: List[np.ndarray]):
        """Update the running average of parameter updates."""
        if self.running_avg_update is None:
            self.running_avg_update = [u.copy() for u in client_update]
            self.update_count = 1
        else:
            self.update_count += 1
            alpha = 1.0 / self.update_count
            for i in range(len(self.running_avg_update)):
                self.running_avg_update[i] = (
                    (1 - alpha) * self.running_avg_update[i] + alpha * client_update[i]
                )


class RandomSelector:
    """Simple random client selection baseline for comparison."""

    def __init__(self, num_clients: int, fraction_fit: float = 0.6, seed: int = 42):
        self.num_clients = num_clients
        self.fraction_fit = fraction_fit
        self.top_k = max(1, int(num_clients * fraction_fit))
        self.rng = np.random.default_rng(seed)

    def select_clients(
        self,
        current_round: int,
        available_clients: Optional[List[int]] = None,
    ) -> List[int]:
        """Randomly select K clients."""
        if available_clients is None:
            available_clients = list(range(self.num_clients))
        return list(self.rng.choice(
            available_clients, size=min(self.top_k, len(available_clients)), replace=False
        ))


if __name__ == "__main__":
    # Quick test
    print(f"\n{'='*60}")
    print(f"  CLIENT SCORER -- Quick Test")
    print(f"{'='*60}")

    scorer = ClientScorer(num_clients=10, fraction_fit=0.5)

    # Simulate some history
    rng = np.random.default_rng(42)
    for cid in range(10):
        scorer.update_client_history(
            client_id=cid,
            current_round=1,
            loss=rng.uniform(0.1, 3.0),
            update_norm=rng.uniform(0.1, 5.0),
            accuracy=rng.uniform(0.3, 0.95),
            resource_state={
                "battery_pct": rng.uniform(10, 100),
                "device_cpu_ghz": rng.uniform(0.8, 2.5),
            }
        )

    selected, scores = scorer.select_clients(current_round=3)
    print(f"\n  All client scores:")
    for cid, score in scores:
        marker = " <-- SELECTED" if cid in selected else ""
        print(f"    Client {cid:>2}: score={score:.4f}{marker}")
    print(f"\n  Selected {len(selected)} clients: {selected}")
    print(f"{'='*60}\n")
