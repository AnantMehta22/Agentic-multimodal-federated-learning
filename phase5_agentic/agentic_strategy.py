"""
agentic_strategy.py -- AgenticFedAvg Strategy
==============================================
Custom FL strategy that integrates:
  - Smart client selection from client_scorer.py
  - Offload decisions from phase4_offload
  - Cosine-similarity-based outlier filtering during aggregation

This replaces the default "all clients participate" strategy from Phase 2.
"""

import os
import sys
import copy
import numpy as np
from typing import List, Tuple, Dict, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from phase2_fl_sim.fl_utils import (
    get_parameters, set_parameters, fedavg_aggregate,
    local_train, local_evaluate,
    compute_parameter_delta, compute_update_norm, compute_update_cosine_similarity,
)
from phase5_agentic.client_scorer import ClientScorer
from phase4_offload.offload_decision import should_offload


class AgenticFedAvg:
    """
    Agentic FedAvg strategy with smart client selection,
    offload decisions, and outlier-filtered aggregation.
    """

    def __init__(
        self,
        num_clients: int = 50,
        fraction_fit: float = 0.6,
        cosine_sim_threshold: float = -0.2,
    ):
        self.num_clients = num_clients
        self.fraction_fit = fraction_fit
        self.scorer = ClientScorer(
            num_clients=num_clients,
            fraction_fit=fraction_fit,
            cosine_sim_threshold=cosine_sim_threshold,
        )
        self.round_logs = []

    def configure_fit(
        self,
        current_round: int,
        client_resource_states: Optional[Dict[int, Dict]] = None,
    ) -> Tuple[List[int], Dict[int, str]]:
        """
        Configure the training round:
        1. Score and select clients
        2. Determine offload decisions for selected clients

        Returns:
            (selected_client_ids, offload_decisions)
            offload_decisions: {client_id: "local" or "edge"}
        """
        # Step 1: Score and select clients
        selected, all_scores = self.scorer.select_clients(current_round)

        # Step 2: Determine offload decisions
        offload_decisions = {}
        if client_resource_states:
            for cid in selected:
                state = client_resource_states.get(cid, {})
                decision = should_offload(state)
                offload_decisions[cid] = decision
        else:
            for cid in selected:
                offload_decisions[cid] = "local"

        return selected, offload_decisions

    def aggregate_fit(
        self,
        current_round: int,
        results: List[Tuple[int, List[np.ndarray], int, Dict]],
        global_params: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Aggregate client updates with outlier filtering.

        Args:
            current_round: Current FL round number.
            results: List of (client_id, updated_params, num_samples, metrics).
            global_params: Parameters before this round's training.

        Returns:
            (aggregated_params, aggregation_info)
        """
        # Compute deltas and check for anomalies
        valid_results = []
        flagged_clients = []

        for cid, params, n_samples, metrics in results:
            delta = compute_parameter_delta(global_params, params)
            norm = compute_update_norm(global_params, params)

            # Update running average
            is_anomalous, cos_sim = self.scorer.check_anomaly(delta)

            if is_anomalous and current_round > 2:
                # Skip anomalous updates (but only after we have enough history)
                flagged_clients.append((cid, cos_sim))
            else:
                valid_results.append((params, n_samples))
                self.scorer.update_running_average(delta)

            # Update client history
            self.scorer.update_client_history(
                client_id=cid,
                current_round=current_round,
                loss=metrics.get("loss", 2.0),
                update_norm=norm,
                accuracy=metrics.get("accuracy", 0.5),
                resource_state=metrics.get("resource_state", {}),
            )

        # FedAvg on valid results
        if valid_results:
            aggregated = fedavg_aggregate(valid_results)
        else:
            # Fallback: use all results if all were flagged
            aggregated = fedavg_aggregate([(p, n) for _, p, n, _ in results])

        info = {
            "num_aggregated": len(valid_results),
            "num_flagged": len(flagged_clients),
            "flagged_clients": flagged_clients,
        }

        # Log round info
        self.round_logs.append({
            "round": current_round,
            "num_selected": len(results),
            "num_aggregated": len(valid_results),
            "num_flagged": len(flagged_clients),
        })

        return aggregated, info
