"""
agentic_orchestrator.py -- Real-time Agentic Orchestrator with Reasoning and Memory
===================================================================================
A cognitive orchestrator that handles incoming multi-modal tasks, routes them to
the correct model pipeline, reasons about client status, and manages client
selection/skipping using memory.

It integrates:
  - Phase 3: ModalityRouter (routes task files)
  - Phase 4: Offload Decision Module (determines local vs edge training)
  - Phase 5: Client Scorer & Anomaly Filter (scores and protects global models)
  - Local LLM Connection (Ollama/llama3 for cognitive "thinking" decisions, with rule-based fallback)
  - Persistent Memory: Saves execution history and metrics to `results/orchestrator_memory.json`

Usage:
    # Run the orchestrator on a mixed folder or individual file
    python agentic_orchestrator.py --task ./sample_files/test_video.mp4
"""

import os
import sys
import json
import time
import requests
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, deque

# Import modules from our package structure
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase3_router.modality_router import ModalityRouter
from phase4_offload.offload_decision import should_offload, get_decision_details
from phase5_agentic.client_scorer import ClientScorer

# Configuration
MEMORY_FILE = os.path.join("results", "orchestrator_memory.json")
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class RealtimeAgenticOrchestrator:
    """
    Stateful Agentic Orchestrator that routes tasks, reasons about client states,
    and updates global FL memory.
    """

    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name
        self.router = ModalityRouter()
        self.memory = self._load_memory()

        # Initialize scorers for each modality
        self.scorers = {
            "image": ClientScorer(num_clients=50, fraction_fit=0.6),
            "text":  ClientScorer(num_clients=50, fraction_fit=0.6),
            "audio": ClientScorer(num_clients=50, fraction_fit=0.6),
            "video": ClientScorer(num_clients=50, fraction_fit=0.6),
        }

        # Sync scorers with historical memory if it exists
        self._sync_scorers_with_memory()

        # Check LLM Availability
        self.llm_available = self._check_llm_health()

    def _check_llm_health(self) -> bool:
        """Check if local Ollama server is running."""
        try:
            r = requests.get(OLLAMA_URL, timeout=1.0)
            if r.status_code == 200:
                print(f"[*] Connected to Ollama API at {OLLAMA_URL}")
                return True
        except Exception:
            pass
        print("[!] Local Ollama not detected at localhost:11434. Falling back to Cognitive Rule Engine.")
        return False

    def _load_memory(self) -> Dict[str, Any]:
        """Load state memory from disk."""
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading memory: {e}")

        # Default empty memory structure
        return {
            "tasks_processed": 0,
            "modality_stats": {
                "image": {"tasks_count": 0, "avg_accuracy": 0.10, "avg_loss": 2.3},
                "text":  {"tasks_count": 0, "avg_accuracy": 0.25, "avg_loss": 1.3},
                "audio": {"tasks_count": 0, "avg_accuracy": 0.10, "avg_loss": 2.3},
                "video": {"tasks_count": 0, "avg_accuracy": 0.20, "avg_loss": 1.6},
            },
            "history": [],  # List of dicts representing past rounds/decisions
            "client_states": {}  # Maps client_id -> metrics
        }

    def _save_memory(self):
        """Save current memory state to disk."""
        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
        try:
            with open(MEMORY_FILE, "w") as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            print(f"Error saving memory: {e}")

    def _sync_scorers_with_memory(self):
        """Populate ClientScorers with historical metrics from memory."""
        client_states = self.memory.get("client_states", {})
        for mod, scorer in self.scorers.items():
            for cid_str, info in client_states.items():
                if info.get("modality") == mod:
                    cid = int(cid_str)
                    scorer.update_client_history(
                        client_id=cid,
                        current_round=self.memory["tasks_processed"] + 1,
                        loss=info.get("loss", 1.0),
                        update_norm=info.get("update_norm", 1.0),
                        accuracy=info.get("accuracy", 0.5),
                        resource_state=info.get("resource_state", {}),
                    )

    def process_task(self, task_file: str, client_resource_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process an incoming task file, classify modality, think/reason about clients,
        make selection & offload decisions, and update memory.
        """
        # Step 1: Modality Routing
        modality = self.router.route(task_file)
        if not modality:
            print(f"[!] Unknown file format for task: {task_file}. Cannot process.")
            return {"status": "error", "message": "Unknown file format"}

        print(f"\n[+] Task Routed: File '{os.path.basename(task_file)}' mapped to modality: {modality.upper()}")

        # Update stats
        self.memory["tasks_processed"] += 1
        self.memory["modality_stats"][modality]["tasks_count"] += 1
        current_round = self.memory["tasks_processed"]

        # Step 2: Client Scoring & Selection
        scorer = self.scorers[modality]

        # Register incoming client resources in scorer
        for state in client_resource_states:
            cid = state["client_id"]
            scorer.resource_states[cid] = state

        # Get candidates (Top 60% by scoring algorithm)
        candidate_ids, all_scores = scorer.select_clients(current_round)

        # Step 3: LLM Cognitive Thinking & Skipping
        print("[*] Agentic AI is thinking about client selection and offloading...")
        decisions = self._think_and_decide(
            modality=modality,
            task_file=task_file,
            candidates=candidate_ids,
            client_states={s["client_id"]: s for s in client_resource_states},
            all_scores=all_scores,
        )

        # Step 4: Offload Decision Calculation
        final_selected = []
        offload_actions = {}
        for cid in candidate_ids:
            # Check if skipped by LLM/reasoning
            if decisions.get(cid, {}).get("skip", False):
                print(f"   [-] Skipped Client {cid}: {decisions[cid].get('reason', 'Resource limit')}")
                continue

            final_selected.append(cid)
            client_state = next(s for s in client_resource_states if s["client_id"] == cid)
            client_state["modality"] = modality
            # Fetch rule-based / cost-function offload recommendation
            action = should_offload(client_state)
            offload_actions[cid] = action
            print(f"   [+] Selected Client {cid} -> Execution: {action.upper()}")

        # Update persistent memory stats
        for state in client_resource_states:
            cid = state["client_id"]
            cid_str = str(cid)
            if cid_str not in self.memory["client_states"]:
                self.memory["client_states"][cid_str] = {}

            self.memory["client_states"][cid_str].update({
                "modality": modality,
                "resource_state": state,
                "last_selected_round": current_round if cid in final_selected else self.memory["client_states"][cid_str].get("last_selected_round", 0)
            })

        # Save record of this decision round
        round_record = {
            "task_id": current_round,
            "task_file": task_file,
            "modality": modality,
            "selected_clients": final_selected,
            "offload_decisions": offload_actions,
            "decisions_details": decisions,
        }
        self.memory["history"].append(round_record)
        self._save_memory()

        return {
            "status": "success",
            "modality": modality,
            "selected_clients": final_selected,
            "offload_decisions": offload_actions,
            "decisions_details": decisions,
        }

    def _think_and_decide(
        self,
        modality: str,
        task_file: str,
        candidates: List[int],
        client_states: Dict[int, Dict],
        all_scores: List[Tuple[int, float]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Cognitive reasoning loop. Queries Ollama/LLM if available,
        otherwise uses a deterministic rule-based heuristic.
        """
        # If LLM is available, perform prompt-based reasoning
        if self.llm_available:
            try:
                prompt = self._build_reasoning_prompt(modality, task_file, candidates, client_states, all_scores)
                response = requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json"
                    },
                    timeout=5.0
                )
                if response.status_code == 200:
                    raw_resp = response.json().get("response", "{}")
                    parsed = json.loads(raw_resp)
                    # Convert string keys back to int
                    decisions = {}
                    for k, v in parsed.items():
                        decisions[int(k)] = v
                    return decisions
            except Exception as e:
                print(f"   (LLM query failed or timed out: {e}. Falling back to Rule Engine.)")

        # Heuristic/Rule-based Fallback (Think Engine)
        decisions = {}
        for cid in candidates:
            state = client_states.get(cid, {})
            battery = state.get("battery_pct", 100)
            cpu = state.get("device_cpu_ghz", 1.5)
            latency = state.get("network_latency_ms", 50)
            temp = state.get("temperature", 40)

            skip = False
            reason = "Optimal condition"

            # Cognitive Skipping logic:
            if battery < 20:
                skip = True
                reason = "Battery critically low"
            elif temp > 80:
                skip = True
                reason = "Device thermal throttling"
            elif latency > 180 and battery < 30:
                skip = True
                reason = "Poor connectivity combined with low battery"

            decisions[cid] = {"skip": skip, "reason": reason}

        return decisions

    def _build_reasoning_prompt(
        self,
        modality: str,
        task_file: str,
        candidates: List[int],
        client_states: Dict[int, Dict],
        all_scores: List[Tuple[int, float]]
    ) -> str:
        """Constructs a structured prompt detailing task modality and client stats."""
        stats = self.memory["modality_stats"][modality]

        client_data_summary = []
        for cid in candidates:
            state = client_states.get(cid, {})
            score = next((s[1] for s in all_scores if s[0] == cid), 0.0)
            client_data_summary.append(
                f"  Client {cid}: score={score:.2f}, battery={state.get('battery_pct')}%,"
                f" CPU={state.get('device_cpu_ghz')}GHz, latency={state.get('network_latency_ms')}ms"
            )

        client_lines = "\n".join(client_data_summary)

        return f"""You are the orchestrator for a multi-modal federated learning system.
A new task has arrived:
  - File: {os.path.basename(task_file)}
  - Modality: {modality}
  - Historical modality accuracy: {stats['avg_accuracy']*100:.1f}%

Here is the status of candidate clients:
{client_lines}

Decide which clients to include or skip. If a client has extremely low battery (<20%) or bad latency, they should be skipped to prevent slowing down the system.

Respond ONLY with a JSON object where keys are client IDs (strings) and values contain "skip" (boolean) and "reason" (string). Example format:
{{
  "0": {{"skip": false, "reason": "Good battery and link"}},
  "1": {{"skip": true, "reason": "Low battery"}}
}}"""


# --- Self-Test Main Loop ---
if __name__ == "__main__":
    # Create sample task files
    os.makedirs("sample_files", exist_ok=True)
    sample_files = [
        "sample_files/camera_input.jpg",
        "sample_files/command_voice.wav",
        "sample_files/chat_message.txt",
        "sample_files/security_clip.mp4"
    ]
    for sf in sample_files:
        with open(sf, "w") as f:
            f.write("dummy test data")

    # Initialize the orchestrator
    orchestrator = RealtimeAgenticOrchestrator()

    # Simulate resource state updates for 10 clients
    import random
    random.seed(42)
    simulated_resources = []
    for i in range(50):
        simulated_resources.append({
            "client_id": i,
            "device_cpu_ghz": round(random.uniform(0.8, 2.5), 2),
            "battery_pct": round(random.uniform(10, 100), 1),
            "network_latency_ms": round(random.uniform(5, 200), 1),
            "privacy_flag": random.random() < 0.10,
            "temperature": round(random.uniform(30, 85), 1),
        })

    # Run orchestrator on a video task
    result = orchestrator.process_task("sample_files/security_clip.mp4", simulated_resources)

    # Output decisions
    print("\n" + "="*50)
    print("  ORCHESTRATOR DECISION LOG")
    print("="*50)
    print(f"Modality Selected: {result['modality'].upper()}")
    print(f"Total Selected Clients: {len(result['selected_clients'])}")
    print("Decisions sample:")
    for cid in list(result["offload_decisions"].keys())[:5]:
        details = result["decisions_details"].get(cid, {})
        print(f"  Client {cid:>2}: Offload = {result['offload_decisions'][cid].upper()} | Skipped = {details.get('skip')} (Reason: {details.get('reason')})")
    print("="*50 + "\n")
