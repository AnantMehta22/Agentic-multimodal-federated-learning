"""
offload_decision.py -- Computation Offload Decision Module
==========================================================
Decides whether a client should train locally on-device or offload
to an edge server, based on a cost function comparing:
  - Model FLOPs vs. device capability
  - Battery level
  - Network latency to edge
  - Data privacy flag

Usage:
    from phase4_offload.offload_decision import should_offload

    decision = should_offload({
        "modality": "video",
        "device_cpu_ghz": 1.2,
        "battery_pct": 30,
        "network_latency_ms": 50,
        "privacy_flag": False,
    })
    # -> "edge"
"""

from typing import Dict, Any


# Estimated model FLOPs per modality (forward + backward pass)
MODEL_FLOPS = {
    "image": 68_000_000,    # 68M FLOPs  - light
    "text":  45_000_000,    # 45M FLOPs  - lightest
    "audio": 120_000_000,   # 120M FLOPs - medium
    "video": 560_000_000,   # 560M FLOPs - heaviest (8 frames x forward)
}

# Edge server capability (simulated)
EDGE_CPU_GHZ = 4.0          # Edge server CPU speed
EDGE_FLOPS_PER_CYCLE = 4    # FLOPs per clock cycle (edge has better pipeline)

# Device capability
DEVICE_FLOPS_PER_CYCLE = 2  # FLOPs per clock cycle (mobile device)

# Thresholds
BATTERY_CRITICAL = 15       # Below this, strongly prefer offload
LATENCY_THRESHOLD = 200     # Above this ms, offloading is too expensive


def should_offload(client_state: Dict[str, Any]) -> str:
    """
    Decide whether a client should train locally or offload to edge.

    Args:
        client_state: Dict with keys:
            - modality (str):          "image", "text", "audio", or "video"
            - device_cpu_ghz (float):  Device CPU speed in GHz (0.8 - 2.5)
            - battery_pct (float):     Battery level 0-100
            - network_latency_ms (float): Network latency to edge in ms
            - privacy_flag (bool):     If True, data must stay on device

    Returns:
        "local" or "edge"
    """
    modality = client_state["modality"]
    device_cpu = client_state.get("device_cpu_ghz", 1.5)
    battery = client_state.get("battery_pct", 50)
    latency = client_state.get("network_latency_ms", 50)
    privacy = client_state.get("privacy_flag", False)

    flops = MODEL_FLOPS.get(modality, 100_000_000)

    # Rule 1: If privacy flag is set, MUST stay local
    if privacy:
        return "local"

    # Rule 2: If network is too slow, stay local
    if latency > LATENCY_THRESHOLD:
        return "local"

    # Cost of local training
    device_speed = device_cpu * 1e9 * DEVICE_FLOPS_PER_CYCLE  # FLOPs/second
    time_local_ms = (flops / device_speed) * 1000

    # Battery drain penalty: low battery makes local more expensive
    battery_factor = 1.0 + max(0, (50 - battery)) / 50.0  # 1.0 to 2.0
    cost_local = time_local_ms * battery_factor

    # Cost of edge training
    edge_speed = EDGE_CPU_GHZ * 1e9 * EDGE_FLOPS_PER_CYCLE
    time_edge_ms = (flops / edge_speed) * 1000

    # Edge cost includes: computation time + 2x network latency (send + receive)
    # + data transfer overhead proportional to model size
    transfer_overhead_ms = flops / 1e8  # Rough proxy for model data transfer
    cost_edge = time_edge_ms + 2 * latency + transfer_overhead_ms

    # Decision
    if cost_edge < cost_local:
        return "edge"
    else:
        return "local"


def get_decision_details(client_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Same as should_offload() but returns full cost breakdown for logging.

    Returns dict with decision and all intermediate cost values.
    """
    modality = client_state["modality"]
    device_cpu = client_state.get("device_cpu_ghz", 1.5)
    battery = client_state.get("battery_pct", 50)
    latency = client_state.get("network_latency_ms", 50)
    privacy = client_state.get("privacy_flag", False)

    flops = MODEL_FLOPS.get(modality, 100_000_000)

    details = {
        "modality": modality,
        "device_cpu_ghz": device_cpu,
        "battery_pct": battery,
        "network_latency_ms": latency,
        "privacy_flag": privacy,
        "model_flops": flops,
    }

    if privacy:
        details["decision"] = "local"
        details["reason"] = "privacy_flag"
        details["cost_local"] = 0
        details["cost_edge"] = 0
        return details

    if latency > LATENCY_THRESHOLD:
        details["decision"] = "local"
        details["reason"] = "high_latency"
        details["cost_local"] = 0
        details["cost_edge"] = float("inf")
        return details

    # Compute costs
    device_speed = device_cpu * 1e9 * DEVICE_FLOPS_PER_CYCLE
    time_local_ms = (flops / device_speed) * 1000
    battery_factor = 1.0 + max(0, (50 - battery)) / 50.0
    cost_local = time_local_ms * battery_factor

    edge_speed = EDGE_CPU_GHZ * 1e9 * EDGE_FLOPS_PER_CYCLE
    time_edge_ms = (flops / edge_speed) * 1000
    transfer_overhead_ms = flops / 1e8
    cost_edge = time_edge_ms + 2 * latency + transfer_overhead_ms

    details["time_local_ms"] = round(time_local_ms, 2)
    details["battery_factor"] = round(battery_factor, 3)
    details["cost_local"] = round(cost_local, 2)
    details["time_edge_ms"] = round(time_edge_ms, 2)
    details["transfer_overhead_ms"] = round(transfer_overhead_ms, 2)
    details["cost_edge"] = round(cost_edge, 2)
    details["decision"] = "edge" if cost_edge < cost_local else "local"
    details["reason"] = "cost_comparison"

    return details


if __name__ == "__main__":
    # Quick test with different modalities and conditions
    print(f"\n{'='*70}")
    print(f"  OFFLOAD DECISION MODULE -- Quick Test")
    print(f"{'='*70}")

    test_cases = [
        {"modality": "image", "device_cpu_ghz": 1.5, "battery_pct": 80, "network_latency_ms": 30, "privacy_flag": False},
        {"modality": "text",  "device_cpu_ghz": 1.5, "battery_pct": 80, "network_latency_ms": 30, "privacy_flag": False},
        {"modality": "audio", "device_cpu_ghz": 1.0, "battery_pct": 20, "network_latency_ms": 30, "privacy_flag": False},
        {"modality": "video", "device_cpu_ghz": 0.8, "battery_pct": 15, "network_latency_ms": 30, "privacy_flag": False},
        {"modality": "video", "device_cpu_ghz": 0.8, "battery_pct": 15, "network_latency_ms": 30, "privacy_flag": True},
        {"modality": "video", "device_cpu_ghz": 2.5, "battery_pct": 95, "network_latency_ms": 30, "privacy_flag": False},
    ]

    for state in test_cases:
        details = get_decision_details(state)
        decision = details["decision"]
        mod = state["modality"]
        print(f"\n  {mod:>6} | CPU={state['device_cpu_ghz']}GHz | Batt={state['battery_pct']}% | "
              f"Lat={state['network_latency_ms']}ms | Priv={state['privacy_flag']}")
        print(f"         -> {decision.upper()} (reason: {details['reason']}, "
              f"cost_local={details.get('cost_local', 'N/A')}, cost_edge={details.get('cost_edge', 'N/A')})")

    print(f"\n{'='*70}\n")
