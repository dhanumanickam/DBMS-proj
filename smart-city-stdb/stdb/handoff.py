"""Mobile Device Location & Cellular/V2X Handoff Management Engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .spatial_ops import haversine_m


@dataclass
class TowerState:
    tower_id: str
    name: str
    lat: float
    lon: float
    radius_m: float
    tx_power_dbm: float
    capacity: int
    zone: str
    connected_count: int = 0


@dataclass
class HandoffDecision:
    device_id: str
    vehicle_id: str
    from_tower_id: Optional[str]
    to_tower_id: str
    rssi_from_dbm: Optional[float]
    rssi_to_dbm: float
    handoff_occurred: bool
    latency_ms: float
    reason: str


HYSTERESIS_MARGIN_DB = 4.5
MIN_USABLE_RSSI_DBM = -95.0


def compute_rssi_dbm(
    lat: float,
    lon: float,
    tower: TowerState,
    load_penalty: bool = True,
) -> float:
    """Estimate Received Signal Strength (dBm) using urban log-distance path loss."""
    dist_m = max(12.0, haversine_m(lat, lon, tower.lat, tower.lon))
    # Urban micro/V2X path loss model: PL(d) = 42 + 29 * log10(d / 10)
    path_loss = 42.0 + 29.0 * math.log10(dist_m / 10.0)
    rssi = tower.tx_power_dbm - path_loss

    # Soft penalty beyond nominal cell radius
    if dist_m > tower.radius_m:
        overshoot = (dist_m - tower.radius_m) / max(100.0, tower.radius_m)
        rssi -= 10.0 * overshoot

    if load_penalty and tower.capacity > 0:
        utilization = tower.connected_count / tower.capacity
        if utilization > 0.75:
            rssi -= (utilization - 0.75) * 16.0

    return round(rssi, 2)


class HandoffManager:
    """Evaluates mobile device signal strength and manages cell/RSU handoffs."""

    def __init__(self, towers: List[TowerState]) -> None:
        self.towers: Dict[str, TowerState] = {t.tower_id: t for t in towers}

    def evaluate_device(
        self,
        device_id: str,
        vehicle_id: str,
        lat: float,
        lon: float,
        current_tower_id: Optional[str],
        speed_kmh: float = 30.0,
    ) -> HandoffDecision:
        if not self.towers:
            raise ValueError("No cell towers configured in HandoffManager")

        # Rank all towers by estimated RSSI
        candidates: List[Tuple[float, TowerState]] = []
        for t in self.towers.values():
            rssi = compute_rssi_dbm(lat, lon, t, load_penalty=True)
            candidates.append((rssi, t))
        candidates.sort(key=lambda item: item[0], reverse=True)

        best_rssi, best_tower = candidates[0]

        # Initial attach if device has no serving tower
        if not current_tower_id or current_tower_id not in self.towers:
            best_tower.connected_count += 1
            return HandoffDecision(
                device_id=device_id,
                vehicle_id=vehicle_id,
                from_tower_id=None,
                to_tower_id=best_tower.tower_id,
                rssi_from_dbm=None,
                rssi_to_dbm=best_rssi,
                handoff_occurred=True,
                latency_ms=14.0,
                reason="initial_attach",
            )

        serving_tower = self.towers[current_tower_id]
        serving_rssi = compute_rssi_dbm(lat, lon, serving_tower, load_penalty=True)

        if best_tower.tower_id == current_tower_id:
            return HandoffDecision(
                device_id=device_id,
                vehicle_id=vehicle_id,
                from_tower_id=current_tower_id,
                to_tower_id=current_tower_id,
                rssi_from_dbm=serving_rssi,
                rssi_to_dbm=serving_rssi,
                handoff_occurred=False,
                latency_ms=0.0,
                reason="stay",
            )

        serving_util = serving_tower.connected_count / max(1, serving_tower.capacity)
        signal_gain = best_rssi - serving_rssi

        trigger_signal = (
            signal_gain >= HYSTERESIS_MARGIN_DB or serving_rssi < MIN_USABLE_RSSI_DBM
        )
        trigger_load = serving_util > 0.85 and signal_gain >= 1.0

        if trigger_signal or trigger_load:
            serving_tower.connected_count = max(0, serving_tower.connected_count - 1)
            best_tower.connected_count += 1
            # Handoff latency scales slightly with vehicle speed (Doppler / signaling overhead)
            latency_ms = round(16.0 + min(22.0, speed_kmh * 0.22), 1)
            reason = "load_balance" if (trigger_load and not trigger_signal) else "signal_threshold"
            return HandoffDecision(
                device_id=device_id,
                vehicle_id=vehicle_id,
                from_tower_id=current_tower_id,
                to_tower_id=best_tower.tower_id,
                rssi_from_dbm=serving_rssi,
                rssi_to_dbm=best_rssi,
                handoff_occurred=True,
                latency_ms=latency_ms,
                reason=reason,
            )

        # Stay on current serving tower (hysteresis prevents ping-ponging)
        return HandoffDecision(
            device_id=device_id,
            vehicle_id=vehicle_id,
            from_tower_id=current_tower_id,
            to_tower_id=current_tower_id,
            rssi_from_dbm=serving_rssi,
            rssi_to_dbm=serving_rssi,
            handoff_occurred=False,
            latency_ms=0.0,
            reason="hysteresis_hold",
        )
