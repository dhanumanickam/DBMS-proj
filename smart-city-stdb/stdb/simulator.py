"""Smart City Road Network Generator, 24-Hour Historical Seeder & Live Mobility Simulator."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .database import SpatioTemporalDB
from .handoff import HandoffManager, TowerState, compute_rssi_dbm
from .spatial_ops import bearing_deg, haversine_m, xy_to_latlon

REF_LAT = 37.7749
REF_LON = -122.4194


@dataclass
class SimVehicle:
    vehicle_id: str
    plate: str
    vehicle_type: str
    device_id: str
    segment_id: str
    from_node: str
    to_node: str
    progress: float  # 0.0 -> 1.0 along current segment
    base_speed_factor: float
    lat: float
    lon: float
    heading_deg: float
    speed_kmh: float
    current_tower_id: Optional[str]
    rssi_dbm: float


class CitySimulator:
    """Simulates vehicle movement on a road graph with V2X handoffs & traffic dynamics."""

    def __init__(self, db: SpatioTemporalDB, seed: int = 42) -> None:
        self.db = db
        self.rng = random.Random(seed)
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.segments: Dict[str, Dict[str, Any]] = {}
        self.adj: Dict[str, List[Tuple[str, str, str]]] = {}  # node -> [(neighbor_node, seg_id, direction)]
        self.handoff_mgr: Optional[HandoffManager] = None
        self.vehicles: Dict[str, SimVehicle] = {}
        self.sim_time: int = int(time.time())

    def load_or_seed(self, num_vehicles: int = 220, force_reseed: bool = False) -> None:
        existing = self.db.get_all_vehicles()
        if existing and not force_reseed:
            self._load_state_from_db()
            return
        self.seed_smart_city(num_vehicles=num_vehicles)

    def _zone_for_grid(self, r: int, c: int) -> str:
        if r >= 3 and c >= 3:
            return "Harbor Expressway"
        if r in (2, 3) and c in (2, 3):
            return "Financial Core"
        if r <= 1 and c <= 2:
            return "Civic Center"
        if r >= 3 and c <= 2:
            return "Tech District"
        if r <= 1 and c >= 3:
            return "South Gate"
        return "Midtown Corridor"

    def _build_road_network(self) -> None:
        self.nodes.clear()
        self.segments.clear()
        self.adj.clear()

        rows, cols = 5, 6
        x_coords = [-2400, -1450, -500, 450, 1400, 2350]
        y_coords = [-1700, -850, 0, 850, 1700]

        avenue_names = [
            "Westgate Ave",
            "Civic Blvd",
            "Central Ave",
            "Market Blvd",
            "Bayfront Pkwy",
            "Harbor Dr",
        ]
        street_names = [
            "South Ring Rd",
            "Oakwood St",
            "Main St",
            "Innovation Way",
            "North Ring Rd",
        ]

        # 1. Intersections (30 nodes with subtle organic jitter)
        for r in range(rows):
            for c in range(cols):
                nid = f"N_{r}_{c}"
                jx = 0.0 if (c in (0, cols - 1)) else self.rng.uniform(-75.0, 75.0)
                jy = 0.0 if (r in (0, rows - 1)) else self.rng.uniform(-65.0, 65.0)
                lat, lon = xy_to_latlon(x_coords[c] + jx, y_coords[r] + jy, REF_LAT, REF_LON)
                zone = self._zone_for_grid(r, c)
                name = f"{street_names[r]} & {avenue_names[c]}"
                self.nodes[nid] = {
                    "node_id": nid,
                    "name": name,
                    "lat": lat,
                    "lon": lon,
                    "zone": zone,
                }
                self.adj[nid] = []
                self.db.upsert_intersection(nid, name, lat, lon, zone, has_signal=1, signal_cycle_sec=60)

        def add_road(
            seg_id: str,
            name: str,
            road_class: str,
            u: str,
            v: str,
            speed_limit: float,
            lanes: int,
            curved: bool = False,
        ) -> None:
            nu, nv = self.nodes[u], self.nodes[v]
            if curved:
                mid_lat = 0.5 * (nu["lat"] + nv["lat"]) + self.rng.uniform(-0.0007, 0.0007)
                mid_lon = 0.5 * (nu["lon"] + nv["lon"]) + self.rng.uniform(-0.0008, 0.0008)
                coords = [[nu["lat"], nu["lon"]], [mid_lat, mid_lon], [nv["lat"], nv["lon"]]]
            else:
                coords = [[nu["lat"], nu["lon"]], [nv["lat"], nv["lon"]]]

            length_m = sum(
                haversine_m(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
                for i in range(len(coords) - 1)
            )
            zone = nu["zone"]
            self.segments[seg_id] = {
                "segment_id": seg_id,
                "name": name,
                "road_class": road_class,
                "from_node": u,
                "to_node": v,
                "coords": coords,
                "length_m": length_m,
                "speed_limit_kmh": speed_limit,
                "lanes": lanes,
                "zone": zone,
            }
            self.adj[u].append((v, seg_id, "fwd"))
            self.adj[v].append((u, seg_id, "rev"))
            self.db.upsert_road_segment(
                seg_id, name, road_class, u, v, coords, speed_limit, lanes, zone
            )

        # Horizontal streets
        for r in range(rows):
            for c in range(cols - 1):
                u = f"N_{r}_{c}"
                v = f"N_{r}_{c+1}"
                is_ring = r in (0, rows - 1)
                is_main = r == 2
                rclass = "highway" if is_ring else ("arterial" if is_main else "collector")
                limit = 70.0 if is_ring else (52.0 if is_main else 40.0)
                lanes = 4 if is_ring else (3 if is_main else 2)
                add_road(
                    f"SEG_H_{r}_{c}",
                    f"{street_names[r]} ({c+1})",
                    rclass,
                    u,
                    v,
                    limit,
                    lanes,
                    curved=is_ring,
                )

        # Vertical avenues
        for r in range(rows - 1):
            for c in range(cols):
                u = f"N_{r}_{c}"
                v = f"N_{r+1}_{c}"
                is_ring = c in (0, cols - 1)
                is_arterial = c in (2, 3)
                rclass = "highway" if is_ring else ("arterial" if is_arterial else "collector")
                limit = 70.0 if is_ring else (50.0 if is_arterial else 40.0)
                lanes = 4 if is_ring else (3 if is_arterial else 2)
                add_road(
                    f"SEG_V_{r}_{c}",
                    f"{avenue_names[c]} ({r+1})",
                    rclass,
                    u,
                    v,
                    limit,
                    lanes,
                    curved=False,
                )

        # Diagonal express connectors & bridge
        diagonals = [
            ("SEG_D_1", "Downtown Crosstown Flyover", "arterial", "N_1_1", "N_2_2", 55.0, 3),
            ("SEG_D_2", "Financial Core Connector", "arterial", "N_2_2", "N_3_3", 50.0, 3),
            ("SEG_D_3", "Harbor Suspension Bridge", "bridge", "N_2_4", "N_3_5", 65.0, 4),
            ("SEG_D_4", "Tech Corridor Express", "arterial", "N_2_1", "N_3_2", 55.0, 3),
        ]
        for sid, name, rclass, u, v, lim, lns in diagonals:
            add_road(sid, name, rclass, u, v, lim, lns, curved=True)

    def _build_cell_towers(self) -> None:
        tower_defs = [
            ("TWR_01", "Westgate 5G Macro", -1900, -1100, 1050, 44.0, "5G-NR", 55, "Civic Center"),
            ("TWR_02", "Civic Plaza RSU", -950, -900, 950, 43.0, "5G-V2X", 50, "Civic Center"),
            ("TWR_03", "South Gate Tower", 950, -1100, 1050, 44.0, "5G-NR", 55, "South Gate"),
            ("TWR_04", "Harbor South RSU", 1950, -900, 1050, 43.0, "5G-V2X", 50, "South Gate"),
            ("TWR_05", "Midtown West RSU", -1100, 100, 920, 43.0, "5G-V2X", 55, "Midtown Corridor"),
            ("TWR_06", "Financial Core Hub", 0, 150, 880, 43.5, "5G-V2X", 45, "Financial Core"),
            ("TWR_07", "Bayfront Exchange", 1050, 100, 920, 43.0, "5G-V2X", 50, "Financial Core"),
            ("TWR_08", "Tech District Node", -950, 1150, 1000, 43.5, "5G-V2X", 55, "Tech District"),
            ("TWR_09", "Innovation Hub 5G", 200, 1150, 960, 43.5, "5G-NR", 50, "Tech District"),
            ("TWR_10", "Harbor Bridge Macro", 1850, 1100, 1150, 44.5, "5G-NR", 60, "Harbor Expressway"),
        ]
        tower_states: List[TowerState] = []
        for tid, name, x, y, rad, txp, tech, cap, zone in tower_defs:
            lat, lon = xy_to_latlon(x, y, REF_LAT, REF_LON)
            self.db.upsert_cell_tower(tid, name, lat, lon, rad, txp, tech, cap, zone)
            tower_states.append(TowerState(tid, name, lat, lon, rad, txp, cap, zone, 0))
        self.handoff_mgr = HandoffManager(tower_states)

    def _interpolate_segment_pos(
        self, seg_id: str, from_node: str, to_node: str, progress: float
    ) -> Tuple[float, float, float]:
        seg = self.segments[seg_id]
        coords = seg["coords"]
        if from_node != seg["from_node"]:
            coords = list(reversed(coords))

        if len(coords) == 2:
            p0, p1 = coords[0], coords[1]
            lat = p0[0] + progress * (p1[0] - p0[0])
            lon = p0[1] + progress * (p1[1] - p0[1])
            hdg = bearing_deg(p0[0], p0[1], p1[0], p1[1])
            return lat, lon, hdg

        # Multi-point polyline
        total = seg["length_m"]
        target = progress * total
        acc = 0.0
        for i in range(len(coords) - 1):
            d = haversine_m(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
            if acc + d >= target or i == len(coords) - 2:
                local_t = (target - acc) / max(1e-6, d)
                local_t = max(0.0, min(1.0, local_t))
                lat = coords[i][0] + local_t * (coords[i + 1][0] - coords[i][0])
                lon = coords[i][1] + local_t * (coords[i + 1][1] - coords[i][1])
                hdg = bearing_deg(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
                return lat, lon, hdg
            acc += d
        return coords[-1][0], coords[-1][1], 0.0

    def _rush_factor(self, ts: int, zone: str) -> float:
        """Return speed multiplier (0.25..1.05) based on hour of day and urban zone."""
        hr = datetime.fromtimestamp(ts).hour + datetime.fromtimestamp(ts).minute / 60.0
        # Morning peak ~8:30, Evening peak ~18:00
        morning_peak = math.exp(-((hr - 8.5) ** 2) / 1.8)
        evening_peak = math.exp(-((hr - 18.0) ** 2) / 2.2)
        rush = max(morning_peak, evening_peak)

        zone_bias = 1.0
        if zone in ("Financial Core", "Midtown Corridor"):
            zone_bias = 1.25
        elif zone == "Harbor Expressway":
            zone_bias = 1.05
        else:
            zone_bias = 0.85

        # Also create 2 persistent realistic hotspot corridors around Downtown / Bridge
        factor = 1.0 - min(0.68, rush * 0.58 * zone_bias)
        return max(0.24, factor)

    def seed_smart_city(self, num_vehicles: int = 220) -> None:
        self.db.clear_all()
        self._build_road_network()
        self._build_cell_towers()

        seg_ids = list(self.segments.keys())
        # Bias ~38% of vehicles near Financial Core / Midtown / Harbor Bridge so hotspots form naturally
        core_segs = [
            sid
            for sid, s in self.segments.items()
            if s["zone"] in ("Financial Core", "Midtown Corridor") or sid in ("SEG_D_2", "SEG_D_3")
        ]
        vtypes = ["car", "car", "car", "ev_taxi", "ev_taxi", "delivery", "bus", "emergency"]

        now_ts = int(time.time())
        # Round to nearest 60s
        now_ts = (now_ts // 60) * 60
        start_ts = now_ts - 24 * 3600

        self.vehicles.clear()
        for i in range(1, num_vehicles + 1):
            vid = f"VEH_{i:03d}"
            plate = f"SC-{1000 + i}"
            vtype = self.rng.choice(vtypes)
            dev_id = f"MOB_{i:03d}"
            sid = self.rng.choice(core_segs if (i % 5 < 2 and core_segs) else seg_ids)
            seg = self.segments[sid]
            if self.rng.random() < 0.5:
                u, v = seg["from_node"], seg["to_node"]
            else:
                u, v = seg["to_node"], seg["from_node"]
            prog = self.rng.random()
            lat, lon, hdg = self._interpolate_segment_pos(sid, u, v, prog)
            sv = SimVehicle(
                vehicle_id=vid,
                plate=plate,
                vehicle_type=vtype,
                device_id=dev_id,
                segment_id=sid,
                from_node=u,
                to_node=v,
                progress=prog,
                base_speed_factor=self.rng.uniform(0.84, 1.12),
                lat=lat,
                lon=lon,
                heading_deg=hdg,
                speed_kmh=seg["speed_limit_kmh"] * 0.7,
                current_tower_id=None,
                rssi_dbm=-65.0,
            )
            self.vehicles[vid] = sv

        # Seed 24h of historical trajectory snapshots (every 20 mins = 72 steps)
        # Keep the last 10 steps denser (every 2 minutes) for crisp recent trajectory playback
        timestamps: List[int] = list(range(start_ts, now_ts - 1200, 1200))
        timestamps.extend(range(now_ts - 1200, now_ts + 1, 120))

        prev_ts = timestamps[0] - 1200
        for step_idx, ts in enumerate(timestamps):
            dt = max(60, ts - prev_ts)
            prev_ts = ts
            # Simulate movement; record handoff events primarily in the recent window to keep log crisp
            record_handoff = step_idx >= len(timestamps) - 18
            self._advance_step(ts=ts, dt_sec=min(dt, 180), record_handoffs=record_handoff, commit=False)

        # Inject one live bottleneck hotspot near Financial Core / Bridge for immediate visual mining insight
        self.db.add_incident("SEG_D_2", severity=0.78, description="Heavy intersection queue (Peak Flow)", duration_sec=600, now_ts=now_ts)
        # Do 2 quick micro-ticks at now_ts so speeds reflect incident & hotspot immediately
        self._advance_step(ts=now_ts + 5, dt_sec=10, record_handoffs=True, commit=False)
        self.sim_time = now_ts + 5
        self.db.commit()
        self.db.run_and_store_mining()

    def _load_state_from_db(self) -> None:
        net = self.db.get_network()
        self.nodes = {n["node_id"]: n for n in net["intersections"]}
        self.segments = {s["segment_id"]: s for s in net["roads"]}
        self.adj = {nid: [] for nid in self.nodes}
        for sid, s in self.segments.items():
            self.adj[s["from_node"]].append((s["to_node"], sid, "fwd"))
            self.adj[s["to_node"]].append((s["from_node"], sid, "rev"))

        tower_states: List[TowerState] = []
        v_rows = self.db.get_all_vehicles()
        counts: Dict[str, int] = {}
        for v in v_rows:
            tid = v.get("current_tower_id")
            if tid:
                counts[tid] = counts.get(tid, 0) + 1
        for t in net["towers"]:
            tower_states.append(
                TowerState(
                    t["tower_id"],
                    t["name"],
                    t["lat"],
                    t["lon"],
                    t["radius_m"],
                    t["tx_power_dbm"],
                    t["capacity"],
                    t["zone"],
                    counts.get(t["tower_id"], 0),
                )
            )
        self.handoff_mgr = HandoffManager(tower_states)

        self.vehicles.clear()
        for v in v_rows:
            sid = v["current_segment_id"] or next(iter(self.segments))
            seg = self.segments[sid]
            self.vehicles[v["vehicle_id"]] = SimVehicle(
                vehicle_id=v["vehicle_id"],
                plate=v["plate"],
                vehicle_type=v["vehicle_type"],
                device_id=v["device_id"],
                segment_id=sid,
                from_node=seg["from_node"],
                to_node=seg["to_node"],
                progress=0.5,
                base_speed_factor=1.0,
                lat=v["lat"],
                lon=v["lon"],
                heading_deg=v["heading_deg"],
                speed_kmh=v["speed_kmh"],
                current_tower_id=v["current_tower_id"],
                rssi_dbm=v["rssi_dbm"],
            )
        bounds = self.db.get_time_bounds()
        self.sim_time = bounds["max_ts"] or int(time.time())

    def _advance_step(
        self,
        ts: int,
        dt_sec: float,
        record_handoffs: bool = True,
        commit: bool = True,
    ) -> Dict[str, Any]:
        assert self.handoff_mgr is not None
        # Compute segment density and active incidents
        seg_counts: Dict[str, int] = {}
        for sv in self.vehicles.values():
            seg_counts[sv.segment_id] = seg_counts.get(sv.segment_id, 0) + 1

        incidents = {
            inc["segment_id"]: float(inc["severity"])
            for inc in self.db.get_active_incidents(now_ts=ts)
        }
        # Also simulate an organic downtown bottleneck cluster near SEG_D_2 & SEG_H_2_2 & SEG_V_2_2
        hot_segments = {"SEG_D_2", "SEG_H_2_2", "SEG_V_2_2", "SEG_V_2_3"}

        handoff_count = 0
        for sv in self.vehicles.values():
            seg = self.segments[sv.segment_id]
            zone = seg["zone"]
            rush_mult = self._rush_factor(ts, zone)

            cnt = seg_counts.get(sv.segment_id, 1)
            density_penalty = max(0.32, 1.0 - max(0, cnt - 3) * 0.06)

            hotspot_mult = 0.28 if sv.segment_id in hot_segments else 1.0
            incident_mult = 1.0 - incidents.get(sv.segment_id, 0.0)

            raw_speed = (
                seg["speed_limit_kmh"]
                * sv.base_speed_factor
                * rush_mult
                * density_penalty
                * hotspot_mult
                * incident_mult
            )
            # Small stochastic fluctuation
            raw_speed *= self.rng.uniform(0.90, 1.08)
            speed_kmh = max(3.5, min(seg["speed_limit_kmh"] * 1.08, raw_speed))
            sv.speed_kmh = round(speed_kmh, 1)

            # Advance along segment
            dist_m = (speed_kmh / 3.6) * dt_sec
            seg_len = max(50.0, seg["length_m"])
            sv.progress += dist_m / seg_len

            while sv.progress >= 1.0:
                overflow_m = (sv.progress - 1.0) * seg_len
                curr_node = sv.to_node
                prev_node = sv.from_node
                choices = self.adj.get(curr_node, [])
                if not choices:
                    sv.progress = 0.0
                    break
                non_uturn = [ch for ch in choices if ch[0] != prev_node]
                nxt_node, nxt_seg, _ = self.rng.choice(non_uturn or choices)
                sv.from_node = curr_node
                sv.to_node = nxt_node
                sv.segment_id = nxt_seg
                seg = self.segments[nxt_seg]
                seg_len = max(50.0, seg["length_m"])
                sv.progress = overflow_m / seg_len

            sv.lat, sv.lon, sv.heading_deg = self._interpolate_segment_pos(
                sv.segment_id, sv.from_node, sv.to_node, sv.progress
            )

            # Evaluate mobile device location & cellular/V2X handoff
            decision = self.handoff_mgr.evaluate_device(
                device_id=sv.device_id,
                vehicle_id=sv.vehicle_id,
                lat=sv.lat,
                lon=sv.lon,
                current_tower_id=sv.current_tower_id,
                speed_kmh=sv.speed_kmh,
            )
            sv.current_tower_id = decision.to_tower_id
            sv.rssi_dbm = decision.rssi_to_dbm
            if decision.handoff_occurred and record_handoffs:
                handoff_count += 1
                self.db.record_handoff(
                    device_id=sv.device_id,
                    vehicle_id=sv.vehicle_id,
                    from_tower_id=decision.from_tower_id,
                    to_tower_id=decision.to_tower_id,
                    lat=sv.lat,
                    lon=sv.lon,
                    rssi_from_dbm=decision.rssi_from_dbm,
                    rssi_to_dbm=decision.rssi_to_dbm,
                    latency_ms=decision.latency_ms,
                    reason=decision.reason,
                    handoff_at=ts,
                )

            status = "stopped" if sv.speed_kmh < 5.0 else ("congested" if sv.speed_kmh < 18.0 else "moving")
            self.db.upsert_vehicle(
                vehicle_id=sv.vehicle_id,
                plate=sv.plate,
                vehicle_type=sv.vehicle_type,
                device_id=sv.device_id,
                lat=round(sv.lat, 6),
                lon=round(sv.lon, 6),
                heading_deg=round(sv.heading_deg, 1),
                speed_kmh=sv.speed_kmh,
                current_segment_id=sv.segment_id,
                current_tower_id=sv.current_tower_id,
                rssi_dbm=sv.rssi_dbm,
                status=status,
                updated_at=ts,
                log_trajectory=True,
            )

        if commit:
            self.db.commit()
        return {"timestamp": ts, "vehicles_updated": len(self.vehicles), "handoffs": handoff_count}

    def tick(self, dt_sec: float = 8.0) -> Dict[str, Any]:
        """Advance live simulation by dt_sec seconds."""
        self.sim_time += max(1, int(dt_sec))
        return self._advance_step(ts=self.sim_time, dt_sec=dt_sec, record_handoffs=True, commit=True)
