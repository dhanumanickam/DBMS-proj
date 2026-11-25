"""Spatio-Temporal Database Layer: SQLite + R-Tree, Quadtree, Temporal & Mining Queries."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .mining import compute_getis_ord_hotspots, detect_congestion_dbscan
from .spatial_index import QuadtreeIndex, SpatialPoint, benchmark_spatial_indexes
from .spatial_ops import (
    compute_bbox,
    haversine_m,
    point_in_polygon,
    point_to_polyline_distance_m,
    radius_bbox,
    to_wkt_linestring,
)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class SpatioTemporalDB:
    """Thread-safe Spatio-Temporal Database backed by SQLite + R*Tree + Quadtree."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.init_schema()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def init_schema(self) -> None:
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        with self._lock:
            self.conn.executescript(sql)
            self.conn.commit()

    def clear_all(self) -> None:
        with self._lock:
            tables = [
                "traffic_incidents",
                "congestion_hotspots",
                "handoff_events",
                "vehicle_trajectories",
                "vehicles_rtree",
                "vehicles_rtree_map",
                "vehicles",
                "towers_rtree",
                "towers_rtree_map",
                "cell_towers",
                "roads_rtree",
                "roads_rtree_map",
                "road_segments",
                "intersections",
            ]
            for tbl in tables:
                self.conn.execute(f"DELETE FROM {tbl}")
            self.conn.commit()

    # ------------------------------------------------------------------
    # Spatial Network & Entity Persistence (keeps R-Trees synchronized)
    # ------------------------------------------------------------------
    def upsert_intersection(
        self,
        node_id: str,
        name: str,
        lat: float,
        lon: float,
        zone: str,
        has_signal: int = 1,
        signal_cycle_sec: int = 60,
    ) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO intersections (node_id, name, lat, lon, zone, has_signal, signal_cycle_sec)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    name=excluded.name, lat=excluded.lat, lon=excluded.lon,
                    zone=excluded.zone, has_signal=excluded.has_signal,
                    signal_cycle_sec=excluded.signal_cycle_sec
                """,
                (node_id, name, lat, lon, zone, has_signal, signal_cycle_sec),
            )

    def upsert_road_segment(
        self,
        segment_id: str,
        name: str,
        road_class: str,
        from_node: str,
        to_node: str,
        coords: Sequence[Sequence[float]],
        speed_limit_kmh: float,
        lanes: int,
        zone: str,
    ) -> None:
        min_lat, max_lat, min_lon, max_lon = compute_bbox(coords)
        length_m = 0.0
        for i in range(len(coords) - 1):
            length_m += haversine_m(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
        wkt = to_wkt_linestring(coords)
        coords_json = json.dumps([[round(c[0], 6), round(c[1], 6)] for c in coords])

        with self._lock:
            self.conn.execute(
                """
                INSERT INTO road_segments (
                    segment_id, name, road_class, from_node, to_node,
                    wkt_geometry, coords_json, min_lat, max_lat, min_lon, max_lon,
                    length_m, speed_limit_kmh, lanes, zone
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(segment_id) DO UPDATE SET
                    name=excluded.name, road_class=excluded.road_class,
                    wkt_geometry=excluded.wkt_geometry, coords_json=excluded.coords_json,
                    min_lat=excluded.min_lat, max_lat=excluded.max_lat,
                    min_lon=excluded.min_lon, max_lon=excluded.max_lon,
                    length_m=excluded.length_m, speed_limit_kmh=excluded.speed_limit_kmh,
                    lanes=excluded.lanes, zone=excluded.zone
                """,
                (
                    segment_id,
                    name,
                    road_class,
                    from_node,
                    to_node,
                    wkt,
                    coords_json,
                    min_lat,
                    max_lat,
                    min_lon,
                    max_lon,
                    round(length_m, 2),
                    speed_limit_kmh,
                    lanes,
                    zone,
                ),
            )
            row = self.conn.execute(
                "SELECT rtree_id FROM roads_rtree_map WHERE segment_id = ?", (segment_id,)
            ).fetchone()
            if row is None:
                cur = self.conn.execute(
                    "INSERT INTO roads_rtree_map (segment_id) VALUES (?)", (segment_id,)
                )
                rtree_id = cur.lastrowid
                self.conn.execute(
                    "INSERT INTO roads_rtree (id, min_lon, max_lon, min_lat, max_lat) VALUES (?, ?, ?, ?, ?)",
                    (rtree_id, min_lon, max_lon, min_lat, max_lat),
                )
            else:
                self.conn.execute(
                    "UPDATE roads_rtree SET min_lon=?, max_lon=?, min_lat=?, max_lat=? WHERE id=?",
                    (min_lon, max_lon, min_lat, max_lat, row[0]),
                )

    def upsert_cell_tower(
        self,
        tower_id: str,
        name: str,
        lat: float,
        lon: float,
        radius_m: float,
        tx_power_dbm: float,
        technology: str,
        capacity: int,
        zone: str,
    ) -> None:
        min_lat, max_lat, min_lon, max_lon = radius_bbox(lat, lon, radius_m)
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO cell_towers (tower_id, name, lat, lon, radius_m, tx_power_dbm, technology, capacity, zone)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tower_id) DO UPDATE SET
                    name=excluded.name, lat=excluded.lat, lon=excluded.lon,
                    radius_m=excluded.radius_m, tx_power_dbm=excluded.tx_power_dbm,
                    technology=excluded.technology, capacity=excluded.capacity, zone=excluded.zone
                """,
                (tower_id, name, lat, lon, radius_m, tx_power_dbm, technology, capacity, zone),
            )
            row = self.conn.execute(
                "SELECT rtree_id FROM towers_rtree_map WHERE tower_id = ?", (tower_id,)
            ).fetchone()
            if row is None:
                cur = self.conn.execute(
                    "INSERT INTO towers_rtree_map (tower_id) VALUES (?)", (tower_id,)
                )
                rtree_id = cur.lastrowid
                self.conn.execute(
                    "INSERT INTO towers_rtree (id, min_lon, max_lon, min_lat, max_lat) VALUES (?, ?, ?, ?, ?)",
                    (rtree_id, min_lon, max_lon, min_lat, max_lat),
                )
            else:
                self.conn.execute(
                    "UPDATE towers_rtree SET min_lon=?, max_lon=?, min_lat=?, max_lat=? WHERE id=?",
                    (min_lon, max_lon, min_lat, max_lat, row[0]),
                )

    def upsert_vehicle(
        self,
        vehicle_id: str,
        plate: str,
        vehicle_type: str,
        device_id: str,
        lat: float,
        lon: float,
        heading_deg: float,
        speed_kmh: float,
        current_segment_id: Optional[str],
        current_tower_id: Optional[str],
        rssi_dbm: float,
        status: str,
        updated_at: int,
        log_trajectory: bool = True,
    ) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO vehicles (
                    vehicle_id, plate, vehicle_type, device_id, lat, lon,
                    heading_deg, speed_kmh, current_segment_id, current_tower_id,
                    rssi_dbm, status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(vehicle_id) DO UPDATE SET
                    lat=excluded.lat, lon=excluded.lon, heading_deg=excluded.heading_deg,
                    speed_kmh=excluded.speed_kmh, current_segment_id=excluded.current_segment_id,
                    current_tower_id=excluded.current_tower_id, rssi_dbm=excluded.rssi_dbm,
                    status=excluded.status, updated_at=excluded.updated_at
                """,
                (
                    vehicle_id,
                    plate,
                    vehicle_type,
                    device_id,
                    lat,
                    lon,
                    heading_deg,
                    speed_kmh,
                    current_segment_id,
                    current_tower_id,
                    rssi_dbm,
                    status,
                    updated_at,
                ),
            )
            row = self.conn.execute(
                "SELECT rtree_id FROM vehicles_rtree_map WHERE vehicle_id = ?", (vehicle_id,)
            ).fetchone()
            if row is None:
                cur = self.conn.execute(
                    "INSERT INTO vehicles_rtree_map (vehicle_id) VALUES (?)", (vehicle_id,)
                )
                rtree_id = cur.lastrowid
                self.conn.execute(
                    "INSERT INTO vehicles_rtree (id, min_lon, max_lon, min_lat, max_lat) VALUES (?, ?, ?, ?, ?)",
                    (rtree_id, lon, lon, lat, lat),
                )
            else:
                self.conn.execute(
                    "UPDATE vehicles_rtree SET min_lon=?, max_lon=?, min_lat=?, max_lat=? WHERE id=?",
                    (lon, lon, lat, lat, row[0]),
                )
            if log_trajectory:
                self.conn.execute(
                    """
                    INSERT INTO vehicle_trajectories (
                        vehicle_id, lat, lon, speed_kmh, heading_deg,
                        segment_id, tower_id, rssi_dbm, recorded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        vehicle_id,
                        lat,
                        lon,
                        speed_kmh,
                        heading_deg,
                        current_segment_id,
                        current_tower_id,
                        rssi_dbm,
                        updated_at,
                    ),
                )

    def record_handoff(
        self,
        device_id: str,
        vehicle_id: str,
        from_tower_id: Optional[str],
        to_tower_id: str,
        lat: float,
        lon: float,
        rssi_from_dbm: Optional[float],
        rssi_to_dbm: float,
        latency_ms: float,
        reason: str,
        handoff_at: int,
    ) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO handoff_events (
                    device_id, vehicle_id, from_tower_id, to_tower_id,
                    lat, lon, rssi_from_dbm, rssi_to_dbm, latency_ms, reason, handoff_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    device_id,
                    vehicle_id,
                    from_tower_id,
                    to_tower_id,
                    lat,
                    lon,
                    rssi_from_dbm,
                    rssi_to_dbm,
                    latency_ms,
                    reason,
                    handoff_at,
                ),
            )

    def commit(self) -> None:
        with self._lock:
            self.conn.commit()

    # ------------------------------------------------------------------
    # Basic Readers
    # ------------------------------------------------------------------
    def get_network(self) -> Dict[str, Any]:
        with self._lock:
            intersections = [
                dict(r)
                for r in self.conn.execute("SELECT * FROM intersections ORDER BY node_id").fetchall()
            ]
            roads = []
            for r in self.conn.execute("SELECT * FROM road_segments ORDER BY segment_id").fetchall():
                d = dict(r)
                d["coords"] = json.loads(d.pop("coords_json"))
                roads.append(d)
            towers = [
                dict(r)
                for r in self.conn.execute("SELECT * FROM cell_towers ORDER BY tower_id").fetchall()
            ]
            bounds = self.get_city_bounds()
        return {
            "intersections": intersections,
            "roads": roads,
            "towers": towers,
            "bounds": {
                "min_lat": bounds[0],
                "max_lat": bounds[1],
                "min_lon": bounds[2],
                "max_lon": bounds[3],
            },
        }

    def get_city_bounds(self) -> Tuple[float, float, float, float]:
        with self._lock:
            row = self.conn.execute(
                "SELECT MIN(lat), MAX(lat), MIN(lon), MAX(lon) FROM intersections"
            ).fetchone()
        if not row or row[0] is None:
            return (37.752, 37.798, -122.452, -122.388)
        return (row[0] - 0.003, row[1] + 0.003, row[2] - 0.004, row[3] + 0.004)

    def get_all_vehicles(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT v.*, r.name AS segment_name, r.road_class, r.speed_limit_kmh, r.zone,
                       t.name AS tower_name
                FROM vehicles v
                LEFT JOIN road_segments r ON v.current_segment_id = r.segment_id
                LEFT JOIN cell_towers t ON v.current_tower_id = t.tower_id
                ORDER BY v.vehicle_id
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def get_active_incidents(self, now_ts: Optional[int] = None) -> List[Dict[str, Any]]:
        ts = now_ts if now_ts is not None else int(time.time())
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT i.*, r.name AS segment_name, r.zone
                FROM traffic_incidents i
                JOIN road_segments r ON i.segment_id = r.segment_id
                WHERE i.expires_at >= ?
                ORDER BY i.created_at DESC
                """,
                (ts,),
            ).fetchall()
        return [dict(r) for r in rows]

    def add_incident(
        self,
        segment_id: str,
        severity: float = 0.85,
        description: str = "Lane blockage / collision reported",
        duration_sec: int = 300,
        now_ts: Optional[int] = None,
    ) -> Dict[str, Any]:
        ts = now_ts if now_ts is not None else int(time.time())
        with self._lock:
            seg = self.conn.execute(
                "SELECT coords_json, name, zone FROM road_segments WHERE segment_id = ?",
                (segment_id,),
            ).fetchone()
            if seg is None:
                raise ValueError(f"Unknown segment_id: {segment_id}")
            coords = json.loads(seg["coords_json"])
            mid = coords[len(coords) // 2]
            cur = self.conn.execute(
                """
                INSERT INTO traffic_incidents (segment_id, lat, lon, severity, description, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (segment_id, mid[0], mid[1], severity, description, ts, ts + duration_sec),
            )
            self.conn.commit()
            return {
                "incident_id": cur.lastrowid,
                "segment_id": segment_id,
                "segment_name": seg["name"],
                "zone": seg["zone"],
                "lat": mid[0],
                "lon": mid[1],
                "severity": severity,
                "description": description,
                "created_at": ts,
                "expires_at": ts + duration_sec,
            }

    # ------------------------------------------------------------------
    # Spatial Operators & Indexed Queries
    # ------------------------------------------------------------------
    def _build_quadtree(self, vehicles: Optional[Sequence[Dict[str, Any]]] = None) -> QuadtreeIndex:
        rows = vehicles if vehicles is not None else self.get_all_vehicles()
        pts = [
            SpatialPoint(v["vehicle_id"], float(v["lat"]), float(v["lon"]), v) for v in rows
        ]
        min_lat, max_lat, min_lon, max_lon = self.get_city_bounds()
        qt = QuadtreeIndex(min_lat - 0.002, max_lat + 0.002, min_lon - 0.002, max_lon + 0.002)
        for pt in pts:
            qt.insert(pt.item_id, pt.lat, pt.lon, pt.payload)
        return qt

    def query_vehicles_radius(
        self, lat: float, lon: float, radius_m: float
    ) -> Dict[str, Any]:
        """R-Tree accelerated radius range query with geodesic verification."""
        t0 = time.perf_counter()
        min_lat, max_lat, min_lon, max_lon = radius_bbox(lat, lon, radius_m)
        with self._lock:
            candidates = self.conn.execute(
                """
                SELECT v.*, r.name AS segment_name, r.zone, t.name AS tower_name
                FROM vehicles_rtree rt
                JOIN vehicles_rtree_map m ON rt.id = m.rtree_id
                JOIN vehicles v ON m.vehicle_id = v.vehicle_id
                LEFT JOIN road_segments r ON v.current_segment_id = r.segment_id
                LEFT JOIN cell_towers t ON v.current_tower_id = t.tower_id
                WHERE rt.min_lon <= ? AND rt.max_lon >= ?
                  AND rt.min_lat <= ? AND rt.max_lat >= ?
                """,
                (max_lon, min_lon, max_lat, min_lat),
            ).fetchall()

        hits: List[Dict[str, Any]] = []
        for row in candidates:
            d = haversine_m(lat, lon, row["lat"], row["lon"])
            if d <= radius_m:
                item = dict(row)
                item["distance_m"] = round(d, 1)
                hits.append(item)
        hits.sort(key=lambda x: x["distance_m"])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "operator": "ST_DWithin (SQLite R-Tree + Haversine)",
            "center": {"lat": lat, "lon": lon},
            "radius_m": radius_m,
            "candidates_examined": len(candidates),
            "count": len(hits),
            "query_ms": round(elapsed_ms, 3),
            "vehicles": hits,
        }

    def query_vehicles_knn(self, lat: float, lon: float, k: int = 5) -> Dict[str, Any]:
        """Best-First Quadtree k-Nearest Neighbors search."""
        t0 = time.perf_counter()
        qt = self._build_quadtree()
        hits, nodes_visited = qt.knn(lat, lon, k=k)
        vehicles: List[Dict[str, Any]] = []
        for pt, dist_m in hits:
            item = dict(pt.payload)
            item["distance_m"] = round(dist_m, 1)
            vehicles.append(item)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "operator": "ST_KNN (Quadtree Best-First Branch-and-Bound)",
            "center": {"lat": lat, "lon": lon},
            "k": k,
            "nodes_visited": nodes_visited,
            "query_ms": round(elapsed_ms, 3),
            "vehicles": vehicles,
        }

    def query_vehicles_polygon(self, polygon: Sequence[Sequence[float]]) -> Dict[str, Any]:
        """Polygon containment query: R-Tree MBR filter + Ray-Casting refinement."""
        t0 = time.perf_counter()
        min_lat, max_lat, min_lon, max_lon = compute_bbox(polygon)
        with self._lock:
            candidates = self.conn.execute(
                """
                SELECT v.*, r.name AS segment_name, r.zone
                FROM vehicles_rtree rt
                JOIN vehicles_rtree_map m ON rt.id = m.rtree_id
                JOIN vehicles v ON m.vehicle_id = v.vehicle_id
                LEFT JOIN road_segments r ON v.current_segment_id = r.segment_id
                WHERE rt.min_lon <= ? AND rt.max_lon >= ?
                  AND rt.min_lat <= ? AND rt.max_lat >= ?
                """,
                (max_lon, min_lon, max_lat, min_lat),
            ).fetchall()

        hits = [
            dict(row)
            for row in candidates
            if point_in_polygon(row["lat"], row["lon"], polygon)
        ]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "operator": "ST_Within (R-Tree MBR + Ray-Casting PIP)",
            "candidates_examined": len(candidates),
            "count": len(hits),
            "query_ms": round(elapsed_ms, 3),
            "vehicles": hits,
        }

    def query_roads_near(
        self, lat: float, lon: float, radius_m: float = 350.0
    ) -> List[Dict[str, Any]]:
        """Spatial join / nearest road segments using roads_rtree + point-to-polyline distance."""
        min_lat, max_lat, min_lon, max_lon = radius_bbox(lat, lon, radius_m)
        with self._lock:
            candidates = self.conn.execute(
                """
                SELECT r.*
                FROM roads_rtree rt
                JOIN roads_rtree_map m ON rt.id = m.rtree_id
                JOIN road_segments r ON m.segment_id = r.segment_id
                WHERE rt.min_lon <= ? AND rt.max_lon >= ?
                  AND rt.min_lat <= ? AND rt.max_lat >= ?
                """,
                (max_lon, min_lon, max_lat, min_lat),
            ).fetchall()

        results: List[Dict[str, Any]] = []
        for row in candidates:
            coords = json.loads(row["coords_json"])
            dist = point_to_polyline_distance_m(lat, lon, coords)
            if dist <= radius_m:
                d = dict(row)
                d["coords"] = coords
                d.pop("coords_json", None)
                d["distance_m"] = round(dist, 1)
                results.append(d)
        results.sort(key=lambda r: r["distance_m"])
        return results

    def get_quadtree_cells(self) -> Dict[str, Any]:
        qt = self._build_quadtree()
        cells = qt.export_cells()
        return {"total_points": qt.size, "leaf_cells": len(cells), "cells": cells}

    def run_index_benchmark(self, num_queries: int = 40, radius_m: float = 450.0) -> Dict[str, Any]:
        vehicles = self.get_all_vehicles()
        pts = [SpatialPoint(v["vehicle_id"], v["lat"], v["lon"], v) for v in vehicles]
        min_lat, max_lat, min_lon, max_lon = self.get_city_bounds()
        query_points = []
        for i in range(num_queries):
            frac_y = ((i * 37) % 100) / 100.0
            frac_x = ((i * 53) % 100) / 100.0
            q_lat = min_lat + 0.15 * (max_lat - min_lat) + 0.7 * frac_y * (max_lat - min_lat)
            q_lon = min_lon + 0.15 * (max_lon - min_lon) + 0.7 * frac_x * (max_lon - min_lon)
            query_points.append((q_lat, q_lon))
        with self._lock:
            return benchmark_spatial_indexes(pts, self.conn, query_points, radius_m=radius_m)

    # ------------------------------------------------------------------
    # Temporal & Spatio-Temporal Queries
    # ------------------------------------------------------------------
    def get_time_bounds(self) -> Dict[str, Optional[int]]:
        with self._lock:
            row = self.conn.execute(
                "SELECT MIN(recorded_at) AS min_ts, MAX(recorded_at) AS max_ts, COUNT(*) AS obs_count FROM vehicle_trajectories"
            ).fetchone()
        return {
            "min_ts": row["min_ts"] if row else None,
            "max_ts": row["max_ts"] if row else None,
            "observation_count": row["obs_count"] if row else 0,
        }

    def snapshot_at(self, target_ts: int, window_sec: int = 900) -> Dict[str, Any]:
        """Reconstruct city-wide vehicle state at historical timestamp target_ts."""
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT t.vehicle_id, t.lat, t.lon, t.speed_kmh, t.heading_deg,
                       t.segment_id AS current_segment_id, t.tower_id AS current_tower_id,
                       t.rssi_dbm, t.recorded_at,
                       v.plate, v.vehicle_type, v.device_id,
                       r.name AS segment_name, r.road_class, r.speed_limit_kmh, r.zone,
                       ct.name AS tower_name
                FROM vehicle_trajectories t
                JOIN vehicles v ON t.vehicle_id = v.vehicle_id
                LEFT JOIN road_segments r ON t.segment_id = r.segment_id
                LEFT JOIN cell_towers ct ON t.tower_id = ct.tower_id
                WHERE t.recorded_at BETWEEN ? AND ?
                ORDER BY ABS(t.recorded_at - ?) ASC
                """,
                (target_ts - window_sec, target_ts + window_sec, target_ts),
            ).fetchall()

        seen: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            vid = row["vehicle_id"]
            if vid not in seen:
                d = dict(row)
                spd = float(d["speed_kmh"])
                d["status"] = "stopped" if spd < 5.0 else ("congested" if spd < 18.0 else "moving")
                seen[vid] = d

        vehicles = list(seen.values())
        avg_speed = (
            round(sum(v["speed_kmh"] for v in vehicles) / len(vehicles), 1)
            if vehicles
            else 0.0
        )
        congested_count = sum(1 for v in vehicles if v["speed_kmh"] < 18.0)
        return {
            "timestamp": target_ts,
            "iso_time": datetime.fromtimestamp(target_ts).strftime("%Y-%m-%d %H:%M:%S"),
            "vehicle_count": len(vehicles),
            "avg_speed_kmh": avg_speed,
            "congested_count": congested_count,
            "vehicles": vehicles,
        }

    def get_vehicle_trajectory(
        self,
        vehicle_id: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        limit: int = 300,
    ) -> List[Dict[str, Any]]:
        query = """
            SELECT obs_id, vehicle_id, lat, lon, speed_kmh, heading_deg,
                   segment_id, tower_id, rssi_dbm, recorded_at
            FROM vehicle_trajectories
            WHERE vehicle_id = ?
        """
        params: List[Any] = [vehicle_id]
        if start_ts is not None:
            query += " AND recorded_at >= ?"
            params.append(start_ts)
        if end_ts is not None:
            query += " AND recorded_at <= ?"
            params.append(end_ts)
        query += " ORDER BY recorded_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self.conn.execute(query, params).fetchall()
        points = [dict(r) for r in reversed(rows)]
        return points

    def query_spatiotemporal_window(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        start_ts: int,
        end_ts: int,
        limit: int = 500,
    ) -> Dict[str, Any]:
        """Find all trajectory observations inside spatial region & temporal window."""
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT t.*, v.plate, v.vehicle_type, r.name AS segment_name
                FROM vehicle_trajectories t
                JOIN vehicles v ON t.vehicle_id = v.vehicle_id
                LEFT JOIN road_segments r ON t.segment_id = r.segment_id
                WHERE t.recorded_at BETWEEN ? AND ?
                  AND t.lat BETWEEN ? AND ?
                  AND t.lon BETWEEN ? AND ?
                ORDER BY t.recorded_at DESC
                LIMIT ?
                """,
                (start_ts, end_ts, min_lat, max_lat, min_lon, max_lon, limit),
            ).fetchall()
        obs = [dict(r) for r in rows]
        unique_vehicles = len({o["vehicle_id"] for o in obs})
        return {
            "observation_count": len(obs),
            "unique_vehicles": unique_vehicles,
            "observations": obs,
        }

    def get_hourly_traffic_profile(self, lookback_hours: int = 24) -> List[Dict[str, Any]]:
        """Aggregate historical observations by hour bucket for trend visualization."""
        bounds = self.get_time_bounds()
        max_ts = bounds["max_ts"]
        if max_ts is None:
            return []
        min_ts = max_ts - lookback_hours * 3600
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT (recorded_at / 3600) * 3600 AS hour_ts,
                       COUNT(*) AS observations,
                       COUNT(DISTINCT vehicle_id) AS active_vehicles,
                       ROUND(AVG(speed_kmh), 1) AS avg_speed_kmh,
                       ROUND(100.0 * SUM(CASE WHEN speed_kmh < 18.0 THEN 1 ELSE 0 END) / COUNT(*), 1) AS congestion_pct
                FROM vehicle_trajectories
                WHERE recorded_at >= ?
                GROUP BY hour_ts
                ORDER BY hour_ts ASC
                """,
                (min_ts,),
            ).fetchall()

        profile: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            d["hour_label"] = datetime.fromtimestamp(d["hour_ts"]).strftime("%H:%M")
            profile.append(d)
        return profile

    def get_segment_live_metrics(self, timestamp: Optional[int] = None) -> List[Dict[str, Any]]:
        """Compute per-road-segment speed and congestion ratio (live or at historical snapshot)."""
        if timestamp is None:
            vehicles = self.get_all_vehicles()
        else:
            vehicles = self.snapshot_at(timestamp)["vehicles"]

        seg_buckets: Dict[str, List[float]] = {}
        for v in vehicles:
            sid = v.get("current_segment_id")
            if sid:
                seg_buckets.setdefault(sid, []).append(float(v["speed_kmh"]))

        with self._lock:
            roads = self.conn.execute(
                "SELECT segment_id, name, road_class, speed_limit_kmh, zone FROM road_segments"
            ).fetchall()

        metrics: List[Dict[str, Any]] = []
        for r in roads:
            sid = r["segment_id"]
            speeds = seg_buckets.get(sid, [])
            limit = float(r["speed_limit_kmh"])
            avg_spd = round(sum(speeds) / len(speeds), 1) if speeds else limit
            congestion_index = round(max(0.0, min(1.0, 1.0 - (avg_spd / max(1.0, limit)))), 2)
            metrics.append(
                {
                    "segment_id": sid,
                    "name": r["name"],
                    "road_class": r["road_class"],
                    "zone": r["zone"],
                    "speed_limit_kmh": limit,
                    "vehicle_count": len(speeds),
                    "avg_speed_kmh": avg_spd,
                    "congestion_index": congestion_index,
                }
            )
        return metrics

    # ------------------------------------------------------------------
    # Handoff & Mining Analytics
    # ------------------------------------------------------------------
    def get_recent_handoffs(self, limit: int = 40) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT h.*, v.plate, v.vehicle_type,
                       ft.name AS from_tower_name, tt.name AS to_tower_name
                FROM handoff_events h
                JOIN vehicles v ON h.vehicle_id = v.vehicle_id
                LEFT JOIN cell_towers ft ON h.from_tower_id = ft.tower_id
                JOIN cell_towers tt ON h.to_tower_id = tt.tower_id
                ORDER BY h.handoff_at DESC, h.handoff_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["time_str"] = datetime.fromtimestamp(d["handoff_at"]).strftime("%H:%M:%S")
            out.append(d)
        return out

    def get_tower_stats(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT t.*,
                       COUNT(v.vehicle_id) AS connected_vehicles,
                       ROUND(AVG(v.rssi_dbm), 1) AS avg_rssi_dbm
                FROM cell_towers t
                LEFT JOIN vehicles v ON v.current_tower_id = t.tower_id
                GROUP BY t.tower_id
                ORDER BY t.tower_id
                """
            ).fetchall()
        stats = []
        for r in rows:
            d = dict(r)
            d["utilization_pct"] = round(
                100.0 * d["connected_vehicles"] / max(1, d["capacity"]), 1
            )
            stats.append(d)
        return stats

    def get_handoff_summary(self) -> Dict[str, Any]:
        with self._lock:
            row = self.conn.execute(
                """
                SELECT COUNT(*) AS total_handoffs,
                       ROUND(AVG(latency_ms), 1) AS avg_latency_ms,
                       SUM(CASE WHEN reason = 'load_balance' THEN 1 ELSE 0 END) AS load_balance_count,
                       SUM(CASE WHEN reason = 'signal_threshold' THEN 1 ELSE 0 END) AS signal_threshold_count
                FROM handoff_events
                WHERE from_tower_id IS NOT NULL
                """
            ).fetchone()
        return dict(row) if row else {}

    def run_and_store_mining(
        self,
        timestamp: Optional[int] = None,
        eps_m: float = 290.0,
        min_samples: int = 4,
        speed_threshold_kmh: float = 18.0,
        persist: bool = True,
    ) -> Dict[str, Any]:
        """Run DBSCAN and Getis-Ord Gi* congestion mining on live or historical snapshot."""
        if timestamp is None:
            vehicles = self.get_all_vehicles()
            ts = int(time.time())
        else:
            vehicles = self.snapshot_at(timestamp)["vehicles"]
            ts = timestamp

        dbscan_clusters = detect_congestion_dbscan(
            vehicles,
            eps_m=eps_m,
            min_samples=min_samples,
            speed_threshold_kmh=speed_threshold_kmh,
        )
        bounds = self.get_city_bounds()
        gi_cells = compute_getis_ord_hotspots(vehicles, bounds=bounds, rows=10, cols=12)

        if persist and timestamp is None:
            with self._lock:
                self.conn.execute("DELETE FROM congestion_hotspots")
                for cl in dbscan_clusters:
                    self.conn.execute(
                        """
                        INSERT INTO congestion_hotspots (
                            algorithm, cluster_label, center_lat, center_lon, radius_m,
                            severity, avg_speed_kmh, vehicle_count, hull_json, zone, detected_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            "DBSCAN",
                            cl["cluster_label"],
                            cl["center_lat"],
                            cl["center_lon"],
                            cl["radius_m"],
                            cl["severity"],
                            cl["avg_speed_kmh"],
                            cl["vehicle_count"],
                            json.dumps(cl["hull"]),
                            cl["zone"],
                            ts,
                        ),
                    )
                self.conn.commit()

        bottlenecks = self.mine_recurring_bottlenecks(limit=8)
        return {
            "timestamp": ts,
            "dbscan_hotspots": dbscan_clusters,
            "getis_ord_cells": gi_cells,
            "recurring_bottlenecks": bottlenecks,
        }

    def mine_recurring_bottlenecks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Identify recurring historical road segment bottlenecks from trajectory logs."""
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT r.segment_id, r.name, r.road_class, r.zone, r.speed_limit_kmh,
                       COUNT(t.obs_id) AS obs_count,
                       ROUND(AVG(t.speed_kmh), 1) AS avg_speed_kmh,
                       ROUND(100.0 * SUM(CASE WHEN t.speed_kmh < 18.0 THEN 1 ELSE 0 END) / COUNT(t.obs_id), 1) AS congestion_pct
                FROM vehicle_trajectories t
                JOIN road_segments r ON t.segment_id = r.segment_id
                GROUP BY r.segment_id
                HAVING obs_count >= 5
                ORDER BY congestion_pct DESC, avg_speed_kmh ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            bottlenecks: List[Dict[str, Any]] = []
            for row in rows:
                d = dict(row)
                peak_row = self.conn.execute(
                    """
                    SELECT CAST(strftime('%H', recorded_at, 'unixepoch', 'localtime') AS INTEGER) AS hr,
                           COUNT(*) AS cnt,
                           AVG(speed_kmh) AS spd
                    FROM vehicle_trajectories
                    WHERE segment_id = ? AND speed_kmh < 18.0
                    GROUP BY hr
                    ORDER BY cnt DESC, spd ASC
                    LIMIT 1
                    """,
                    (d["segment_id"],),
                ).fetchone()
                d["peak_hour"] = (
                    f"{peak_row['hr']:02d}:00" if peak_row and peak_row["hr"] is not None else "N/A"
                )
                bottlenecks.append(d)
        return bottlenecks
