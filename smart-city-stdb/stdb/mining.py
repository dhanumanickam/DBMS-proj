"""Spatial Data Mining: DBSCAN Congestion Clustering, Getis-Ord Gi* Hotspots, and Pattern Mining."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

from .spatial_index import QuadtreeIndex, SpatialPoint
from .spatial_ops import haversine_m, pad_hull_if_degenerate


def detect_congestion_dbscan(
    observations: Sequence[Dict[str, Any]],
    eps_m: float = 280.0,
    min_samples: int = 4,
    speed_threshold_kmh: float = 18.0,
) -> List[Dict[str, Any]]:
    """Run Quadtree-accelerated Spatial DBSCAN on slow/congested vehicle observations."""
    congested = [
        obs for obs in observations if float(obs.get("speed_kmh", 0.0)) <= speed_threshold_kmh
    ]
    n = len(congested)
    if n < min_samples:
        return []

    pts = [
        SpatialPoint(str(i), float(obs["lat"]), float(obs["lon"]), obs)
        for i, obs in enumerate(congested)
    ]
    qt = QuadtreeIndex.from_points(pts, capacity=6, max_depth=9)

    labels = [-1] * n  # -1 = unvisited/noise, >= 0 = cluster id
    visited = [False] * n
    cluster_id = 0

    def region_query(idx: int) -> List[int]:
        p = pts[idx]
        hits, _ = qt.query_radius(p.lat, p.lon, eps_m)
        return [int(hit.item_id) for hit, _ in hits]

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = region_query(i)
        if len(neighbors) < min_samples:
            labels[i] = -1  # Noise
            continue

        labels[i] = cluster_id
        queue = list(neighbors)
        in_queue = set(neighbors)
        head = 0
        while head < len(queue):
            nb = queue[head]
            head += 1
            if not visited[nb]:
                visited[nb] = True
                nb_neighbors = region_query(nb)
                if len(nb_neighbors) >= min_samples:
                    for nxt in nb_neighbors:
                        if nxt not in in_queue:
                            queue.append(nxt)
                            in_queue.add(nxt)
            if labels[nb] == -1:
                labels[nb] = cluster_id
        cluster_id += 1

    clusters: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for idx, lbl in enumerate(labels):
        if lbl >= 0:
            clusters[lbl].append(congested[idx])

    results: List[Dict[str, Any]] = []
    for lbl, members in clusters.items():
        if len(members) < min_samples:
            continue
        c_lat = sum(float(m["lat"]) for m in members) / len(members)
        c_lon = sum(float(m["lon"]) for m in members) / len(members)
        avg_speed = sum(float(m["speed_kmh"]) for m in members) / len(members)
        max_dist = max(haversine_m(c_lat, c_lon, float(m["lat"]), float(m["lon"])) for m in members)
        radius_m = max(95.0, max_dist + 45.0)
        coords = [(float(m["lat"]), float(m["lon"])) for m in members]
        hull = pad_hull_if_degenerate(coords, min_pad_m=85.0)

        # Severity combines vehicle density and speed deficit
        speed_deficit = max(0.0, 1.0 - (avg_speed / max(1.0, speed_threshold_kmh)))
        density_factor = min(1.0, len(members) / 14.0)
        severity = round(min(1.0, 0.55 * speed_deficit + 0.45 * density_factor), 3)

        zones = [str(m.get("zone", "Downtown")) for m in members if m.get("zone")]
        dominant_zone = max(set(zones), key=zones.count) if zones else "City Core"

        results.append(
            {
                "algorithm": "DBSCAN",
                "cluster_label": int(lbl),
                "center_lat": round(c_lat, 6),
                "center_lon": round(c_lon, 6),
                "radius_m": round(radius_m, 1),
                "severity": severity,
                "avg_speed_kmh": round(avg_speed, 2),
                "vehicle_count": len(members),
                "hull": hull,
                "zone": dominant_zone,
                "vehicle_ids": [m.get("vehicle_id") for m in members if m.get("vehicle_id")],
            }
        )

    results.sort(key=lambda c: (c["severity"], c["vehicle_count"]), reverse=True)
    return results


def compute_getis_ord_hotspots(
    observations: Sequence[Dict[str, Any]],
    bounds: Tuple[float, float, float, float],
    rows: int = 10,
    cols: int = 12,
) -> List[Dict[str, Any]]:
    """Compute Getis-Ord Gi* local spatial autocorrelation z-scores on a city grid."""
    min_lat, max_lat, min_lon, max_lon = bounds
    d_lat = (max_lat - min_lat) / max(1, rows)
    d_lon = (max_lon - min_lon) / max(1, cols)

    grid_counts = [[0 for _ in range(cols)] for _ in range(rows)]
    grid_speed_sum = [[0.0 for _ in range(cols)] for _ in range(rows)]

    for obs in observations:
        lat = float(obs["lat"])
        lon = float(obs["lon"])
        if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
            continue
        r = min(rows - 1, max(0, int((lat - min_lat) / d_lat)))
        c = min(cols - 1, max(0, int((lon - min_lon) / d_lon)))
        grid_counts[r][c] += 1
        grid_speed_sum[r][c] += float(obs.get("speed_kmh", 35.0))

    # Congestion intensity x_j per cell = vehicle_count * (1 + slow_ratio)
    n_cells = rows * cols
    values: List[float] = []
    cell_meta: List[Tuple[int, int, int, float]] = []
    for r in range(rows):
        for c in range(cols):
            cnt = grid_counts[r][c]
            avg_spd = (grid_speed_sum[r][c] / cnt) if cnt > 0 else 45.0
            slow_weight = max(0.2, (50.0 - min(50.0, avg_spd)) / 50.0)
            intensity = cnt * (0.5 + 1.5 * slow_weight)
            values.append(intensity)
            cell_meta.append((r, c, cnt, avg_spd))

    if n_cells <= 1:
        return []

    mean_x = sum(values) / n_cells
    variance = sum((v - mean_x) ** 2 for v in values) / n_cells
    std_s = math.sqrt(variance)

    if std_s < 1e-9:
        return []

    results: List[Dict[str, Any]] = []
    for i, (r, c, cnt, avg_spd) in enumerate(cell_meta):
        # 3x3 Queen spatial neighborhood including self (w_ij = 1 for |dr|<=1, |dc|<=1)
        sum_wx = 0.0
        sum_w = 0.0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    j = nr * cols + nc
                    sum_wx += values[j]
                    sum_w += 1.0

        numerator = sum_wx - mean_x * sum_w
        denom_inner = ((n_cells * sum_w) - (sum_w ** 2)) / max(1, n_cells - 1)
        denom = std_s * math.sqrt(max(1e-9, denom_inner))
        z_score = numerator / denom

        cell_min_lat = min_lat + r * d_lat
        cell_max_lat = cell_min_lat + d_lat
        cell_min_lon = min_lon + c * d_lon
        cell_max_lon = cell_min_lon + d_lon

        if z_score >= 2.58:
            category = "hotspot_99"
        elif z_score >= 1.96:
            category = "hotspot_95"
        elif z_score <= -1.96:
            category = "coldspot"
        else:
            category = "neutral"

        if cnt > 0 or abs(z_score) >= 1.65:
            results.append(
                {
                    "row": r,
                    "col": c,
                    "min_lat": round(cell_min_lat, 6),
                    "max_lat": round(cell_max_lat, 6),
                    "min_lon": round(cell_min_lon, 6),
                    "max_lon": round(cell_max_lon, 6),
                    "center_lat": round(0.5 * (cell_min_lat + cell_max_lat), 6),
                    "center_lon": round(0.5 * (cell_min_lon + cell_max_lon), 6),
                    "vehicle_count": cnt,
                    "avg_speed_kmh": round(avg_spd, 1),
                    "intensity": round(values[i], 2),
                    "z_score": round(z_score, 3),
                    "category": category,
                }
            )

    results.sort(key=lambda item: item["z_score"], reverse=True)
    return results
