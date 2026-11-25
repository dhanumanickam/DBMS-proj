"""Spatial Indexing Engine: From-scratch Quadtree (with Best-First k-NN) & SQLite R-Tree helpers."""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .spatial_ops import haversine_m, radius_bbox


@dataclass
class SpatialPoint:
    item_id: str
    lat: float
    lon: float
    payload: Any = None


@dataclass
class QuadtreeNode:
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    depth: int = 0
    capacity: int = 8
    max_depth: int = 10
    points: List[SpatialPoint] = field(default_factory=list)
    children: Optional[List["QuadtreeNode"]] = None

    def contains(self, lat: float, lon: float) -> bool:
        return self.min_lat <= lat <= self.max_lat and self.min_lon <= lon <= self.max_lon

    def intersects_bbox(
        self, min_lat: float, max_lat: float, min_lon: float, max_lon: float
    ) -> bool:
        return not (
            max_lat < self.min_lat
            or min_lat > self.max_lat
            or max_lon < self.min_lon
            or min_lon > self.max_lon
        )

    def min_distance_m(self, lat: float, lon: float) -> float:
        """Lower-bound great-circle distance from (lat, lon) to this node's bounding box."""
        clamped_lat = min(max(lat, self.min_lat), self.max_lat)
        clamped_lon = min(max(lon, self.min_lon), self.max_lon)
        if clamped_lat == lat and clamped_lon == lon:
            return 0.0
        return haversine_m(lat, lon, clamped_lat, clamped_lon)

    def subdivide(self) -> None:
        mid_lat = 0.5 * (self.min_lat + self.max_lat)
        mid_lon = 0.5 * (self.min_lon + self.max_lon)
        nd = self.depth + 1
        self.children = [
            # SW, SE, NW, NE
            QuadtreeNode(self.min_lat, mid_lat, self.min_lon, mid_lon, nd, self.capacity, self.max_depth),
            QuadtreeNode(self.min_lat, mid_lat, mid_lon, self.max_lon, nd, self.capacity, self.max_depth),
            QuadtreeNode(mid_lat, self.max_lat, self.min_lon, mid_lon, nd, self.capacity, self.max_depth),
            QuadtreeNode(mid_lat, self.max_lat, mid_lon, self.max_lon, nd, self.capacity, self.max_depth),
        ]
        existing = self.points
        self.points = []
        for pt in existing:
            placed = False
            for child in self.children:
                if child.contains(pt.lat, pt.lon):
                    child.insert(pt)
                    placed = True
                    break
            if not placed:
                self.children[0].insert(pt)

    def insert(self, pt: SpatialPoint) -> bool:
        if not self.contains(pt.lat, pt.lon):
            return False
        if self.children is None:
            if len(self.points) < self.capacity or self.depth >= self.max_depth:
                self.points.append(pt)
                return True
            self.subdivide()
        assert self.children is not None
        for child in self.children:
            if child.contains(pt.lat, pt.lon):
                return child.insert(pt)
        return self.children[0].insert(pt)


class QuadtreeIndex:
    """In-memory Quadtree spatial index supporting range queries, BBox queries, and Best-First k-NN."""

    def __init__(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        capacity: int = 8,
        max_depth: int = 10,
    ) -> None:
        self.bounds = (min_lat, max_lat, min_lon, max_lon)
        self.capacity = capacity
        self.max_depth = max_depth
        self.root = QuadtreeNode(min_lat, max_lat, min_lon, max_lon, 0, capacity, max_depth)
        self.size = 0

    @classmethod
    def from_points(
        cls,
        points: Sequence[SpatialPoint],
        capacity: int = 8,
        max_depth: int = 10,
        padding: float = 0.002,
    ) -> "QuadtreeIndex":
        if not points:
            return cls(37.74, 37.81, -122.46, -122.38, capacity=capacity, max_depth=max_depth)
        min_lat = min(p.lat for p in points) - padding
        max_lat = max(p.lat for p in points) + padding
        min_lon = min(p.lon for p in points) - padding
        max_lon = max(p.lon for p in points) + padding
        qt = cls(min_lat, max_lat, min_lon, max_lon, capacity=capacity, max_depth=max_depth)
        for pt in points:
            qt.insert(pt.item_id, pt.lat, pt.lon, pt.payload)
        return qt

    def insert(self, item_id: str, lat: float, lon: float, payload: Any = None) -> bool:
        ok = self.root.insert(SpatialPoint(item_id, lat, lon, payload))
        if ok:
            self.size += 1
        return ok

    def query_bbox(
        self, min_lat: float, max_lat: float, min_lon: float, max_lon: float
    ) -> Tuple[List[SpatialPoint], int]:
        results: List[SpatialPoint] = []
        nodes_visited = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            nodes_visited += 1
            if not node.intersects_bbox(min_lat, max_lat, min_lon, max_lon):
                continue
            if node.children is None:
                for pt in node.points:
                    if min_lat <= pt.lat <= max_lat and min_lon <= pt.lon <= max_lon:
                        results.append(pt)
            else:
                stack.extend(node.children)
        return results, nodes_visited

    def query_radius(
        self, lat: float, lon: float, radius_m: float
    ) -> Tuple[List[Tuple[SpatialPoint, float]], int]:
        min_lat, max_lat, min_lon, max_lon = radius_bbox(lat, lon, radius_m)
        candidates, nodes_visited = self.query_bbox(min_lat, max_lat, min_lon, max_lon)
        hits: List[Tuple[SpatialPoint, float]] = []
        for pt in candidates:
            d = haversine_m(lat, lon, pt.lat, pt.lon)
            if d <= radius_m:
                hits.append((pt, d))
        hits.sort(key=lambda pair: pair[1])
        return hits, nodes_visited

    def knn(
        self, lat: float, lon: float, k: int = 5
    ) -> Tuple[List[Tuple[SpatialPoint, float]], int]:
        """Best-First Branch-and-Bound k-Nearest Neighbors search."""
        if k <= 0 or self.size == 0:
            return [], 0

        # Priority queue entries: (lower_bound_dist_m, tie_breaker, is_point, obj)
        pq: List[Tuple[float, int, int, Any]] = []
        counter = 0
        heapq.heappush(pq, (self.root.min_distance_m(lat, lon), counter, 0, self.root))
        results: List[Tuple[SpatialPoint, float]] = []
        nodes_visited = 0

        while pq and len(results) < k:
            dist, _, is_point, obj = heapq.heappop(pq)
            if is_point == 1:
                results.append((obj, dist))
            else:
                node: QuadtreeNode = obj
                nodes_visited += 1
                if node.children is None:
                    for pt in node.points:
                        counter += 1
                        d = haversine_m(lat, lon, pt.lat, pt.lon)
                        heapq.heappush(pq, (d, counter, 1, pt))
                else:
                    for child in node.children:
                        counter += 1
                        cd = child.min_distance_m(lat, lon)
                        heapq.heappush(pq, (cd, counter, 0, child))
        return results, nodes_visited

    def export_cells(self) -> List[Dict[str, Any]]:
        """Export leaf quadrants for visualization on the city map."""
        cells: List[Dict[str, Any]] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.children is None:
                cells.append(
                    {
                        "min_lat": node.min_lat,
                        "max_lat": node.max_lat,
                        "min_lon": node.min_lon,
                        "max_lon": node.max_lon,
                        "depth": node.depth,
                        "count": len(node.points),
                    }
                )
            else:
                stack.extend(node.children)
        return cells


def benchmark_spatial_indexes(
    points: Sequence[SpatialPoint],
    conn: Any,
    query_points: Sequence[Tuple[float, float]],
    radius_m: float = 450.0,
) -> Dict[str, Any]:
    """Compare Linear Scan vs. Quadtree vs. SQLite R-Tree on the same radius queries."""
    qt = QuadtreeIndex.from_points(points, capacity=8, max_depth=10)

    # 1. Linear Scan
    t0 = time.perf_counter()
    linear_counts: List[int] = []
    for q_lat, q_lon in query_points:
        cnt = 0
        for pt in points:
            if haversine_m(q_lat, q_lon, pt.lat, pt.lon) <= radius_m:
                cnt += 1
        linear_counts.append(cnt)
    linear_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Quadtree
    t1 = time.perf_counter()
    qt_counts: List[int] = []
    qt_nodes = 0
    for q_lat, q_lon in query_points:
        hits, visited = qt.query_radius(q_lat, q_lon, radius_m)
        qt_counts.append(len(hits))
        qt_nodes += visited
    qt_ms = (time.perf_counter() - t1) * 1000.0

    # 3. SQLite R-Tree
    t2 = time.perf_counter()
    rtree_counts: List[int] = []
    rtree_candidates = 0
    for q_lat, q_lon in query_points:
        min_lat, max_lat, min_lon, max_lon = radius_bbox(q_lat, q_lon, radius_m)
        rows = conn.execute(
            """
            SELECT v.vehicle_id, v.lat, v.lon
            FROM vehicles_rtree r
            JOIN vehicles_rtree_map m ON r.id = m.rtree_id
            JOIN vehicles v ON m.vehicle_id = v.vehicle_id
            WHERE r.min_lon <= ? AND r.max_lon >= ?
              AND r.min_lat <= ? AND r.max_lat >= ?
            """,
            (max_lon, min_lon, max_lat, min_lat),
        ).fetchall()
        rtree_candidates += len(rows)
        cnt = sum(1 for row in rows if haversine_m(q_lat, q_lon, row[1], row[2]) <= radius_m)
        rtree_counts.append(cnt)
    rtree_ms = (time.perf_counter() - t2) * 1000.0

    return {
        "dataset_size": len(points),
        "queries_executed": len(query_points),
        "radius_m": radius_m,
        "results_match": (linear_counts == qt_counts == rtree_counts),
        "total_hits": sum(linear_counts),
        "linear_scan": {
            "total_ms": round(linear_ms, 3),
            "avg_ms": round(linear_ms / max(1, len(query_points)), 4),
            "points_scanned": len(points) * len(query_points),
        },
        "quadtree": {
            "total_ms": round(qt_ms, 3),
            "avg_ms": round(qt_ms / max(1, len(query_points)), 4),
            "nodes_visited": qt_nodes,
            "leaf_cells": len(qt.export_cells()),
            "speedup_vs_linear": round(linear_ms / max(qt_ms, 1e-4), 2),
        },
        "sqlite_rtree": {
            "total_ms": round(rtree_ms, 3),
            "avg_ms": round(rtree_ms / max(1, len(query_points)), 4),
            "candidates_examined": rtree_candidates,
            "speedup_vs_linear": round(linear_ms / max(rtree_ms, 1e-4), 2),
        },
    }
