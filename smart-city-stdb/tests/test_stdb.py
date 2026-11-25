"""Comprehensive unit & integration tests for all 7 Spatio-Temporal Database requirements."""

from __future__ import annotations

import io
import json
import unittest
from pathlib import Path

from server import AppContext, create_handler
from stdb.database import SpatioTemporalDB
from stdb.handoff import HandoffManager, TowerState
from stdb.mining import compute_getis_ord_hotspots, detect_congestion_dbscan
from stdb.simulator import CitySimulator
from stdb.spatial_index import QuadtreeIndex, SpatialPoint
from stdb.spatial_ops import (
    convex_hull,
    haversine_m,
    point_in_polygon,
    point_to_segment_distance_m,
    to_wkt_linestring,
)


class TestSpatioTemporalDB(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db = SpatioTemporalDB(":memory:")
        cls.sim = CitySimulator(cls.db, seed=123)
        cls.sim.seed_smart_city(num_vehicles=140)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.db.close()

    def test_01_spatial_schema_and_network_storage(self) -> None:
        """Req 1: Spatial database stores coordinates, WKT road networks, and vehicle locations."""
        net = self.db.get_network()
        self.assertGreaterEqual(len(net["intersections"]), 25)
        self.assertGreaterEqual(len(net["roads"]), 40)
        self.assertGreaterEqual(len(net["towers"]), 8)
        self.assertTrue(net["roads"][0]["wkt_geometry"].startswith("LINESTRING("))
        vehicles = self.db.get_all_vehicles()
        self.assertEqual(len(vehicles), 140)

    def test_02_spatial_operators_and_queries(self) -> None:
        """Req 2: Spatial operators (Haversine, PIP, segment dist, radius, k-NN, polygon)."""
        d = haversine_m(37.7749, -122.4194, 37.7749, -122.4194)
        self.assertAlmostEqual(d, 0.0, places=3)

        poly = [[37.768, -122.430], [37.768, -122.408], [37.782, -122.408], [37.782, -122.430]]
        self.assertTrue(point_in_polygon(37.775, -122.419, poly))
        self.assertFalse(point_in_polygon(37.800, -122.419, poly))

        seg_d = point_to_segment_distance_m(37.775, -122.419, 37.775, -122.425, 37.775, -122.415)
        self.assertLess(seg_d, 1.0)
        self.assertTrue(to_wkt_linestring([[37.77, -122.42], [37.78, -122.41]]).startswith("LINESTRING("))

        rad_res = self.db.query_vehicles_radius(37.7749, -122.4194, 900.0)
        self.assertGreater(rad_res["count"], 0)
        for v in rad_res["vehicles"]:
            self.assertLessEqual(v["distance_m"], 900.0)

        knn_res = self.db.query_vehicles_knn(37.7749, -122.4194, k=5)
        self.assertEqual(len(knn_res["vehicles"]), 5)
        dists = [v["distance_m"] for v in knn_res["vehicles"]]
        self.assertEqual(dists, sorted(dists))

        poly_res = self.db.query_vehicles_polygon(poly)
        self.assertGreater(poly_res["count"], 0)

        roads = self.db.query_roads_near(37.7749, -122.4194, 600.0)
        self.assertGreater(len(roads), 0)

    def test_03_temporal_history_and_trajectories(self) -> None:
        """Req 3: Maintain temporal logs & reconstruct snapshots/trajectories/hourly profiles."""
        bounds = self.db.get_time_bounds()
        self.assertIsNotNone(bounds["min_ts"])
        self.assertIsNotNone(bounds["max_ts"])
        self.assertGreater(bounds["observation_count"], 5000)

        mid_ts = (bounds["min_ts"] + bounds["max_ts"]) // 2
        snap = self.db.snapshot_at(mid_ts)
        self.assertGreater(snap["vehicle_count"], 100)

        traj = self.db.get_vehicle_trajectory("VEH_001")
        self.assertGreater(len(traj), 20)
        timestamps = [p["recorded_at"] for p in traj]
        self.assertEqual(timestamps, sorted(timestamps))

        st_win = self.db.query_spatiotemporal_window(
            37.765, 37.785, -122.435, -122.405, bounds["min_ts"], bounds["max_ts"]
        )
        self.assertGreater(st_win["observation_count"], 50)

        profile = self.db.get_hourly_traffic_profile(24)
        self.assertGreaterEqual(len(profile), 12)

    def test_04_spatial_indexing_rtree_and_quadtree_parity(self) -> None:
        """Req 4: Spatial indexing (Quadtree + SQLite R*Tree) matches brute-force exactly."""
        bench = self.db.run_index_benchmark(num_queries=25, radius_m=500.0)
        self.assertTrue(bench["results_match"])
        self.assertEqual(bench["dataset_size"], 140)
        self.assertGreater(bench["quadtree"]["leaf_cells"], 4)

        # Verify Best-First k-NN on Quadtree matches brute-force sorted distances
        vehicles = self.db.get_all_vehicles()
        pts = [SpatialPoint(v["vehicle_id"], v["lat"], v["lon"], v) for v in vehicles]
        qt = QuadtreeIndex.from_points(pts, capacity=6, max_depth=9)
        q_lat, q_lon = 37.775, -122.419
        qt_hits, _ = qt.knn(q_lat, q_lon, k=7)
        brute = sorted(pts, key=lambda p: haversine_m(q_lat, q_lon, p.lat, p.lon))[:7]
        self.assertEqual([p.item_id for p, _ in qt_hits], [p.item_id for p in brute])

    def test_05_mobile_location_and_handoff_management(self) -> None:
        """Req 5: Mobile device location & hysteresis-based cellular/V2X handoff."""
        t1 = TowerState("T1", "Tower 1", 37.770, -122.430, 800, 43.0, 50, "West", 0)
        t2 = TowerState("T2", "Tower 2", 37.770, -122.405, 800, 43.0, 50, "East", 0)
        mgr = HandoffManager([t1, t2])

        # Attach near T1
        d1 = mgr.evaluate_device("MOB_X", "VEH_X", 37.770, -122.429, None, speed_kmh=40.0)
        self.assertTrue(d1.handoff_occurred)
        self.assertEqual(d1.to_tower_id, "T1")
        self.assertEqual(d1.reason, "initial_attach")

        # Slightly closer to midpoint: hysteresis should hold T1 (no ping-pong)
        d2 = mgr.evaluate_device("MOB_X", "VEH_X", 37.770, -122.417, "T1", speed_kmh=40.0)
        self.assertFalse(d2.handoff_occurred)
        self.assertEqual(d2.to_tower_id, "T1")

        # Move right next to T2: handoff must trigger
        d3 = mgr.evaluate_device("MOB_X", "VEH_X", 37.770, -122.406, "T1", speed_kmh=45.0)
        self.assertTrue(d3.handoff_occurred)
        self.assertEqual(d3.from_tower_id, "T1")
        self.assertEqual(d3.to_tower_id, "T2")
        self.assertGreater(d3.latency_ms, 10.0)

        # Also verify simulator ticks generate handoffs in DB
        for _ in range(4):
            self.sim.tick(dt_sec=25.0)
        handoffs = self.db.get_recent_handoffs(limit=50)
        self.assertGreater(len(handoffs), 0)

    def test_06_spatial_data_mining_dbscan_and_getis_ord(self) -> None:
        """Req 6: Spatial data mining (DBSCAN clusters + Getis-Ord Gi* + recurring bottlenecks)."""
        # Synthetic dense slow cluster + scattered noise
        obs = [
            {"vehicle_id": f"V{i}", "lat": 37.775 + i * 0.0003, "lon": -122.419 + i * 0.0002, "speed_kmh": 8.0, "zone": "Core"}
            for i in range(6)
        ] + [
            {"vehicle_id": "NOISE_1", "lat": 37.795, "lon": -122.445, "speed_kmh": 9.0, "zone": "Edge"},
            {"vehicle_id": "FAST_1", "lat": 37.775, "lon": -122.419, "speed_kmh": 55.0, "zone": "Core"},
        ]
        clusters = detect_congestion_dbscan(obs, eps_m=220.0, min_samples=4, speed_threshold_kmh=18.0)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["vehicle_count"], 6)
        self.assertGreaterEqual(len(clusters[0]["hull"]), 3)

        hull = convex_hull([(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (0.4, 0.4)])
        self.assertEqual(len(hull), 4)

        gi = compute_getis_ord_hotspots(
            self.db.get_all_vehicles(), self.db.get_city_bounds(), rows=8, cols=10
        )
        self.assertGreater(len(gi), 0)
        self.assertTrue(any(c["z_score"] > 1.5 for c in gi))

        bottlenecks = self.db.mine_recurring_bottlenecks(limit=5)
        self.assertGreater(len(bottlenecks), 0)

    def test_07_rest_api_endpoints(self) -> None:
        """Req 7: Verify HTTP REST API serving map state, spatial queries, mining, and web UI."""
        tmp_db = Path("test_api_temp.sqlite")
        if tmp_db.exists():
            tmp_db.unlink()
        ctx = AppContext(tmp_db)
        handler_cls = create_handler(ctx)

        class _FakeSocket:
            def __init__(self, raw_req: bytes) -> None:
                self._r = io.BytesIO(raw_req)
                self._w = io.BytesIO()

            def makefile(self, mode: str, *args, **kwargs):
                return self._r if "r" in mode else self._w

            def sendall(self, data: bytes) -> None:
                self._w.write(data)

            def close(self) -> None:
                pass

        def dispatch(method: str, path: str, body: bytes = b""):
            headers = f"Host: localhost\r\nContent-Length: {len(body)}\r\n"
            if body:
                headers += "Content-Type: application/json\r\n"
            raw_req = f"{method} {path} HTTP/1.1\r\n{headers}\r\n".encode("utf-8") + body
            sock = _FakeSocket(raw_req)
            handler_cls(sock, ("127.0.0.1", 0), None)
            raw_resp = sock._w.getvalue()
            head, payload = raw_resp.split(b"\r\n\r\n", 1)
            status_code = int(head.decode("utf-8").splitlines()[0].split()[1])
            return status_code, payload

        try:
            code, body = dispatch("GET", "/api/health")
            self.assertEqual(code, 200)
            self.assertEqual(json.loads(body)["status"], "ok")

            code, body = dispatch("GET", "/api/state")
            self.assertEqual(code, 200)
            st = json.loads(body)
            self.assertEqual(st["mode"], "live")
            self.assertGreater(st["vehicle_count"], 100)

            code, body = dispatch("GET", "/api/spatial/radius?lat=37.7749&lon=-122.4194&radius_m=600")
            self.assertEqual(code, 200)
            self.assertIn("vehicles", json.loads(body))

            code, body = dispatch("GET", "/api/mining")
            self.assertEqual(code, 200)
            mining = json.loads(body)
            self.assertIn("dbscan_hotspots", mining)
            self.assertIn("getis_ord_cells", mining)

            code, body = dispatch("GET", "/")
            self.assertEqual(code, 200)
            self.assertIn("MetroPulse STDB", body.decode("utf-8"))

            code, body = dispatch("POST", "/api/sim/tick", b'{"dt_sec": 5}')
            self.assertEqual(code, 200)
            self.assertGreater(json.loads(body)["vehicles_updated"], 100)
        finally:
            ctx.db.close()
            for suffix in ("", "-wal", "-shm"):
                p = Path(f"{tmp_db}{suffix}")
                if p.exists():
                    p.unlink()


if __name__ == "__main__":
    unittest.main(verbosity=2)
