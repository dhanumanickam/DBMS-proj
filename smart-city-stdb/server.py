"""HTTP REST API & Static Web Server for the Smart City Spatio-Temporal Database."""

from __future__ import annotations

import json
import mimetypes
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from stdb.database import SpatioTemporalDB
from stdb.simulator import CitySimulator

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
DATA_DIR = BASE_DIR / "data"


class AppContext:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = SpatioTemporalDB(db_path)
        self.sim = CitySimulator(self.db, seed=42)
        self.sim_lock = threading.Lock()
        self.sim.load_or_seed(num_vehicles=220)


def create_handler(ctx: AppContext):
    class STDBHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            # Keep server output concise
            pass

        def _send_json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))

        def _serve_static(self, rel_path: str) -> None:
            clean = rel_path.lstrip("/") or "index.html"
            target = (WEB_DIR / clean).resolve()
            if not str(target).startswith(str(WEB_DIR.resolve())) or not target.is_file():
                self._send_json({"error": "Not found"}, status=404)
                return
            mime, _ = mimetypes.guess_type(str(target))
            data = target.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"{mime or 'application/octet-stream'}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self) -> None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)

            try:
                if path == "/api/health":
                    bounds = ctx.db.get_time_bounds()
                    self._send_json({"status": "ok", "time_bounds": bounds})
                    return

                if path == "/api/network":
                    self._send_json(ctx.db.get_network())
                    return

                if path == "/api/state":
                    ts_param = qs.get("ts", [None])[0]
                    ts: Optional[int] = int(ts_param) if ts_param else None
                    if ts is None:
                        vehicles = ctx.db.get_all_vehicles()
                        avg_spd = (
                            round(sum(v["speed_kmh"] for v in vehicles) / len(vehicles), 1)
                            if vehicles
                            else 0.0
                        )
                        congested = sum(1 for v in vehicles if v["speed_kmh"] < 18.0)
                        sim_ts = ctx.sim.sim_time
                        mode = "live"
                    else:
                        snap = ctx.db.snapshot_at(ts)
                        vehicles = snap["vehicles"]
                        avg_spd = snap["avg_speed_kmh"]
                        congested = snap["congested_count"]
                        sim_ts = ts
                        mode = "historical"

                    seg_metrics = ctx.db.get_segment_live_metrics(timestamp=ts)
                    bounds = ctx.db.get_time_bounds()
                    incidents = ctx.db.get_active_incidents(now_ts=sim_ts)
                    handoff_summary = ctx.db.get_handoff_summary()

                    self._send_json(
                        {
                            "mode": mode,
                            "timestamp": sim_ts,
                            "vehicle_count": len(vehicles),
                            "avg_speed_kmh": avg_spd,
                            "congested_count": congested,
                            "vehicles": vehicles,
                            "segment_metrics": seg_metrics,
                            "incidents": incidents,
                            "time_bounds": bounds,
                            "handoff_summary": handoff_summary,
                        }
                    )
                    return

                if path == "/api/spatial/radius":
                    lat = float(qs["lat"][0])
                    lon = float(qs["lon"][0])
                    radius_m = float(qs.get("radius_m", ["450"])[0])
                    self._send_json(ctx.db.query_vehicles_radius(lat, lon, radius_m))
                    return

                if path == "/api/spatial/knn":
                    lat = float(qs["lat"][0])
                    lon = float(qs["lon"][0])
                    k = int(qs.get("k", ["6"])[0])
                    self._send_json(ctx.db.query_vehicles_knn(lat, lon, k=k))
                    return

                if path == "/api/spatial/roads":
                    lat = float(qs["lat"][0])
                    lon = float(qs["lon"][0])
                    radius_m = float(qs.get("radius_m", ["350"])[0])
                    self._send_json({"roads": ctx.db.query_roads_near(lat, lon, radius_m)})
                    return

                if path == "/api/spatial/quadtree":
                    self._send_json(ctx.db.get_quadtree_cells())
                    return

                if path == "/api/spatial/benchmark":
                    num_q = int(qs.get("queries", ["50"])[0])
                    radius_m = float(qs.get("radius_m", ["450"])[0])
                    self._send_json(ctx.db.run_index_benchmark(num_queries=num_q, radius_m=radius_m))
                    return

                if path == "/api/temporal/trajectory":
                    vid = qs["vehicle_id"][0]
                    start_ts = int(qs["start_ts"][0]) if "start_ts" in qs else None
                    end_ts = int(qs["end_ts"][0]) if "end_ts" in qs else None
                    pts = ctx.db.get_vehicle_trajectory(vid, start_ts=start_ts, end_ts=end_ts)
                    self._send_json({"vehicle_id": vid, "count": len(pts), "points": pts})
                    return

                if path == "/api/temporal/profile":
                    self._send_json({"hours": ctx.db.get_hourly_traffic_profile(24)})
                    return

                if path == "/api/temporal/window":
                    min_lat = float(qs["min_lat"][0])
                    max_lat = float(qs["max_lat"][0])
                    min_lon = float(qs["min_lon"][0])
                    max_lon = float(qs["max_lon"][0])
                    start_ts = int(qs["start_ts"][0])
                    end_ts = int(qs["end_ts"][0])
                    self._send_json(
                        ctx.db.query_spatiotemporal_window(
                            min_lat, max_lat, min_lon, max_lon, start_ts, end_ts
                        )
                    )
                    return

                if path == "/api/handoffs":
                    limit = int(qs.get("limit", ["35"])[0])
                    self._send_json(
                        {
                            "summary": ctx.db.get_handoff_summary(),
                            "towers": ctx.db.get_tower_stats(),
                            "events": ctx.db.get_recent_handoffs(limit=limit),
                        }
                    )
                    return

                if path == "/api/mining":
                    ts_param = qs.get("ts", [None])[0]
                    ts = int(ts_param) if ts_param else None
                    eps_m = float(qs.get("eps_m", ["290"])[0])
                    min_samples = int(qs.get("min_samples", ["4"])[0])
                    speed_thr = float(qs.get("speed_threshold", ["18"])[0])
                    self._send_json(
                        ctx.db.run_and_store_mining(
                            timestamp=ts,
                            eps_m=eps_m,
                            min_samples=min_samples,
                            speed_threshold_kmh=speed_thr,
                            persist=(ts is None),
                        )
                    )
                    return

                if not path.startswith("/api/"):
                    self._serve_static(path)
                    return

                self._send_json({"error": f"Unknown endpoint: {path}"}, status=404)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            try:
                body = self._read_json_body()

                if path == "/api/sim/tick":
                    dt_sec = float(body.get("dt_sec", 8.0))
                    with ctx.sim_lock:
                        res = ctx.sim.tick(dt_sec=dt_sec)
                    self._send_json(res)
                    return

                if path == "/api/sim/incident":
                    with ctx.sim_lock:
                        seg_id = body.get("segment_id")
                        if not seg_id:
                            # Pick an arterial/highway segment
                            candidates = [
                                s["segment_id"]
                                for s in ctx.sim.segments.values()
                                if s["road_class"] in ("arterial", "bridge")
                            ]
                            seg_id = ctx.sim.rng.choice(candidates or list(ctx.sim.segments.keys()))
                        inc = ctx.db.add_incident(
                            segment_id=seg_id,
                            severity=float(body.get("severity", 0.85)),
                            description=str(body.get("description", "Reported traffic incident / lane closure")),
                            duration_sec=int(body.get("duration_sec", 240)),
                            now_ts=ctx.sim.sim_time,
                        )
                        ctx.sim.tick(dt_sec=6.0)
                    self._send_json(inc)
                    return

                if path == "/api/sim/reset":
                    with ctx.sim_lock:
                        ctx.sim.seed_smart_city(num_vehicles=int(body.get("num_vehicles", 220)))
                    self._send_json({"status": "reset_complete", "timestamp": ctx.sim.sim_time})
                    return

                if path == "/api/spatial/polygon":
                    polygon = body.get("polygon", [])
                    self._send_json(ctx.db.query_vehicles_polygon(polygon))
                    return

                self._send_json({"error": f"Unknown endpoint: {path}"}, status=404)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)

    return STDBHandler


def run_server(host: str = "127.0.0.1", port: int = 8090) -> None:
    db_path = DATA_DIR / "smart_city_stdb.sqlite"
    print(f"[STDB] Initializing Spatio-Temporal Database at {db_path} ...")
    t0 = time.perf_counter()
    ctx = AppContext(db_path)
    print(f"[STDB] Ready in {(time.perf_counter() - t0):.2f}s. Vehicles: {len(ctx.sim.vehicles)}")
    server = ThreadingHTTPServer((host, port), create_handler(ctx))
    print(f"[STDB] Dashboard & API listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[STDB] Shutting down.")
        server.server_close()
        ctx.db.close()


if __name__ == "__main__":
    run_server()
