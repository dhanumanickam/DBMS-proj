# MetroPulse STDB — Spatio-Temporal Database System for Smart City Traffic & Mobility Management

A full-stack, zero-external-dependency Spatio-Temporal Database (STDB) application built with Python 3, SQLite 3 (native R*Tree spatial index), custom spatial indexing & data mining engines, and an interactive real-time + historical map dashboard.

## Architecture & Key Capabilities

1. **Spatial Database Design (`stdb/schema.sql`, `stdb/database.py`)**
   - Stores WGS84 intersection coordinates (`intersections`), WKT `LINESTRING` road network segments with bounding boxes (`road_segments`), 5G/V2X cell towers (`cell_towers`), and live vehicle/device states (`vehicles`).
   - Automatically synchronizes three SQLite `rtree` virtual tables (`vehicles_rtree`, `roads_rtree`, `towers_rtree`).

2. **Spatial Operators & Queries (`stdb/spatial_ops.py`, `stdb/database.py`)**
   - Great-circle `haversine_m`, compass `bearing_deg`, local equirectangular projection, `point_in_polygon` (ray casting), `point_to_polyline_distance_m`, and Andrew's monotone chain `convex_hull`.
   - Implements `ST_DWithin` radius queries (`query_vehicles_radius`), Best-First `ST_KNN` (`query_vehicles_knn`), `ST_Within` polygon containment (`query_vehicles_polygon`), and nearest road segment spatial joins (`query_roads_near`).

3. **Temporal Information & Historical Analysis (`stdb/database.py`, `stdb/simulator.py`)**
   - Maintains time-indexed `vehicle_trajectories` (24-hour historical seed + live updates).
   - Supports point-in-time snapshot reconstruction (`snapshot_at`), individual vehicle trajectory extraction (`get_vehicle_trajectory`), spatio-temporal window queries (`query_spatiotemporal_window`), and 24-hour hourly speed/congestion profiling (`get_hourly_traffic_profile`).

4. **Spatial Indexing (`stdb/spatial_index.py`)**
   - Dual indexing engine: persistent **SQLite R*Tree** virtual tables + from-scratch **In-Memory Quadtree** (`QuadtreeIndex`) supporting BBox, radius, and Best-First Branch-and-Bound $k$-NN searches.
   - Includes a live benchmark comparing Linear Scan vs. Quadtree vs. SQLite R*Tree (`benchmark_spatial_indexes`).

5. **Mobile Location & Handoff Management (`stdb/handoff.py`)**
   - Urban log-distance RSSI path-loss model (`compute_rssi_dbm`) across 10 5G/V2X cell towers.
   - Hysteresis (`4.5 dBm`) and load-balancing handoff controller (`HandoffManager`) preventing ping-pong handoffs while logging `handoff_events` with signal delta and latency (ms).

6. **Spatial Data Mining (`stdb/mining.py`)**
   - Quadtree-accelerated **Spatial DBSCAN** (`detect_congestion_dbscan`) to cluster congested vehicles and extract severity-scored convex hulls.
   - **Getis-Ord $G_i^*$** spatial autocorrelation grid analysis (`compute_getis_ord_hotspots`) to detect statistically significant traffic hot spots ($z \ge 1.96$).
   - Historical recurring bottleneck ranking (`mine_recurring_bottlenecks`) with peak congestion hour detection.

7. **Interactive Map & Dashboard (`server.py`, `web/`)**
   - Live pan/zoom city map showing road congestion colors, directional vehicle headings, 5G-V2X coverage rings, Quadtree grid partitions, DBSCAN convex hulls, Getis-Ord $G_i^*$ heatmap, trajectory trails, 24h timeline playback scrubber, index benchmarks, and mobile handoff stream.

## Quick Start

```bash
# Run the test suite
python -m unittest discover -s tests -v

# Start the server & interactive dashboard (http://127.0.0.1:8090)
python server.py
```
