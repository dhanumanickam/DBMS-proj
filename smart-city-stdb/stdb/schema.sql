-- Spatio-Temporal Database Schema for Smart City Traffic & Mobility Management
-- Uses SQLite 3 with native R*Tree Virtual Tables for spatial indexing

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- 1. Road Network Nodes (Intersections)
CREATE TABLE IF NOT EXISTS intersections (
    node_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    zone TEXT NOT NULL,
    has_signal INTEGER NOT NULL DEFAULT 1,
    signal_cycle_sec INTEGER NOT NULL DEFAULT 60
);

-- 2. Road Network Segments (Directed / Undirected Links with WKT LineString)
CREATE TABLE IF NOT EXISTS road_segments (
    segment_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    road_class TEXT NOT NULL, -- 'arterial', 'highway', 'collector', 'bridge'
    from_node TEXT NOT NULL REFERENCES intersections(node_id),
    to_node TEXT NOT NULL REFERENCES intersections(node_id),
    wkt_geometry TEXT NOT NULL, -- LINESTRING(lon1 lat1, lon2 lat2, ...)
    coords_json TEXT NOT NULL,  -- [[lat, lon], ...]
    min_lat REAL NOT NULL,
    max_lat REAL NOT NULL,
    min_lon REAL NOT NULL,
    max_lon REAL NOT NULL,
    length_m REAL NOT NULL,
    speed_limit_kmh REAL NOT NULL,
    lanes INTEGER NOT NULL DEFAULT 2,
    zone TEXT NOT NULL
);

-- Spatial R-Tree Index for Road Segments
CREATE VIRTUAL TABLE IF NOT EXISTS roads_rtree USING rtree(
    id,
    min_lon, max_lon,
    min_lat, max_lat
);

CREATE TABLE IF NOT EXISTS roads_rtree_map (
    rtree_id INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id TEXT UNIQUE NOT NULL REFERENCES road_segments(segment_id) ON DELETE CASCADE
);

-- 3. Cell Towers / Roadside Units (RSUs) for Mobile Location & Handoff
CREATE TABLE IF NOT EXISTS cell_towers (
    tower_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    radius_m REAL NOT NULL,
    tx_power_dbm REAL NOT NULL DEFAULT 43.0,
    technology TEXT NOT NULL DEFAULT '5G-V2X',
    capacity INTEGER NOT NULL DEFAULT 60,
    zone TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS towers_rtree USING rtree(
    id,
    min_lon, max_lon,
    min_lat, max_lat
);

CREATE TABLE IF NOT EXISTS towers_rtree_map (
    rtree_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tower_id TEXT UNIQUE NOT NULL REFERENCES cell_towers(tower_id) ON DELETE CASCADE
);

-- 4. Current Vehicle & Mobile Device Locations
CREATE TABLE IF NOT EXISTS vehicles (
    vehicle_id TEXT PRIMARY KEY,
    plate TEXT NOT NULL,
    vehicle_type TEXT NOT NULL, -- 'car', 'bus', 'ev_taxi', 'delivery', 'emergency'
    device_id TEXT UNIQUE NOT NULL,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    heading_deg REAL NOT NULL DEFAULT 0.0,
    speed_kmh REAL NOT NULL DEFAULT 0.0,
    current_segment_id TEXT REFERENCES road_segments(segment_id),
    current_tower_id TEXT REFERENCES cell_towers(tower_id),
    rssi_dbm REAL NOT NULL DEFAULT -65.0,
    status TEXT NOT NULL DEFAULT 'moving', -- 'moving', 'congested', 'stopped'
    updated_at INTEGER NOT NULL -- Unix epoch seconds
);

-- Spatial R-Tree Index for Live Vehicle Positions
CREATE VIRTUAL TABLE IF NOT EXISTS vehicles_rtree USING rtree(
    id,
    min_lon, max_lon,
    min_lat, max_lat
);

CREATE TABLE IF NOT EXISTS vehicles_rtree_map (
    rtree_id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id TEXT UNIQUE NOT NULL REFERENCES vehicles(vehicle_id) ON DELETE CASCADE
);

-- 5. Temporal Trajectory Log (Historical Spatio-Temporal Observations)
CREATE TABLE IF NOT EXISTS vehicle_trajectories (
    obs_id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id TEXT NOT NULL REFERENCES vehicles(vehicle_id) ON DELETE CASCADE,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    speed_kmh REAL NOT NULL,
    heading_deg REAL NOT NULL,
    segment_id TEXT REFERENCES road_segments(segment_id),
    tower_id TEXT REFERENCES cell_towers(tower_id),
    rssi_dbm REAL,
    recorded_at INTEGER NOT NULL -- Unix epoch seconds
);

CREATE INDEX IF NOT EXISTS idx_traj_vehicle_time
    ON vehicle_trajectories(vehicle_id, recorded_at);
CREATE INDEX IF NOT EXISTS idx_traj_time_segment
    ON vehicle_trajectories(recorded_at, segment_id);
CREATE INDEX IF NOT EXISTS idx_traj_spatial_time
    ON vehicle_trajectories(recorded_at, lat, lon);

-- 6. Mobile Device Handoff Events
CREATE TABLE IF NOT EXISTS handoff_events (
    handoff_id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT NOT NULL,
    vehicle_id TEXT NOT NULL REFERENCES vehicles(vehicle_id) ON DELETE CASCADE,
    from_tower_id TEXT REFERENCES cell_towers(tower_id),
    to_tower_id TEXT NOT NULL REFERENCES cell_towers(tower_id),
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    rssi_from_dbm REAL,
    rssi_to_dbm REAL NOT NULL,
    latency_ms REAL NOT NULL,
    reason TEXT NOT NULL, -- 'signal_threshold', 'load_balance', 'initial_attach'
    handoff_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_handoff_time ON handoff_events(handoff_at DESC);
CREATE INDEX IF NOT EXISTS idx_handoff_device ON handoff_events(device_id, handoff_at DESC);

-- 7. Spatial Data Mining: Detected Congestion Hotspots
CREATE TABLE IF NOT EXISTS congestion_hotspots (
    hotspot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    algorithm TEXT NOT NULL, -- 'DBSCAN', 'GETIS_ORD_GI'
    cluster_label INTEGER NOT NULL,
    center_lat REAL NOT NULL,
    center_lon REAL NOT NULL,
    radius_m REAL NOT NULL,
    severity REAL NOT NULL, -- 0.0 to 1.0
    avg_speed_kmh REAL NOT NULL,
    vehicle_count INTEGER NOT NULL,
    hull_json TEXT NOT NULL, -- [[lat, lon], ...]
    zone TEXT NOT NULL,
    detected_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_hotspots_time ON congestion_hotspots(detected_at DESC);

-- 8. Active Traffic Incidents (for simulation / anomaly testing)
CREATE TABLE IF NOT EXISTS traffic_incidents (
    incident_id INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id TEXT NOT NULL REFERENCES road_segments(segment_id),
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    severity REAL NOT NULL, -- speed reduction factor e.g. 0.85
    description TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL
);
