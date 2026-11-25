"""Spatial geometry primitives, distance metrics, WKT helpers, and spatial operators."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

EARTH_RADIUS_M = 6_371_000.0
# Reference origin for our smart city local planar projection (approx. city center)
DEFAULT_REF_LAT = 37.7749
DEFAULT_REF_LON = -122.4194


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in meters between two WGS84 points."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return EARTH_RADIUS_M * c


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute initial compass bearing in degrees [0, 360) from point 1 to point 2."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_lon = math.radians(lon2 - lon1)

    y = math.sin(d_lon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lon)
    brng = math.degrees(math.atan2(y, x))
    return (brng + 360.0) % 360.0


def latlon_to_xy(
    lat: float,
    lon: float,
    ref_lat: float = DEFAULT_REF_LAT,
    ref_lon: float = DEFAULT_REF_LON,
) -> Tuple[float, float]:
    """Equirectangular local projection in meters relative to reference point."""
    cos_lat = math.cos(math.radians(ref_lat))
    x = math.radians(lon - ref_lon) * EARTH_RADIUS_M * cos_lat
    y = math.radians(lat - ref_lat) * EARTH_RADIUS_M
    return x, y


def xy_to_latlon(
    x: float,
    y: float,
    ref_lat: float = DEFAULT_REF_LAT,
    ref_lon: float = DEFAULT_REF_LON,
) -> Tuple[float, float]:
    """Inverse local equirectangular projection from meters back to WGS84 lat/lon."""
    cos_lat = math.cos(math.radians(ref_lat))
    lat = ref_lat + math.degrees(y / EARTH_RADIUS_M)
    lon = ref_lon + math.degrees(x / (EARTH_RADIUS_M * cos_lat))
    return lat, lon


def radius_bbox(lat: float, lon: float, radius_m: float) -> Tuple[float, float, float, float]:
    """Return (min_lat, max_lat, min_lon, max_lon) enclosing a circle of radius_m."""
    d_lat = math.degrees(radius_m / EARTH_RADIUS_M)
    cos_lat = max(1e-6, math.cos(math.radians(lat)))
    d_lon = math.degrees(radius_m / (EARTH_RADIUS_M * cos_lat))
    return lat - d_lat, lat + d_lat, lon - d_lon, lon + d_lon


def point_in_bbox(
    lat: float,
    lon: float,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
) -> bool:
    """Check if (lat, lon) falls inside axis-aligned bounding box."""
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def point_in_polygon(lat: float, lon: float, polygon: Sequence[Sequence[float]]) -> bool:
    """Ray-casting point-in-polygon test for polygon vertices [[lat, lon], ...]."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        yi, xi = polygon[i][0], polygon[i][1]
        yj, xj = polygon[j][0], polygon[j][1]
        intersects = ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / ((yj - yi) if (yj - yi) != 0 else 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def point_to_segment_distance_m(
    lat: float,
    lon: float,
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Shortest distance in meters from point (lat, lon) to line segment (1->2)."""
    px, py = latlon_to_xy(lat, lon, lat1, lon1)
    ax, ay = 0.0, 0.0
    bx, by = latlon_to_xy(lat2, lon2, lat1, lon1)
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-9:
        return math.hypot(px, py)
    t = max(0.0, min(1.0, (px * dx + py * dy) / seg_len_sq))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def point_to_polyline_distance_m(
    lat: float,
    lon: float,
    coords: Sequence[Sequence[float]],
) -> float:
    """Shortest distance in meters from point to a multi-vertex polyline."""
    if not coords:
        return float("inf")
    if len(coords) == 1:
        return haversine_m(lat, lon, coords[0][0], coords[0][1])
    best = float("inf")
    for i in range(len(coords) - 1):
        d = point_to_segment_distance_m(
            lat, lon, coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1]
        )
        if d < best:
            best = d
    return best


def to_wkt_linestring(coords: Sequence[Sequence[float]]) -> str:
    """Serialize [[lat, lon], ...] to OGC WKT LINESTRING(lon lat, ...)."""
    pts = ", ".join(f"{pt[1]:.6f} {pt[0]:.6f}" for pt in coords)
    return f"LINESTRING({pts})"


def to_wkt_point(lat: float, lon: float) -> str:
    """Serialize (lat, lon) to OGC WKT POINT(lon lat)."""
    return f"POINT({lon:.6f} {lat:.6f})"


def compute_bbox(coords: Iterable[Sequence[float]]) -> Tuple[float, float, float, float]:
    """Compute (min_lat, max_lat, min_lon, max_lon) for a sequence of [lat, lon]."""
    lats: List[float] = []
    lons: List[float] = []
    for c in coords:
        lats.append(c[0])
        lons.append(c[1])
    return min(lats), max(lats), min(lons), max(lons)


def convex_hull(points: Sequence[Tuple[float, float]]) -> List[List[float]]:
    """Compute 2D convex hull of [(lat, lon), ...] using Andrew's monotone chain.

    Returns vertices in counter-clockwise order as [[lat, lon], ...].
    """
    unique = sorted(set((round(p[0], 7), round(p[1], 7)) for p in points))
    if len(unique) <= 2:
        return [[p[0], p[1]] for p in unique]

    def cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return (a[1] - o[1]) * (b[0] - o[0]) - (a[0] - o[0]) * (b[1] - o[1])

    lower: List[Tuple[float, float]] = []
    for p in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[Tuple[float, float]] = []
    for p in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return [[p[0], p[1]] for p in hull]


def pad_hull_if_degenerate(
    points: Sequence[Tuple[float, float]], min_pad_m: float = 90.0
) -> List[List[float]]:
    """Return convex hull, or a small buffered octagon if fewer than 3 non-collinear points."""
    hull = convex_hull(points)
    if len(hull) >= 3:
        # Check non-zero area
        area = 0.0
        for i in range(len(hull)):
            j = (i + 1) % len(hull)
            area += hull[i][1] * hull[j][0] - hull[j][1] * hull[i][0]
        if abs(area) > 1e-11:
            return hull

    c_lat = sum(p[0] for p in points) / len(points)
    c_lon = sum(p[1] for p in points) / len(points)
    cx, cy = latlon_to_xy(c_lat, c_lon)
    ring: List[List[float]] = []
    for k in range(8):
        ang = 2.0 * math.pi * k / 8.0
        lat, lon = xy_to_latlon(cx + min_pad_m * math.cos(ang), cy + min_pad_m * math.sin(ang))
        ring.append([lat, lon])
    return ring
