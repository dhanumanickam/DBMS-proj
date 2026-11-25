/* MetroPulse STDB — Interactive Spatio-Temporal Smart City Frontend */
(() => {
  const state = {
    network: null,
    bounds: null,
    vehicles: [],
    segmentMetrics: {},
    incidents: [],
    dbscanHotspots: [],
    giCells: [],
    bottlenecks: [],
    quadtreeCells: [],
    handoffs: [],
    towers: [],
    hourlyProfile: [],
    timeBounds: { min_ts: null, max_ts: null, observation_count: 0 },

    mode: "live", // "live" | "historical"
    historicalTs: null,
    liveRunning: true,
    histPlaying: false,

    activeTool: "inspect", // "inspect" | "radius" | "knn"
    selectedVehicle: null,
    selectedTrajectory: [],
    spatialHighlight: null, // { type: "radius"|"knn", lat, lon, radius_m, vehicles, meta }

    // Map viewport transform
    zoom: 1.0,
    panX: 0,
    panY: 0,
    dragging: false,
    dragStartX: 0,
    dragStartY: 0,
    dragMoved: false,
  };

  const canvas = document.getElementById("city-canvas");
  const ctx = canvas.getContext("2d");

  async function apiGet(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  async function apiPost(url, body = {}) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  function formatTime(ts) {
    if (!ts) return "--:--:--";
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function formatShortTime(ts) {
    if (!ts) return "--:--";
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  // Coordinate projection <-> Canvas pixels
  function getMapScale() {
    const w = canvas.clientWidth || 800;
    const h = canvas.clientHeight || 600;
    const b = state.bounds || { min_lat: 37.752, max_lat: 37.798, min_lon: -122.452, max_lon: -122.388 };
    const pad = 42;
    const dLat = Math.max(1e-5, b.max_lat - b.min_lat);
    const dLon = Math.max(1e-5, b.max_lon - b.min_lon);
    const cosLat = Math.cos(((b.min_lat + b.max_lat) * 0.5 * Math.PI) / 180);
    const lonSpan = dLon * cosLat;
    const scale = Math.min((w - 2 * pad) / lonSpan, (h - 2 * pad) / dLat) * state.zoom;
    const cx = w * 0.5 + state.panX;
    const cy = h * 0.5 + state.panY;
    const midLat = 0.5 * (b.min_lat + b.max_lat);
    const midLon = 0.5 * (b.min_lon + b.max_lon);
    return { scale, cx, cy, midLat, midLon, cosLat };
  }

  function latLonToPx(lat, lon) {
    const { scale, cx, cy, midLat, midLon, cosLat } = getMapScale();
    const x = cx + (lon - midLon) * cosLat * scale;
    const y = cy - (lat - midLat) * scale;
    return [x, y];
  }

  function pxToLatLon(x, y) {
    const { scale, cx, cy, midLat, midLon, cosLat } = getMapScale();
    const lon = midLon + (x - cx) / (scale * cosLat);
    const lat = midLat - (y - cy) / scale;
    return [lat, lon];
  }

  function metersToPx(meters) {
    const { scale } = getMapScale();
    const degPerMeter = 1 / 111320;
    return meters * degPerMeter * scale;
  }

  function speedColor(spd) {
    if (spd < 18) return "#ef4444";
    if (spd < 35) return "#f59e0b";
    return "#22c55e";
  }

  // ---------------------------------------------------------------
  // Canvas Rendering
  // ---------------------------------------------------------------
  function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    renderMap();
  }

  function renderMap() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    ctx.clearRect(0, 0, w, h);

    if (!state.network) return;

    // 1. Subtle background grid
    ctx.strokeStyle = "rgba(56, 189, 248, 0.04)";
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 60) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 0; y < h; y += 60) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // 2. Getis-Ord Gi* Grid Heatmap Layer
    if (document.getElementById("layer-gi").checked && state.giCells.length) {
      for (const cell of state.giCells) {
        const [x1, y1] = latLonToPx(cell.max_lat, cell.min_lon);
        const [x2, y2] = latLonToPx(cell.min_lat, cell.max_lon);
        let fill = "rgba(56, 189, 248, 0.05)";
        if (cell.z_score >= 2.58) fill = "rgba(239, 68, 68, 0.36)";
        else if (cell.z_score >= 1.96) fill = "rgba(249, 115, 22, 0.28)";
        else if (cell.z_score >= 1.0) fill = "rgba(245, 158, 11, 0.14)";
        else if (cell.z_score <= -1.96) fill = "rgba(34, 197, 94, 0.12)";
        ctx.fillStyle = fill;
        ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
        ctx.strokeStyle = "rgba(148, 163, 184, 0.12)";
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      }
    }

    // 3. Quadtree Spatial Index Partition Overlay
    if (document.getElementById("layer-quadtree").checked && state.quadtreeCells.length) {
      ctx.lineWidth = 1;
      for (const cell of state.quadtreeCells) {
        const [x1, y1] = latLonToPx(cell.max_lat, cell.min_lon);
        const [x2, y2] = latLonToPx(cell.min_lat, cell.max_lon);
        ctx.strokeStyle = "rgba(56, 189, 248, 0.38)";
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      }
    }

    // 4. Road Network Layer (colored by congestion index)
    if (document.getElementById("layer-roads").checked) {
      for (const road of state.network.roads) {
        const metric = state.segmentMetrics[road.segment_id];
        const cong = metric ? metric.congestion_index : 0.1;
        let stroke = "#25406b";
        if (metric && metric.vehicle_count > 0) {
          if (cong >= 0.6) stroke = "#b91c1c";
          else if (cong >= 0.35) stroke = "#d97706";
          else stroke = "#15803d";
        }
        const width = road.road_class === "highway" || road.road_class === "bridge" ? 4.5 : road.road_class === "arterial" ? 3.2 : 2.0;

        ctx.beginPath();
        road.coords.forEach((pt, idx) => {
          const [px, py] = latLonToPx(pt[0], pt[1]);
          if (idx === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.strokeStyle = stroke;
        ctx.lineWidth = width;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.stroke();
      }

      // Intersections
      ctx.fillStyle = "#475569";
      for (const node of state.network.intersections) {
        const [px, py] = latLonToPx(node.lat, node.lon);
        ctx.beginPath();
        ctx.arc(px, py, 2.4, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // 5. DBSCAN Congestion Hotspot Convex Hulls
    if (document.getElementById("layer-dbscan").checked && state.dbscanHotspots.length) {
      for (const hs of state.dbscanHotspots) {
        if (!hs.hull || hs.hull.length < 3) continue;
        ctx.beginPath();
        hs.hull.forEach((pt, idx) => {
          const [px, py] = latLonToPx(pt[0], pt[1]);
          if (idx === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.closePath();
        ctx.fillStyle = "rgba(239, 68, 68, 0.20)";
        ctx.fill();
        ctx.strokeStyle = "rgba(248, 113, 113, 0.85)";
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 4]);
        ctx.stroke();
        ctx.setLineDash([]);

        const [cx, cy] = latLonToPx(hs.center_lat, hs.center_lon);
        ctx.fillStyle = "rgba(15, 23, 42, 0.86)";
        ctx.fillRect(cx - 46, cy - 11, 92, 18);
        ctx.fillStyle = "#fca5a5";
        ctx.font = "600 10px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(`HOTSPOT (${hs.vehicle_count}v)`, cx, cy + 2);
      }
    }

    // 6. Active Traffic Incidents
    for (const inc of state.incidents) {
      const [px, py] = latLonToPx(inc.lat, inc.lon);
      ctx.beginPath();
      ctx.arc(px, py, 10, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(239, 68, 68, 0.25)";
      ctx.fill();
      ctx.strokeStyle = "#ef4444";
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = "#fff";
      ctx.font = "bold 11px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("!", px, py + 4);
    }

    // 7. 5G-V2X Cell Towers
    if (document.getElementById("layer-towers").checked) {
      for (const twr of state.network.towers) {
        const [tx, ty] = latLonToPx(twr.lat, twr.lon);
        const radPx = metersToPx(twr.radius_m);

        ctx.beginPath();
        ctx.arc(tx, ty, radPx, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(167, 139, 250, 0.035)";
        ctx.fill();
        ctx.strokeStyle = "rgba(167, 139, 250, 0.22)";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);

        // Tower antenna icon
        ctx.fillStyle = "#a78bfa";
        ctx.beginPath();
        ctx.moveTo(tx, ty - 7);
        ctx.lineTo(tx - 5, ty + 5);
        ctx.lineTo(tx + 5, ty + 5);
        ctx.closePath();
        ctx.fill();

        ctx.fillStyle = "#c4b5fd";
        ctx.font = "600 9.5px monospace";
        ctx.textAlign = "center";
        ctx.fillText(twr.tower_id, tx, ty - 10);
      }
    }

    // 8. Selected Vehicle Trajectory Trail & Serving Tower Link
    if (state.selectedVehicle && state.selectedTrajectory.length > 1) {
      ctx.beginPath();
      state.selectedTrajectory.forEach((pt, idx) => {
        const [px, py] = latLonToPx(pt.lat, pt.lon);
        if (idx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 2.6;
      ctx.stroke();
    }
    if (state.selectedVehicle && state.selectedVehicle.current_tower_id && state.network) {
      const twr = state.network.towers.find((t) => t.tower_id === state.selectedVehicle.current_tower_id);
      if (twr) {
        const [vx, vy] = latLonToPx(state.selectedVehicle.lat, state.selectedVehicle.lon);
        const [tx, ty] = latLonToPx(twr.lat, twr.lon);
        ctx.beginPath();
        ctx.moveTo(vx, vy);
        ctx.lineTo(tx, ty);
        ctx.strokeStyle = "#c084fc";
        ctx.lineWidth = 1.6;
        ctx.setLineDash([4, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    // 9. Spatial Query Visual Overlay (Radius / k-NN)
    const highlightedIds = new Set();
    if (state.spatialHighlight) {
      const sh = state.spatialHighlight;
      const [qx, qy] = latLonToPx(sh.lat, sh.lon);
      if (sh.type === "radius") {
        const rPx = metersToPx(sh.radius_m);
        ctx.beginPath();
        ctx.arc(qx, qy, rPx, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(56, 189, 248, 0.11)";
        ctx.fill();
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2;
        ctx.stroke();
      } else if (sh.type === "knn") {
        for (const v of sh.vehicles) {
          const [vx, vy] = latLonToPx(v.lat, v.lon);
          ctx.beginPath();
          ctx.moveTo(qx, qy);
          ctx.lineTo(vx, vy);
          ctx.strokeStyle = "rgba(56, 189, 248, 0.75)";
          ctx.lineWidth = 1.6;
          ctx.setLineDash([4, 3]);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
      ctx.fillStyle = "#38bdf8";
      ctx.beginPath();
      ctx.arc(qx, qy, 5, 0, Math.PI * 2);
      ctx.fill();

      for (const v of sh.vehicles) highlightedIds.add(v.vehicle_id);
    }

    // 10. Vehicles (directional chevrons/dots)
    for (const v of state.vehicles) {
      const [vx, vy] = latLonToPx(v.lat, v.lon);
      const isSelected = state.selectedVehicle && state.selectedVehicle.vehicle_id === v.vehicle_id;
      const isHi = highlightedIds.has(v.vehicle_id);
      const color = speedColor(v.speed_kmh);

      if (isSelected || isHi) {
        ctx.beginPath();
        ctx.arc(vx, vy, isSelected ? 10 : 8, 0, Math.PI * 2);
        ctx.strokeStyle = isSelected ? "#ffffff" : "#38bdf8";
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      const rad = ((v.heading_deg - 90) * Math.PI) / 180;
      const sz = v.vehicle_type === "bus" ? 5.2 : 4.2;
      ctx.save();
      ctx.translate(vx, vy);
      ctx.rotate(rad);
      ctx.beginPath();
      ctx.moveTo(sz + 1.5, 0);
      ctx.lineTo(-sz, -sz * 0.75);
      ctx.lineTo(-sz * 0.5, 0);
      ctx.lineTo(-sz, sz * 0.75);
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
      ctx.restore();
    }
  }

  // ---------------------------------------------------------------
  // Data Refresh & UI Updaters
  // ---------------------------------------------------------------
  async function refreshState() {
    const tsParam = state.mode === "historical" && state.historicalTs ? `?ts=${state.historicalTs}` : "";
    const data = await apiGet(`/api/state${tsParam}`);
    state.vehicles = data.vehicles;
    state.incidents = data.incidents || [];
    state.timeBounds = data.time_bounds || state.timeBounds;
    state.segmentMetrics = {};
    for (const m of data.segment_metrics || []) {
      state.segmentMetrics[m.segment_id] = m;
    }

    // Update KPI strip
    document.getElementById("kpi-vehicles").textContent = data.vehicle_count;
    document.getElementById("kpi-obs").textContent = `${(state.timeBounds.observation_count || 0).toLocaleString()} trajectory logs`;
    document.getElementById("kpi-speed").textContent = `${data.avg_speed_kmh} km/h`;
    document.getElementById("kpi-congested").textContent = `${data.congested_count} congested (<18 km/h)`;
    document.getElementById("kpi-incidents").textContent = `${state.incidents.length} active incidents`;

    const hs = data.handoff_summary || {};
    document.getElementById("kpi-handoffs").textContent = hs.total_handoffs ?? 0;
    document.getElementById("kpi-latency").textContent = `avg ${hs.avg_latency_ms ?? 0} ms latency`;

    document.getElementById("clock-display").textContent = formatTime(data.timestamp);
    const pill = document.getElementById("mode-pill");
    if (state.mode === "live") {
      pill.textContent = state.liveRunning ? "● LIVE STREAM" : "❚❚ LIVE PAUSED";
      pill.className = "pill pill-live";
    } else {
      pill.textContent = "◷ HISTORICAL";
      pill.className = "pill pill-hist";
    }

    // Update timeline bounds
    if (state.timeBounds.min_ts && state.timeBounds.max_ts) {
      document.getElementById("ts-start").textContent = formatShortTime(state.timeBounds.min_ts);
      document.getElementById("ts-end").textContent = formatShortTime(state.timeBounds.max_ts);
      document.getElementById("ts-current").textContent =
        state.mode === "live" ? "LIVE" : formatTime(data.timestamp);
      if (state.mode === "live") {
        document.getElementById("time-slider").value = 100;
      }
    }

    if (state.selectedVehicle) {
      const updated = state.vehicles.find((v) => v.vehicle_id === state.selectedVehicle.vehicle_id);
      if (updated) {
        state.selectedVehicle = updated;
        renderInspectorVehicle(updated);
      }
    }

    if (document.getElementById("layer-quadtree").checked) {
      await refreshQuadtree();
    }
    renderMap();
  }

  async function refreshMining() {
    const tsParam = state.mode === "historical" && state.historicalTs ? `?ts=${state.historicalTs}` : "";
    const data = await apiGet(`/api/mining${tsParam}`);
    state.dbscanHotspots = data.dbscan_hotspots || [];
    state.giCells = data.getis_ord_cells || [];
    state.bottlenecks = data.recurring_bottlenecks || [];

    document.getElementById("kpi-hotspots").textContent = state.dbscanHotspots.length;
    renderHotspots();
    renderBottlenecks();
    renderMap();
  }

  async function refreshQuadtree() {
    const data = await apiGet("/api/spatial/quadtree");
    state.quadtreeCells = data.cells || [];
    document.getElementById("kpi-qt-sub").textContent = `${data.leaf_cells} QT leaves + SQLite R*Tree`;
  }

  async function refreshHandoffs() {
    const data = await apiGet("/api/handoffs?limit=30");
    state.towers = data.towers || [];
    state.handoffs = data.events || [];
    renderTowers();
    renderHandoffEvents();
  }

  async function refreshHourlyProfile() {
    const data = await apiGet("/api/temporal/profile");
    state.hourlyProfile = data.hours || [];
    renderHourlyChart();
  }

  function renderHotspots() {
    const el = document.getElementById("hotspot-list");
    if (!state.dbscanHotspots.length) {
      el.innerHTML = `<div class="muted-empty">No severe DBSCAN congestion clusters detected right now.</div>`;
      return;
    }
    el.innerHTML = state.dbscanHotspots
      .map((hs, idx) => {
        const sevPct = Math.round(hs.severity * 100);
        const sevCls = sevPct >= 65 ? "sev-high" : "sev-med";
        return `
          <div class="list-item clickable" data-hs-idx="${idx}">
            <div class="item-top">
              <span>Cluster #${idx + 1} · ${hs.zone}</span>
              <span class="${sevCls}">${sevPct}% sev</span>
            </div>
            <div class="item-meta">
              <span>${hs.vehicle_count} slow vehicles</span>
              <span>avg ${hs.avg_speed_kmh} km/h · R=${Math.round(hs.radius_m)}m</span>
            </div>
          </div>`;
      })
      .join("");

    el.querySelectorAll("[data-hs-idx]").forEach((node) => {
      node.addEventListener("click", () => {
        const hs = state.dbscanHotspots[Number(node.dataset.hsIdx)];
        if (!hs) return;
        // Center map on hotspot
        state.zoom = 2.0;
        const [px, py] = latLonToPx(hs.center_lat, hs.center_lon);
        state.panX += canvas.clientWidth * 0.5 - px;
        state.panY += canvas.clientHeight * 0.5 - py;
        renderMap();
      });
    });
  }

  function renderBottlenecks() {
    const el = document.getElementById("bottleneck-list");
    if (!state.bottlenecks.length) {
      el.innerHTML = `<div class="muted-empty">No bottlenecks mined yet.</div>`;
      return;
    }
    el.innerHTML = state.bottlenecks
      .map(
        (b) => `
      <div class="list-item">
        <div class="item-top">
          <span>${b.name}</span>
          <span class="sev-high">${b.congestion_pct}%</span>
        </div>
        <div class="item-meta">
          <span>${b.zone} · Peak ${b.peak_hour}</span>
          <span>avg ${b.avg_speed_kmh}/${b.speed_limit_kmh} km/h</span>
        </div>
      </div>`
      )
      .join("");
  }

  function renderTowers() {
    const el = document.getElementById("tower-list");
    el.innerHTML = state.towers
      .map((t) => {
        const pct = Math.min(100, t.utilization_pct || 0);
        return `
          <div class="list-item">
            <div class="item-top">
              <span>${t.tower_id} · ${t.name}</span>
              <span>${t.connected_vehicles}/${t.capacity}</span>
            </div>
            <div class="item-meta">
              <span>${t.technology} · ${t.zone}</span>
              <span>${t.avg_rssi_dbm ? t.avg_rssi_dbm + " dBm" : "idle"}</span>
            </div>
            <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
          </div>`;
      })
      .join("");
  }

  function renderHandoffEvents() {
    const el = document.getElementById("handoff-list");
    if (!state.handoffs.length) {
      el.innerHTML = `<div class="muted-empty">No recent handoff events.</div>`;
      return;
    }
    el.innerHTML = state.handoffs
      .slice(0, 18)
      .map(
        (h) => `
      <div class="list-item">
        <div class="item-top">
          <span>${h.device_id} (${h.vehicle_id})</span>
          <span class="badge">${h.latency_ms} ms</span>
        </div>
        <div class="item-meta">
          <span>${h.from_tower_id || "INIT"} &rarr; ${h.to_tower_id}</span>
          <span>${h.rssi_to_dbm} dBm · ${h.time_str}</span>
        </div>
      </div>`
      )
      .join("");
  }

  function renderHourlyChart() {
    const chart = document.getElementById("hourly-chart");
    const c = chart.getContext("2d");
    const w = chart.width;
    const h = chart.height;
    c.clearRect(0, 0, w, h);
    const hrs = state.hourlyProfile;
    if (!hrs.length) return;

    const padL = 28, padR = 26, padT = 14, padB = 24;
    const pw = w - padL - padR;
    const ph = h - padT - padB;

    c.strokeStyle = "#233354";
    c.lineWidth = 1;
    c.beginPath();
    c.moveTo(padL, padT);
    c.lineTo(padL, padT + ph);
    c.lineTo(padL + pw, padT + ph);
    c.stroke();

    // Congestion bars
    const barW = Math.max(4, pw / hrs.length - 3);
    hrs.forEach((pt, i) => {
      const x = padL + (i / hrs.length) * pw + 1.5;
      const barH = (Math.min(100, pt.congestion_pct || 0) / 100) * ph;
      c.fillStyle = "rgba(239, 68, 68, 0.45)";
      c.fillRect(x, padT + ph - barH, barW, barH);
    });

    // Speed line
    c.beginPath();
    hrs.forEach((pt, i) => {
      const x = padL + ((i + 0.5) / hrs.length) * pw;
      const y = padT + ph - (Math.min(70, pt.avg_speed_kmh || 0) / 70) * ph;
      if (i === 0) c.moveTo(x, y);
      else c.lineTo(x, y);
    });
    c.strokeStyle = "#38bdf8";
    c.lineWidth = 2.2;
    c.stroke();

    // Labels
    c.fillStyle = "#8ea3c7";
    c.font = "10px monospace";
    c.textAlign = "center";
    [0, Math.floor(hrs.length / 2), hrs.length - 1].forEach((idx) => {
      if (hrs[idx]) {
        const x = padL + ((idx + 0.5) / hrs.length) * pw;
        c.fillText(hrs[idx].hour_label, x, h - 6);
      }
    });
  }

  function renderInspectorVehicle(v) {
    const card = document.getElementById("inspector-card");
    card.classList.remove("hidden");
    document.getElementById("insp-title").textContent = `${v.vehicle_id} · ${v.plate}`;
    document.getElementById("insp-body").innerHTML = `
      <div class="kv-row"><span>Type</span><strong>${v.vehicle_type}</strong></div>
      <div class="kv-row"><span>Speed</span><strong style="color:${speedColor(v.speed_kmh)}">${v.speed_kmh} km/h</strong></div>
      <div class="kv-row"><span>Road</span><strong>${v.segment_name || v.current_segment_id}</strong></div>
      <div class="kv-row"><span>Zone</span><strong>${v.zone || "-"}</strong></div>
      <div class="kv-row"><span>Mobile Device</span><strong>${v.device_id}</strong></div>
      <div class="kv-row"><span>Serving Tower</span><strong>${v.current_tower_id || "-"} (${v.rssi_dbm} dBm)</strong></div>
      <div class="kv-row"><span>Coords</span><strong>${v.lat.toFixed(5)}, ${v.lon.toFixed(5)}</strong></div>
    `;
  }

  // ---------------------------------------------------------------
  // Interactive Map Clicks & Tools
  // ---------------------------------------------------------------
  async function handleMapClick(px, py) {
    const [lat, lon] = pxToLatLon(px, py);

    if (state.activeTool === "inspect") {
      // Find closest vehicle within 18px
      let best = null;
      let bestDist = 18;
      for (const v of state.vehicles) {
        const [vx, vy] = latLonToPx(v.lat, v.lon);
        const d = Math.hypot(vx - px, vy - py);
        if (d < bestDist) {
          bestDist = d;
          best = v;
        }
      }
      if (best) {
        state.selectedVehicle = best;
        const traj = await apiGet(`/api/temporal/trajectory?vehicle_id=${best.vehicle_id}`);
        state.selectedTrajectory = traj.points || [];
        renderInspectorVehicle(best);
      } else {
        state.selectedVehicle = null;
        state.selectedTrajectory = [];
        document.getElementById("inspector-card").classList.add("hidden");
      }
      renderMap();
      return;
    }

    if (state.activeTool === "radius") {
      const r = Number(document.getElementById("input-radius").value) || 450;
      const res = await apiGet(`/api/spatial/radius?lat=${lat}&lon=${lon}&radius_m=${r}`);
      state.spatialHighlight = { type: "radius", lat, lon, radius_m: r, vehicles: res.vehicles };
      document.getElementById("spatial-op-badge").textContent = `${res.query_ms} ms`;
      document.getElementById("spatial-query-result").innerHTML = `
        <div class="kv-row"><span>Operator</span><strong>ST_DWithin (R-Tree)</strong></div>
        <div class="kv-row"><span>Matches / Examined</span><strong>${res.count} / ${res.candidates_examined}</strong></div>
        <div class="list-stack" style="margin-top:6px;">
          ${res.vehicles
            .slice(0, 6)
            .map(
              (v) => `<div class="list-item"><div class="item-top"><span>${v.vehicle_id} (${v.vehicle_type})</span><span>${v.distance_m} m</span></div><div class="item-meta"><span>${v.segment_name || ""}</span><span>${v.speed_kmh} km/h</span></div></div>`
            )
            .join("") || '<div class="muted-empty">No vehicles inside radius.</div>'}
        </div>`;
      renderMap();
      return;
    }

    if (state.activeTool === "knn") {
      const k = Number(document.getElementById("input-knn").value) || 6;
      const res = await apiGet(`/api/spatial/knn?lat=${lat}&lon=${lon}&k=${k}`);
      state.spatialHighlight = { type: "knn", lat, lon, vehicles: res.vehicles };
      document.getElementById("spatial-op-badge").textContent = `${res.query_ms} ms`;
      document.getElementById("spatial-query-result").innerHTML = `
        <div class="kv-row"><span>Operator</span><strong>Quadtree Best-First k-NN</strong></div>
        <div class="kv-row"><span>QT Nodes Visited</span><strong>${res.nodes_visited}</strong></div>
        <div class="list-stack" style="margin-top:6px;">
          ${res.vehicles
            .map(
              (v) => `<div class="list-item"><div class="item-top"><span>${v.vehicle_id}</span><span>${v.distance_m} m</span></div><div class="item-meta"><span>${v.segment_name || ""}</span><span>${v.speed_kmh} km/h</span></div></div>`
            )
            .join("")}
        </div>`;
      renderMap();
    }
  }

  // ---------------------------------------------------------------
  // Event Wiring
  // ---------------------------------------------------------------
  function initControls() {
    // Tabs
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById(btn.dataset.tab).classList.add("active");
      });
    });

    // Map tools
    document.querySelectorAll(".tool-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".tool-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.activeTool = btn.dataset.tool;
        document.getElementById("radius-param-wrap").classList.toggle("hidden", state.activeTool !== "radius");
        document.getElementById("knn-param-wrap").classList.toggle("hidden", state.activeTool !== "knn");
        canvas.classList.toggle("tool-crosshair", state.activeTool !== "inspect");
      });
    });

    // Layers
    ["layer-roads", "layer-towers", "layer-dbscan", "layer-gi", "layer-quadtree"].forEach((id) => {
      document.getElementById(id).addEventListener("change", async () => {
        if (id === "layer-quadtree" && document.getElementById(id).checked) {
          await refreshQuadtree();
        }
        renderMap();
      });
    });

    // Zoom / Pan
    document.getElementById("btn-zoom-in").addEventListener("click", () => {
      state.zoom = Math.min(6, state.zoom * 1.25);
      renderMap();
    });
    document.getElementById("btn-zoom-out").addEventListener("click", () => {
      state.zoom = Math.max(0.6, state.zoom / 1.25);
      renderMap();
    });
    document.getElementById("btn-zoom-reset").addEventListener("click", () => {
      state.zoom = 1.0;
      state.panX = 0;
      state.panY = 0;
      renderMap();
    });

    canvas.addEventListener("mousedown", (e) => {
      state.dragging = true;
      state.dragMoved = false;
      state.dragStartX = e.clientX;
      state.dragStartY = e.clientY;
      canvas.classList.add("dragging");
    });

    window.addEventListener("mousemove", (e) => {
      if (!state.dragging) return;
      const dx = e.clientX - state.dragStartX;
      const dy = e.clientY - state.dragStartY;
      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) state.dragMoved = true;
      state.panX += dx;
      state.panY += dy;
      state.dragStartX = e.clientX;
      state.dragStartY = e.clientY;
      renderMap();
    });

    window.addEventListener("mouseup", (e) => {
      if (!state.dragging) return;
      state.dragging = false;
      canvas.classList.remove("dragging");
      if (!state.dragMoved && e.target === canvas) {
        const rect = canvas.getBoundingClientRect();
        handleMapClick(e.clientX - rect.left, e.clientY - rect.top);
      }
    });

    canvas.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        const factor = e.deltaY < 0 ? 1.12 : 0.89;
        state.zoom = Math.max(0.6, Math.min(6.0, state.zoom * factor));
        renderMap();
      },
      { passive: false }
    );

    document.getElementById("insp-close").addEventListener("click", () => {
      state.selectedVehicle = null;
      state.selectedTrajectory = [];
      document.getElementById("inspector-card").classList.add("hidden");
      renderMap();
    });

    // Simulation controls
    document.getElementById("btn-play-pause").addEventListener("click", () => {
      state.liveRunning = !state.liveRunning;
      document.getElementById("btn-play-pause").textContent = state.liveRunning ? "Pause Live" : "Resume Live";
      refreshState();
    });

    document.getElementById("btn-step").addEventListener("click", async () => {
      state.mode = "live";
      await apiPost("/api/sim/tick", { dt_sec: 8 });
      await Promise.all([refreshState(), refreshHandoffs()]);
    });

    document.getElementById("btn-incident").addEventListener("click", async () => {
      await apiPost("/api/sim/incident", {});
      await Promise.all([refreshState(), refreshMining()]);
    });

    document.getElementById("btn-reset").addEventListener("click", async () => {
      await apiPost("/api/sim/reset", { num_vehicles: 220 });
      state.mode = "live";
      state.spatialHighlight = null;
      await bootstrapData();
    });

    document.getElementById("btn-run-mining").addEventListener("click", refreshMining);

    document.getElementById("btn-run-bench").addEventListener("click", async () => {
      const box = document.getElementById("benchmark-results");
      box.innerHTML = `<div class="muted-empty">Running 50 spatial queries...</div>`;
      const b = await apiGet("/api/spatial/benchmark?queries=50&radius_m=450");
      const maxMs = Math.max(b.linear_scan.total_ms, b.quadtree.total_ms, b.sqlite_rtree.total_ms, 0.01);
      box.innerHTML = `
        <div class="bench-row">
          <div class="item-top"><span>Linear Scan (${b.linear_scan.points_scanned.toLocaleString()} checks)</span><span>${b.linear_scan.total_ms} ms</span></div>
          <div class="bar-track"><div class="bar-fill" style="width:${(b.linear_scan.total_ms / maxMs) * 100}%; background:#ef4444;"></div></div>
        </div>
        <div class="bench-row">
          <div class="item-top"><span>In-Memory Quadtree (${b.quadtree.speedup_vs_linear}x faster)</span><span>${b.quadtree.total_ms} ms</span></div>
          <div class="bar-track"><div class="bar-fill" style="width:${(b.quadtree.total_ms / maxMs) * 100}%; background:#38bdf8;"></div></div>
        </div>
        <div class="bench-row">
          <div class="item-top"><span>SQLite R*Tree (${b.sqlite_rtree.speedup_vs_linear}x faster)</span><span>${b.sqlite_rtree.total_ms} ms</span></div>
          <div class="bar-track"><div class="bar-fill" style="width:${(b.sqlite_rtree.total_ms / maxMs) * 100}%; background:#22c55e;"></div></div>
        </div>
        <div class="item-meta"><span>Exact result parity: ${b.results_match ? "VERIFIED" : "MISMATCH"}</span><span>${b.total_hits} total hits</span></div>
      `;
    });

    // Timeline scrubber
    const slider = document.getElementById("time-slider");
    slider.addEventListener("input", async () => {
      if (!state.timeBounds.min_ts || !state.timeBounds.max_ts) return;
      const frac = Number(slider.value) / 100;
      const ts = Math.round(
        state.timeBounds.min_ts + frac * (state.timeBounds.max_ts - state.timeBounds.min_ts)
      );
      state.mode = "historical";
      state.historicalTs = ts;
      await refreshState();
    });
    slider.addEventListener("change", () => {
      if (state.mode === "historical") refreshMining();
    });

    document.getElementById("btn-go-live").addEventListener("click", async () => {
      state.mode = "live";
      state.histPlaying = false;
      state.historicalTs = null;
      await Promise.all([refreshState(), refreshMining()]);
    });

    document.getElementById("btn-hist-play").addEventListener("click", () => {
      if (!state.timeBounds.min_ts || !state.timeBounds.max_ts) return;
      state.histPlaying = !state.histPlaying;
      if (state.histPlaying) {
        state.mode = "historical";
        if (Number(slider.value) >= 99) slider.value = 0;
      }
      document.getElementById("btn-hist-play").textContent = state.histPlaying ? "Stop Replay" : "Replay 24h";
    });
  }

  async function bootstrapData() {
    const net = await apiGet("/api/network");
    state.network = net;
    state.bounds = net.bounds;
    await Promise.all([
      refreshState(),
      refreshMining(),
      refreshQuadtree(),
      refreshHandoffs(),
      refreshHourlyProfile(),
    ]);
    renderMap();
  }

  // Background simulation & historical replay loop
  setInterval(async () => {
    try {
      if (state.histPlaying && state.mode === "historical") {
        const slider = document.getElementById("time-slider");
        let val = Number(slider.value) + 2;
        if (val > 100) {
          val = 100;
          state.histPlaying = false;
          document.getElementById("btn-hist-play").textContent = "Replay 24h";
        }
        slider.value = val;
        const frac = val / 100;
        state.historicalTs = Math.round(
          state.timeBounds.min_ts + frac * (state.timeBounds.max_ts - state.timeBounds.min_ts)
        );
        await refreshState();
        if (val % 10 === 0) await refreshMining();
      } else if (state.mode === "live" && state.liveRunning) {
        await apiPost("/api/sim/tick", { dt_sec: 6 });
        await Promise.all([refreshState(), refreshHandoffs()]);
      }
    } catch (err) {
      console.error("Loop error:", err);
    }
  }, 2200);

  // Periodic DBSCAN re-mining in live mode
  setInterval(() => {
    if (state.mode === "live" && state.liveRunning) {
      refreshMining().catch(() => {});
    }
  }, 9000);

  window.addEventListener("resize", resizeCanvas);
  initControls();
  bootstrapData().then(resizeCanvas);
})();
