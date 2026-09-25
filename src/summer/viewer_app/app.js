// SUMMER viewer. Every time here is paradigm time in seconds (paradigm frame p = p * 0.04 s).
// Tracks come from /api/manifest; each track type has a renderer in RENDERERS below.
// To support a new kind of track, add a renderer with: init, height, draw, and optionally now/hover.

const $ = (s) => document.querySelector(s);
const video = $("#video");
const canvas = $("#timeline");
const ctx = canvas.getContext("2d");
const GUTTER = 170;             // css px reserved for lane labels on the left
const FPS = 25;

const state = {
  manifest: null,
  tracks: [],                   // {meta, data, enabled, ui: {...}}
  windowS: 10,
  dirty: true,
  lastT: -1,
  layout: [],                   // [{track, y, h}] from the last draw, for clicks/hover
};

// ---------- small utils ----------
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const fmtTime = (t) => {
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(2).padStart(5, "0")}`;
};
const frameAt = (t) => Math.floor(t * FPS + 1e-6);
function lowerBound(arr, x) { // first index with arr[i] >= x
  let lo = 0, hi = arr.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < x) lo = m + 1; else hi = m; }
  return lo;
}
const prefs = (() => {
  let p = {};
  try { p = JSON.parse(localStorage.getItem("summer-viewer") || "{}"); } catch { }
  let timer = null;
  return {
    get: (k, d) => (k in p ? p[k] : d),
    set: (k, v) => {
      p[k] = v;
      clearTimeout(timer);
      timer = setTimeout(() => { try { localStorage.setItem("summer-viewer", JSON.stringify(p)); } catch { } }, 300);
    },
  };
})();
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "on") for (const [ev, fn] of Object.entries(v)) e.addEventListener(ev, fn);
    else if (k in e && k !== "list") e[k] = v; else e.setAttribute(k, v);
  }
  for (const k of kids) e.append(k);
  return e;
}
function checkbox(label, checked, onchange, title = "") {
  const cb = el("input", { type: "checkbox", checked });
  cb.addEventListener("change", () => onchange(cb.checked));
  return el("label", { title: title || label }, cb, " " + label);
}
const markDirty = () => { state.dirty = true; };
const status = (s) => { $("#status").textContent = s; };

// ---------- renderers ----------
const GROUP_COLORS = {
  Transitions: "#ff6b6b", Characters: "#4dabf7", Faces: "#b197fc", Speaking: "#69db7c",
  Presence: "#38d9a9", Setting: "#adb5bd", Locations: "#ffa94d", Alignment: "#f783ac",
};
const colorFor = (g) => GROUP_COLORS[g] || "#ced4da";
const REGION_COLORS = ["#4dabf7", "#ff8787", "#69db7c", "#ffd43b", "#b197fc", "#ffa94d", "#38d9a9",
  "#f783ac", "#a9e34b", "#74c0fc", "#e599f7", "#ffc078"];

const RENDERERS = {
  // Named on/off labels, one lane per enabled label.
  intervals: {
    ROW: 12,
    init(tr) {
      const d = tr.data;
      tr.groupOf = {};
      for (const [g, names] of Object.entries(d.label_groups)) for (const n of names) tr.groupOf[n] = g;
      tr.idx = {};
      for (const [n, iv] of Object.entries(d.labels)) {
        tr.idx[n] = { a: Float64Array.from(iv, (x) => x[0]), b: Float64Array.from(iv, (x) => x[1]) };
      }
      const defaults = ["camera-cuts", "scenes", "days-of-summer", "tom", "summer", "tom-faces", "summer-faces",
        "tom-speaking", "summer-speaking", "indoor-setting"];
      const saved = prefs.get(`labels:${tr.meta.id}`, null);
      tr.on = new Set(saved ?? Object.keys(d.labels).filter((n) => defaults.includes(n) || tr.meta.id !== "annotations"));
      const save = () => { prefs.set(`labels:${tr.meta.id}`, [...tr.on]); markDirty(); };
      const box = el("div");
      for (const [g, names] of Object.entries(d.label_groups)) {
        const items = el("div", { className: "items" });
        const boxes = [];
        for (const n of names) {
          const lab = checkbox(n, tr.on.has(n), (v) => { v ? tr.on.add(n) : tr.on.delete(n); save(); });
          boxes.push([n, lab.querySelector("input")]);
          items.append(lab);
        }
        const all = el("button", { className: "small", textContent: "all", on: { click: (e) => { e.preventDefault(); boxes.forEach(([n, c]) => { c.checked = true; tr.on.add(n); }); save(); } } });
        const none = el("button", { className: "small", textContent: "none", on: { click: (e) => { e.preventDefault(); boxes.forEach(([n, c]) => { c.checked = false; tr.on.delete(n); }); save(); } } });
        const sw = el("span", { textContent: "■ " }); sw.style.color = colorFor(g);
        box.append(el("details", { className: "group", open: g === "Transitions" || g === "Characters" },
          el("summary", {}, sw, `${g} (${names.length}) `, all, " ", none), items));
      }
      return box;
    },
    labels(tr) {
      const order = Object.values(tr.data.label_groups).flat();
      return order.filter((n) => tr.on.has(n));
    },
    height(tr) { return this.labels(tr).length * this.ROW; },
    draw(tr, g, y, t0, t1) {
      const names = this.labels(tr);
      names.forEach((n, i) => {
        const yy = y + i * this.ROW;
        const { a, b } = tr.idx[n];
        g.fillStyle = i % 2 ? "#1c1f23" : "#202328";
        g.fillRect(GUTTER, yy, g.plotW, this.ROW);
        g.fillStyle = colorFor(tr.groupOf[n]);
        let j = Math.max(0, lowerBound(b, t0));
        for (; j < a.length && a[j] < t1; j++) {
          const x0 = g.x(a[j]), x1 = g.x(b[j]);
          g.fillRect(Math.max(GUTTER, x0), yy + 1, Math.max(1.5, Math.min(x1, GUTTER + g.plotW) - Math.max(GUTTER, x0)), this.ROW - 2);
        }
        g.label(n, yy + this.ROW - 2, colorFor(tr.groupOf[n]));
      });
    },
    now(tr, t) {
      const out = [];
      for (const [n, { a, b }] of Object.entries(tr.idx)) {
        const j = lowerBound(b, t + 1e-9);
        if (j < a.length && a[j] <= t) out.push([n, tr.groupOf[n]]);
      }
      return { chips: out };
    },
    hover(tr, t, dy) {
      const n = this.labels(tr)[Math.floor(dy / this.ROW)];
      return n ? `${n} (${tr.groupOf[n]})` : "";
    },
  },

  // Timed text: subtitles, captions, VLM descriptions.
  text: {
    H: 30,
    init(tr) {
      tr.starts = Float64Array.from(tr.data.segments, (s) => s[0]);
      tr.ends = Float64Array.from(tr.data.segments, (s) => s[1]);
      $("#overlaysel").append(el("option", { value: tr.meta.id, textContent: tr.meta.name }));
      return null;
    },
    height() { return this.H; },
    current(tr, t) {
      const j = lowerBound(tr.ends, t + 1e-9);
      return j < tr.starts.length && tr.starts[j] <= t ? tr.data.segments[j][2] : "";
    },
    draw(tr, g, y, t0, t1) {
      g.fillStyle = "#1c1f23"; g.fillRect(GUTTER, y, g.plotW, this.H);
      g.font = "11px system-ui";
      let j = lowerBound(tr.ends, t0);
      for (let k = 0; j < tr.starts.length && tr.starts[j] < t1; j++, k++) {
        const [a, b, txt] = tr.data.segments[j];
        const x0 = Math.max(GUTTER, g.x(a)), x1 = Math.min(GUTTER + g.plotW, g.x(b));
        g.fillStyle = k % 2 ? "#3b4a5a" : "#34495e";
        g.fillRect(x0, y + 2, Math.max(1, x1 - x0 - 1), this.H - 4);
        if (x1 - x0 > 12) {
          g.save(); g.beginPath(); g.rect(x0, y, x1 - x0 - 2, this.H); g.clip();
          g.fillStyle = "#e6e6e6"; g.fillText(txt.replace(/\n/g, " / "), x0 + 3, y + this.H / 2 + 4);
          g.restore();
        }
      }
      g.label(tr.meta.name, y + this.H / 2 + 4);
    },
    now(tr, t) { const s = this.current(tr, t); return s ? { text: [tr.meta.name, s] } : {}; },
    hover(tr, t) { return this.current(tr, t); },
  },

  // A regularly sampled numeric signal.
  series: {
    H: 46,
    init(tr) {
      const v = tr.data.values.map((x) => (x === null ? NaN : x));
      tr.v = Float32Array.from(v);
      const s = [...tr.v].filter(Number.isFinite).sort((a, b) => a - b);
      tr.ymax = s.length ? s[Math.floor(0.995 * (s.length - 1))] || 1 : 1;
      tr.ymin = Math.min(0, s[0] ?? 0);
      return null;
    },
    height() { return this.H; },
    draw(tr, g, y, t0, t1) {
      const { t0: s0, dt } = tr.data, v = tr.v;
      g.fillStyle = "#1c1f23"; g.fillRect(GUTTER, y, g.plotW, this.H);
      const yOf = (val) => y + this.H - 2 - (clamp(val, tr.ymin, tr.ymax) - tr.ymin) / (tr.ymax - tr.ymin) * (this.H - 4);
      g.strokeStyle = "#f0b429"; g.lineWidth = 1; g.beginPath();
      const i0 = Math.max(0, Math.floor((t0 - s0) / dt)), i1 = Math.min(v.length - 1, Math.ceil((t1 - s0) / dt));
      const perPx = (i1 - i0) / g.plotW;
      if (perPx <= 2) {
        let pen = false;
        for (let i = i0; i <= i1; i++) {
          if (!Number.isFinite(v[i])) { pen = false; continue; }
          const x = g.x(s0 + i * dt), yy = yOf(v[i]);
          pen ? g.lineTo(x, yy) : g.moveTo(x, yy); pen = true;
        }
      } else { // min/max per pixel column
        for (let px = 0; px < g.plotW; px++) {
          const a = i0 + Math.floor(px * perPx), b = Math.min(i1, i0 + Math.floor((px + 1) * perPx));
          let lo = Infinity, hi = -Infinity;
          for (let i = a; i <= b; i++) if (Number.isFinite(v[i])) { lo = Math.min(lo, v[i]); hi = Math.max(hi, v[i]); }
          if (hi >= lo) { g.moveTo(GUTTER + px + 0.5, yOf(lo)); g.lineTo(GUTTER + px + 0.5, yOf(hi) - 0.5); }
        }
      }
      g.stroke();
      g.label(`${tr.meta.name}`, y + 14);
      if (tr.data.unit) g.label(tr.data.unit, y + 28, "#9aa0a8");
    },
    hover(tr, t) {
      const i = Math.round((t - tr.data.t0) / tr.data.dt);
      return i >= 0 && i < tr.v.length ? `${tr.meta.name}: ${tr.v[i].toFixed(3)}` : "";
    },
  },

  // Spike trains of many units across patients.
  spikes: {
    RATE_H: 40,
    init(tr) {
      tr.loaded = {};            // patient -> Float32Array (all units back to back)
      tr.loading = new Set();
      const pats = tr.data.patients;
      tr.selPatients = new Set(prefs.get("spk:patients", pats.slice(0, 3).map((p) => p.patient)));
      const regions = [...new Set(pats.flatMap((p) => p.units.map((u) => u.region)))].sort();
      tr.regionColor = Object.fromEntries(regions.map((r, i) => [r, REGION_COLORS[i % REGION_COLORS.length]]));
      tr.selRegions = new Set(prefs.get("spk:regions", regions));
      tr.suOnly = prefs.get("spk:suOnly", false);
      tr.mode = prefs.get("spk:mode", "raster");
      tr.bin = prefs.get("spk:bin", 0.08);
      tr.rowH = prefs.get("spk:rowH", 2);
      tr.sort = prefs.get("spk:sort", "patient");
      const save = () => {
        prefs.set("spk:patients", [...tr.selPatients]); prefs.set("spk:regions", [...tr.selRegions]);
        prefs.set("spk:suOnly", tr.suOnly); prefs.set("spk:mode", tr.mode); prefs.set("spk:bin", tr.bin);
        prefs.set("spk:rowH", tr.rowH); prefs.set("spk:sort", tr.sort);
        this.rebuild(tr); markDirty();
      };
      const sel = (opts, cur, fn) => {
        const s = el("select", {}, ...opts.map(([v, l]) => el("option", { value: v, textContent: l, selected: String(v) === String(cur) })));
        s.addEventListener("change", () => fn(s.value)); return s;
      };
      const patBoxes = [], regBoxes = [];
      const patItems = el("div", { className: "items" });
      for (const p of pats) {
        const n = p.units.length, su = p.units.filter((u) => u.single_unit).length;
        const lab = checkbox(`sub-${p.patient} (${n}, ${su} SU)`, tr.selPatients.has(p.patient),
          (v) => { v ? tr.selPatients.add(p.patient) : tr.selPatients.delete(p.patient); save(); });
        patBoxes.push([p.patient, lab.querySelector("input")]); patItems.append(lab);
      }
      const regItems = el("div", { className: "items" });
      for (const r of regions) {
        const lab = checkbox(r, tr.selRegions.has(r), (v) => { v ? tr.selRegions.add(r) : tr.selRegions.delete(r); save(); });
        lab.style.color = tr.regionColor[r];
        regBoxes.push([r, lab.querySelector("input")]); regItems.append(lab);
      }
      const allNone = (boxes, set) => [
        el("button", { className: "small", textContent: "all", on: { click: (e) => { e.preventDefault(); boxes.forEach(([k, c]) => { c.checked = true; set.add(k); }); save(); } } }),
        " ",
        el("button", { className: "small", textContent: "none", on: { click: (e) => { e.preventDefault(); boxes.forEach(([k, c]) => { c.checked = false; set.delete(k); }); save(); } } }),
      ];
      return el("div", { className: "opts" },
        el("div", { className: "inline" },
          checkbox("single units only", tr.suOnly, (v) => { tr.suOnly = v; save(); }),
          el("label", {}, "sort ", sel([["patient", "patient"], ["region", "region"]], tr.sort, (v) => { tr.sort = v; save(); }))),
        el("div", { className: "inline" },
          el("label", {}, "mode ", sel([["raster", "raster"], ["heat", "rate heatmap"]], tr.mode, (v) => { tr.mode = v; save(); })),
          el("label", {}, "bin ", sel([[0.04, "40 ms"], [0.08, "80 ms"], [0.2, "200 ms"], [0.48, "480 ms"], [1, "1 s"]], tr.bin, (v) => { tr.bin = +v; save(); })),
          el("label", {}, "row ", sel([[1, "1px"], [2, "2px"], [3, "3px"], [5, "5px"]], tr.rowH, (v) => { tr.rowH = +v; save(); }))),
        el("details", { className: "group", open: true }, el("summary", {}, `Patients (${pats.length}) `, ...allNone(patBoxes, tr.selPatients)), patItems),
        el("details", { className: "group" }, el("summary", {}, `Regions (${regions.length}) `, ...allNone(regBoxes, tr.selRegions)), regItems),
      );
    },
    async ensureLoaded(tr, pid) {
      if (tr.loaded[pid] || tr.loading.has(pid)) return;
      tr.loading.add(pid);
      const p = tr.data.patients.find((x) => x.patient === pid);
      const buf = await (await fetch(`/data/viewer/tracks/${p.bin}`)).arrayBuffer();
      tr.loaded[pid] = new Float32Array(buf);
      tr.loading.delete(pid);
      this.rebuild(tr); markDirty();
    },
    rebuild(tr) {
      const rows = [];
      for (const p of tr.data.patients) {
        if (!tr.selPatients.has(p.patient)) continue;
        if (!tr.loaded[p.patient]) { this.ensureLoaded(tr, p.patient); continue; }
        const all = tr.loaded[p.patient];
        for (const u of p.units) {
          if (!tr.selRegions.has(u.region) || (tr.suOnly && !u.single_unit)) continue;
          const spikes = all.subarray(u.offset, u.offset + u.count);
          const dur = spikes.length ? spikes[spikes.length - 1] - spikes[0] : 1;
          rows.push({ patient: p.patient, u, spikes, rate: spikes.length / Math.max(dur, 1) });
        }
      }
      const key = tr.sort === "region" ? (r) => [r.u.region, r.patient, r.u.unit_id] : (r) => [r.patient, r.u.region, r.u.unit_id];
      rows.sort((a, b) => { const ka = key(a), kb = key(b); for (let i = 0; i < 3; i++) if (ka[i] !== kb[i]) return ka[i] < kb[i] ? -1 : 1; return 0; });
      tr.rows = rows;
    },
    height(tr) { if (!tr.rows) this.rebuild(tr); return this.RATE_H + tr.rows.length * tr.rowH + 4; },
    draw(tr, g, y, t0, t1) {
      const rows = tr.rows || [];
      const W = g.plotW, span = t1 - t0;
      // population rate (spikes/s per unit), binned
      const nb = Math.max(1, Math.min(W, Math.round(span / tr.bin)));
      const counts = new Float32Array(nb);
      for (const r of rows) {
        const s = r.spikes;
        for (let j = lowerBound(s, t0); j < s.length && s[j] < t1; j++) counts[Math.floor((s[j] - t0) / span * nb)]++;
      }
      g.fillStyle = "#1c1f23"; g.fillRect(GUTTER, y, W, this.RATE_H);
      if (rows.length) {
        const rate = counts.map((c) => c / (span / nb) / rows.length);
        const mx = Math.max(1, ...rate);
        g.fillStyle = "#74c0fc";
        for (let i = 0; i < nb; i++) {
          const h = rate[i] / mx * (this.RATE_H - 4);
          g.fillRect(GUTTER + i * W / nb, y + this.RATE_H - 2 - h, Math.max(1, W / nb - (nb < W / 2 ? 1 : 0)), h);
        }
        g.label(`rate, max ${mx.toFixed(1)} Hz/unit`, y + 14);
        g.label(`${rows.length} units`, y + 28, "#9aa0a8");
      } else {
        g.label(tr.loading.size ? "loading spikes…" : "no units selected", y + 20, "#9aa0a8");
      }
      if (!rows.length) return;
      // raster / heatmap straight into pixels (device resolution)
      const dpr = g.dpr, top = y + this.RATE_H + 2;
      const pw = Math.round(W * dpr), rh = Math.max(1, Math.round(tr.rowH * dpr)), ph = rows.length * rh;
      const img = g.ctx.createImageData(pw, ph);
      const px = new Uint32Array(img.data.buffer);
      px.fill(0xff231f1c);                      // ABGR of #1c1f23
      const abgr = (hex, a = 1) => {
        const n = parseInt(hex.slice(1), 16), r = n >> 16, gg = (n >> 8) & 255, b = n & 255;
        return (255 << 24) | (Math.round(b * a + 0x23 * (1 - a)) << 16) | (Math.round(gg * a + 0x1f * (1 - a)) << 8) | Math.round(r * a + 0x1c * (1 - a));
      };
      const nbins = tr.mode === "heat" ? Math.max(1, Math.round(span / tr.bin)) : 0;
      const bc = new Float32Array(nbins);
      rows.forEach((r, i) => {
        const s = r.spikes, y0 = i * rh, col = tr.regionColor[r.u.region];
        const lo = lowerBound(s, t0);
        if (tr.mode === "raster") {
          const c = abgr(col);
          for (let j = lo; j < s.length && s[j] < t1; j++) {
            const x = Math.floor((s[j] - t0) / span * pw);
            for (let k = 0; k < rh; k++) px[(y0 + k) * pw + x] = c;
          }
        } else {
          bc.fill(0);
          for (let j = lo; j < s.length && s[j] < t1; j++) bc[Math.floor((s[j] - t0) / span * nbins)]++;
          const expect = r.rate * (span / nbins);
          for (let b = 0; b < nbins; b++) {
            if (!bc[b]) continue;
            const c = abgr(col, clamp(bc[b] / (expect * 4 + 1e-9), 0.08, 1));
            const xa = Math.floor(b / nbins * pw), xb = Math.floor((b + 1) / nbins * pw);
            for (let k = 0; k < rh; k++) px.fill(c, (y0 + k) * pw + xa, (y0 + k) * pw + xb);
          }
        }
      });
      g.ctx.putImageData(img, Math.round(GUTTER * dpr), Math.round(top * dpr));
      // gutter: region bands and patient separators
      let start = 0;
      for (let i = 1; i <= rows.length; i++) {
        const a = rows[start], b = rows[i];
        const grpKey = (r) => (tr.sort === "region" ? r.u.region : `${r.patient}|${r.u.region}`);
        if (i === rows.length || grpKey(a) !== grpKey(b)) {
          const ya = top + start * rh / dpr, yb = top + i * rh / dpr;
          g.fillStyle = tr.regionColor[a.u.region];
          g.fillRect(GUTTER - 6, ya, 4, Math.max(1, yb - ya - 0.5));
          if (yb - ya >= 10) {
            g.font = "10px system-ui"; g.textAlign = "right"; g.fillStyle = tr.regionColor[a.u.region];
            g.fillText(tr.sort === "region" ? a.u.region : `sub-${a.patient} ${a.u.region}`, GUTTER - 9, (ya + yb) / 2 + 3);
            g.textAlign = "left";
          }
          start = i;
        }
      }
    },
    hover(tr, t, dy) {
      const i = Math.floor((dy - this.RATE_H - 2) / tr.rowH);
      const r = (tr.rows || [])[i];
      if (!r) return "";
      const u = r.u;
      return `sub-${r.patient} unit ${u.unit_id} · ${u.region} ${u.hemisphere} · ${u.single_unit ? "single unit" : "multi-unit"} · ${u.cell_type} · ${r.rate.toFixed(2)} Hz`;
    },
  },
};

// ---------- sidebar ----------
function buildFilters() {
  const root = $("#trackfilters");
  const groups = {};
  for (const tr of state.tracks) (groups[tr.meta.group] ??= []).push(tr);
  for (const [g, trs] of Object.entries(groups)) {
    root.append(el("div", { className: "desc", textContent: g.toUpperCase() }));
    for (const tr of trs) {
      const head = checkbox(tr.meta.name, tr.enabled, (v) => { tr.enabled = v; prefs.set(`on:${tr.meta.id}`, v); body.style.display = v ? "" : "none"; markDirty(); });
      const body = el("div");
      if (tr.meta.description) body.append(el("div", { className: "desc", textContent: tr.meta.description }));
      if (tr.ui) body.append(tr.ui);
      body.style.display = tr.enabled ? "" : "none";
      root.append(el("div", { className: "track" }, head, body));
    }
  }
}

// ---------- timeline drawing ----------
function geometry(t0, t1) {
  const cssW = canvas.clientWidth, plotW = cssW - GUTTER - 8;
  const g = ctx;
  g.plotW = plotW; g.dpr = window.devicePixelRatio || 1; g.ctx = ctx;
  g.x = (t) => GUTTER + (t - t0) / (t1 - t0) * plotW;
  g.label = (txt, y, color = "#e6e6e6") => {
    g.font = "11px system-ui"; g.fillStyle = color; g.textAlign = "right";
    g.fillText(txt.length > 26 ? txt.slice(0, 25) + "…" : txt, GUTTER - 8, y); g.textAlign = "left";
  };
  return g;
}

function drawTimeline(t) {
  const W = state.windowS, t0 = t - W / 2, t1 = t + W / 2;
  const active = state.tracks.filter((tr) => tr.enabled);
  const AXIS = 18, GAP = 6;
  let H = AXIS;
  const layout = active.map((tr) => { const h = RENDERERS[tr.meta.type].height(tr); const l = { tr, y: H, h }; H += h + GAP; return l; });
  const dpr = window.devicePixelRatio || 1, cssW = canvas.clientWidth;
  if (canvas.height !== Math.round(H * dpr) || canvas.width !== Math.round(cssW * dpr)) {
    canvas.style.height = `${H}px`; canvas.height = Math.round(H * dpr); canvas.width = Math.round(cssW * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#16181c"; ctx.fillRect(0, 0, cssW, H);
  const g = geometry(t0, t1);
  // axis: ticks every nice step, frames when zoomed in
  const step = [0.04, 0.2, 0.5, 1, 2, 5, 10, 30, 60, 120, 300].find((s) => W / s <= 12) || 600;
  ctx.font = "10px system-ui"; ctx.fillStyle = "#9aa0a8"; ctx.strokeStyle = "#33373e";
  for (let s = Math.ceil(t0 / step) * step; s <= t1; s += step) {
    const x = g.x(s);
    ctx.beginPath(); ctx.moveTo(x, AXIS - 5); ctx.lineTo(x, H); ctx.stroke();
    ctx.fillText(step < 1 ? `p${frameAt(s + 1e-6)}` : fmtTime(s).replace(/\.00$/, ""), x + 2, 11);
  }
  // outside the movie
  ctx.fillStyle = "rgba(0,0,0,0.5)";
  const dur = state.manifest.n_frames / FPS;
  if (t0 < 0) ctx.fillRect(GUTTER, AXIS, g.x(0) - GUTTER, H);
  if (t1 > dur) ctx.fillRect(g.x(dur), AXIS, GUTTER + g.plotW - g.x(dur), H);
  for (const l of layout) {
    ctx.save(); ctx.beginPath(); ctx.rect(0, l.y, cssW, l.h); ctx.clip();
    RENDERERS[l.tr.meta.type].draw(l.tr, g, l.y, t0, t1);
    ctx.restore();
  }
  // current frame: shade its 40 ms, then the playhead
  const p = frameAt(t);
  ctx.fillStyle = "rgba(255,77,77,0.18)";
  ctx.fillRect(g.x(p / FPS), AXIS, Math.max(1, g.x((p + 1) / FPS) - g.x(p / FPS)), H);
  ctx.strokeStyle = "#ff4d4d"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(g.x(t), 0); ctx.lineTo(g.x(t), H); ctx.stroke();
  state.layout = layout; state.t0 = t0; state.t1 = t1;
}

function drawSeekbar(t) {
  const c = $("#seekbar"), g = c.getContext("2d"), dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = 18;
  if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const dur = state.manifest.n_frames / FPS, x = (s) => s / dur * w;
  g.fillStyle = "#26292e"; g.fillRect(0, 4, w, 10);
  const [a, b] = state.manifest.analysis;
  g.fillStyle = "#34495e"; g.fillRect(x(a / FPS), 4, x((b - a) / FPS), 10);
  g.fillStyle = "#f0b429"; g.fillRect(0, 7, x(t), 4);
  g.fillStyle = "#fff"; g.fillRect(x(t) - 1, 1, 2, 16);
}

// ---------- now panel / overlay ----------
function drawNow(t) {
  const chips = [], texts = [];
  for (const tr of state.tracks) {
    const r = RENDERERS[tr.meta.type].now?.(tr, t) || {};
    if (r.chips) chips.push(...r.chips);
    if (r.text && tr.enabled) texts.push(r.text);
  }
  const lab = $("#nowlabels"); lab.replaceChildren(...chips.map(([n, grp]) => { const c = el("span", { className: "chip", textContent: n }); c.style.background = colorFor(grp); return c; }));
  $("#nowtexts").replaceChildren(...texts.map(([name, s]) => el("div", {}, el("b", { textContent: name + ": " }), s)));
  const ov = state.tracks.find((tr) => tr.meta.id === $("#overlaysel").value);
  $("#overlay").textContent = ov ? RENDERERS.text.current(ov, t) : "";
}

// ---------- audio: separate elements kept in sync with the (muted) video ----------
const audio = { el: {}, cur: prefs.get("audio", "de") };
function setupAudio(urls) {
  for (const [k, u] of Object.entries(urls)) { const a = new Audio(u); a.preload = "auto"; audio.el[k] = a; }
  for (const o of [...$("#audiosel").options]) if (o.value !== "off" && !(o.value in urls)) o.remove();
  if (!(audio.cur in urls) && audio.cur !== "off") audio.cur = Object.keys(urls)[0] ?? "off";
  $("#audiosel").value = audio.cur;
  $("#audiosel").addEventListener("change", (e) => {
    for (const a of Object.values(audio.el)) a.pause();
    audio.cur = e.target.value; prefs.set("audio", audio.cur); syncAudio(true);
  });
  const vol = $("#volume"); vol.value = prefs.get("volume", 0.8);
  const setVol = () => { for (const a of Object.values(audio.el)) a.volume = +vol.value; prefs.set("volume", +vol.value); };
  vol.addEventListener("input", setVol); setVol();
  video.addEventListener("play", () => syncAudio(true));
  video.addEventListener("pause", () => { for (const a of Object.values(audio.el)) a.pause(); });
  video.addEventListener("seeked", () => syncAudio(true));
  video.addEventListener("ratechange", () => { for (const a of Object.values(audio.el)) a.playbackRate = video.playbackRate; });
  video.addEventListener("waiting", () => audio.el[audio.cur]?.pause());
  video.addEventListener("playing", () => syncAudio(true));
}
function syncAudio(force = false) {
  const a = audio.el[audio.cur];
  if (!a) return;
  const drift = a.currentTime - video.currentTime;
  if (force || Math.abs(drift) > 0.3) a.currentTime = video.currentTime;   // big jump: seek
  // small offsets (e.g. audio start-up latency): nudge the audio rate (pitch is preserved) to catch up
  const nudge = Math.abs(drift) > 0.015 && !force ? clamp(-drift * 0.5, -0.03, 0.03) : 0;
  a.playbackRate = video.playbackRate * (1 + nudge);
  if (!video.paused && a.paused) a.play().catch(() => { });
  if (video.paused && !a.paused) a.pause();
}

// ---------- controls ----------
function seek(t) { video.currentTime = clamp(t, 0, state.manifest.n_frames / FPS - 0.001); markDirty(); }
function stepFrame(d) { video.pause(); seek((frameAt(video.currentTime) + d + 0.5) / FPS); }
function setupControls() {
  const toggle = () => (video.paused ? video.play() : video.pause());
  $("#play").onclick = toggle;
  video.onclick = toggle;
  video.addEventListener("play", () => ($("#play").textContent = "❚❚"));
  video.addEventListener("pause", () => ($("#play").textContent = "▶"));
  $("#back5").onclick = () => seek(video.currentTime - 5);
  $("#fwd5").onclick = () => seek(video.currentTime + 5);
  $("#prevf").onclick = () => stepFrame(-1);
  $("#nextf").onclick = () => stepFrame(1);
  $("#goto").addEventListener("change", (e) => { video.pause(); seek((+e.target.value + 0.5) / FPS); });
  $("#speed").addEventListener("change", (e) => { video.playbackRate = +e.target.value; });
  const win = $("#window"); win.value = String(prefs.get("window", 10)); state.windowS = +win.value;
  win.addEventListener("change", () => { state.windowS = +win.value; prefs.set("window", state.windowS); markDirty(); });
  const ov = $("#overlaysel");
  ov.value = prefs.get("overlay", ""); ov.addEventListener("change", () => { prefs.set("overlay", ov.value); markDirty(); });
  document.addEventListener("keydown", (e) => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
    if (e.key === " ") { e.preventDefault(); toggle(); }
    else if (e.key === "ArrowLeft") seek(video.currentTime - (e.shiftKey ? 30 : 5));
    else if (e.key === "ArrowRight") seek(video.currentTime + (e.shiftKey ? 30 : 5));
    else if (e.key === ",") stepFrame(-1);
    else if (e.key === ".") stepFrame(1);
  });
  const seekbar = $("#seekbar");
  const seekTo = (e) => seek((e.offsetX / seekbar.clientWidth) * state.manifest.n_frames / FPS);
  seekbar.addEventListener("mousedown", (e) => { seekTo(e); const mv = (ev) => seekTo(ev); seekbar.addEventListener("mousemove", mv); window.addEventListener("mouseup", () => seekbar.removeEventListener("mousemove", mv), { once: true }); });
  canvas.addEventListener("click", (e) => {
    if (e.offsetX < GUTTER) return;
    seek(state.t0 + (e.offsetX - GUTTER) / (canvas.clientWidth - GUTTER - 8) * (state.t1 - state.t0));
  });
  canvas.addEventListener("wheel", (e) => {
    if (!(e.ctrlKey || e.metaKey)) return;         // pinch / ctrl+wheel zooms the window
    e.preventDefault();
    state.windowS = clamp(state.windowS * Math.exp(e.deltaY * 0.01), 1, 3600); markDirty();
  }, { passive: false });
  canvas.addEventListener("mousemove", (e) => {
    const t = state.t0 + (e.offsetX - GUTTER) / (canvas.clientWidth - GUTTER - 8) * (state.t1 - state.t0);
    const l = state.layout.find((l) => e.offsetY >= l.y && e.offsetY < l.y + l.h);
    const info = l ? RENDERERS[l.tr.meta.type].hover?.(l.tr, t, e.offsetY - l.y) : "";
    canvas.title = e.offsetX >= GUTTER ? `p${frameAt(t)}  ${fmtTime(t)}${info ? "\n" + info : ""}` : info || "";
  });
  window.addEventListener("resize", markDirty);
}

// ---------- main loop ----------
function tick() {
  const t = video.currentTime;
  if (t !== state.lastT || state.dirty) {
    const p = frameAt(t), [a, b] = state.manifest.analysis;
    $("#clock").textContent = `p ${p.toLocaleString()} / ${(state.manifest.n_frames - 1).toLocaleString()} · ${fmtTime(t)}` +
      (p < a || p > b ? " · outside analysis window" : "");
    drawTimeline(t); drawSeekbar(t); drawNow(t);
    state.lastT = t; state.dirty = false;
  }
  if (!video.paused) syncAudio();
  requestAnimationFrame(tick);
}

async function main() {
  status("loading manifest…");
  state.manifest = await (await fetch("/api/manifest")).json();
  video.src = state.manifest.video;
  setupAudio(state.manifest.audio);
  for (const meta of state.manifest.tracks) {
    if (!RENDERERS[meta.type]) { console.warn("no renderer for", meta.type); continue; }
    status(`loading ${meta.name}…`);
    const data = await (await fetch(meta.url)).json();
    const tr = { meta, data, enabled: prefs.get(`on:${meta.id}`, !["subs_en", "frame_change"].includes(meta.id)) };
    tr.ui = RENDERERS[meta.type].init(tr);
    state.tracks.push(tr);
  }
  buildFilters();
  setupControls();
  const t = prefs.get("t", 36.72);
  video.addEventListener("loadedmetadata", () => seek(t), { once: true });
  setInterval(() => prefs.set("t", video.currentTime), 2000);
  status(`${state.tracks.length} tracks`);
  requestAnimationFrame(tick);
}
main().catch((e) => { status("error: " + e.message); console.error(e); });
window.viewer = { state, audio, RENDERERS };   // for poking around in the devtools console
