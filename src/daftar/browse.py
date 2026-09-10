"""A local HTML view over the run store.

``daftar list`` and ``daftar vary`` are fine for tens of runs and poor for
hundreds. This writes a **single self-contained HTML file** — no server, no
network, no dependencies, no build step — that you open in a browser.

Self-contained matters more than it sounds. The file embeds the manifests as
JSON, so it keeps working when emailed to a collaborator, attached to a paper,
or opened in five years on a machine that has never heard of daftar. That is the
same property the manifest format itself has, and breaking it here to get a
nicer table would be a poor trade.

Everything the browser does — filtering, sorting, selecting two runs and
diffing them — happens client-side over the embedded data, using the same
cause/effect split the ``diff`` command uses.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .manifest import (
    CAUSE_NAMESPACES, EFFECT_NAMESPACES, NEUTRAL_NAMESPACES, namespace_of,
)
from .store import RunStore

_ALWAYS_IGNORE = ["meta.run_id", "meta.started_at", "meta.finished_at",
                  "cost.hostname"]


def build_html(store: RunStore, limit: int | None = None,
               label: str | None = None) -> str:
    """Render the store as one self-contained HTML document."""
    manifests = store.list(limit=limit, label=label)
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "store": str(store.dir),
        "causes": list(CAUSE_NAMESPACES),
        "effects": list(EFFECT_NAMESPACES),
        "neutral": list(NEUTRAL_NAMESPACES),
        "ignore": _ALWAYS_IGNORE,
        "runs": [
            {
                "id": m.run_id,
                "label": m.label,
                "started": m.started_at,
                "status": m.get("meta.status", "unknown"),
                "duration": m.get("cost.wall_clock_s", ""),
                "commit": m.get("code.commit_short", ""),
                "dirty": m.get("code.dirty", ""),
                "entrypoint": m.get("code.entrypoint", ""),
                "fields": {k: m.fields[k] for k in m.ordered_keys},
            }
            for m in manifests
        ],
    }
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return _TEMPLATE.replace("__DAFTAR_DATA__", data).replace(
        "__DAFTAR_COUNT__", str(len(payload["runs"]))
    ).replace("__DAFTAR_STORE__", html.escape(str(store.dir)))


def write_html(store: RunStore, path: str | Path, limit: int | None = None,
               label: str | None = None) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(store, limit=limit, label=label), encoding="utf-8")
    return out


_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>daftar - run browser</title>
<style>
:root{--bg:#fbfbfa;--fg:#1b1b18;--muted:#6b6b63;--line:#e3e3de;--card:#fff;
--cause:#8a5a00;--effect:#005f73;--ok:#2d6a4f;--bad:#9b2226;}
*{box-sizing:border-box}
body{margin:0;font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",
Roboto,sans-serif;background:var(--bg);color:var(--fg)}
header{padding:20px 24px 14px;border-bottom:1px solid var(--line);background:var(--card)}
h1{margin:0;font-size:17px;font-weight:650;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:12px;margin-top:3px}
main{padding:18px 24px 48px;max-width:1500px}
.bar{display:flex;gap:10px;align-items:center;margin-bottom:14px;flex-wrap:wrap}
input[type=search],select{padding:6px 10px;border:1px solid var(--line);
border-radius:7px;font:inherit;background:var(--card)}
input[type=search]{min-width:260px}
button{padding:6px 13px;border:1px solid var(--line);border-radius:7px;
background:var(--card);font:inherit;cursor:pointer}
button:hover{background:#f2f2ef}
button:disabled{opacity:.4;cursor:default}
button.primary{background:var(--fg);color:var(--bg);border-color:var(--fg)}
table{width:100%;border-collapse:collapse;background:var(--card);
border:1px solid var(--line);border-radius:9px;overflow:hidden}
th,td{text-align:left;padding:8px 11px;border-bottom:1px solid var(--line);
font-size:13px;vertical-align:top}
th{font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.04em;
color:var(--muted);cursor:pointer;user-select:none;white-space:nowrap}
tr:last-child td{border-bottom:none}
tr.sel{background:#eef4f6}
td.mono,th.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.pill{display:inline-block;padding:1px 7px;border-radius:20px;font-size:11px;
border:1px solid var(--line)}
.ok{color:var(--ok);border-color:#bcd9c8}.bad{color:var(--bad);border-color:#e7c2c3}
.panel{margin-top:18px;background:var(--card);border:1px solid var(--line);
border-radius:9px;padding:16px}
.panel h2{margin:0 0 4px;font-size:14px}
.verdict{margin:10px 0 14px;padding:9px 12px;border-radius:7px;background:#f4f4f1;
font-size:13px}
.grp{margin:14px 0 4px;font-size:11px;text-transform:uppercase;
letter-spacing:.04em;color:var(--muted)}
.chg{display:grid;grid-template-columns:minmax(0,22rem) 1fr;gap:10px;
padding:3px 0;font-family:ui-monospace,Menlo,monospace;font-size:12px}
.k.cause{color:var(--cause)}.k.effect{color:var(--effect)}.k.other{color:var(--muted)}
.old{color:var(--bad)}.new{color:var(--ok)}.arrow{color:var(--muted);margin:0 6px}
.empty{color:var(--muted);padding:22px;text-align:center}
details{margin-top:9px}summary{cursor:pointer;color:var(--muted);font-size:12px}
.kv{display:grid;grid-template-columns:minmax(0,22rem) 1fr;gap:10px;
font-family:ui-monospace,Menlo,monospace;font-size:12px;padding:2px 0}
.kv .k{color:var(--muted)}
</style></head><body>
<header>
  <h1>daftar &mdash; run browser</h1>
  <div class="sub">__DAFTAR_COUNT__ runs &middot; __DAFTAR_STORE__ &middot;
  self-contained: no server, no network</div>
</header>
<main>
  <div class="bar">
    <input type="search" id="q" placeholder="filter by label, id, or field value">
    <select id="lbl"><option value="">all labels</option></select>
    <select id="st"><option value="">any status</option>
      <option value="completed">completed</option><option value="failed">failed</option></select>
    <button id="cmp" class="primary" disabled>compare selected</button>
    <button id="clr" disabled>clear</button>
    <span class="sub" id="hint">select two runs to diff</span>
  </div>
  <table id="tbl"><thead><tr>
    <th data-k="id" class="mono">run</th><th data-k="label">label</th>
    <th data-k="status">status</th><th data-k="started">started</th>
    <th data-k="duration">secs</th><th data-k="commit" class="mono">commit</th>
    <th>fields</th></tr></thead><tbody id="rows"></tbody></table>
  <div id="out"></div>
</main>
<script>
const DATA = __DAFTAR_DATA__;
const sel = [];
let sortKey = "started", sortAsc = false;

const nsOf = k => k.split(".")[0];
const kind = k => DATA.causes.includes(nsOf(k)) ? "cause"
  : DATA.effects.includes(nsOf(k)) ? "effect"
  : DATA.neutral.includes(nsOf(k)) ? "neutral" : "other";
const esc = s => String(s).replace(/[&<>"]/g, c =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

function visible(){
  const q = document.getElementById("q").value.toLowerCase();
  const lbl = document.getElementById("lbl").value;
  const st = document.getElementById("st").value;
  return DATA.runs.filter(r => {
    if (lbl && r.label !== lbl) return false;
    if (st && r.status !== st) return false;
    if (!q) return true;
    if ((r.label+" "+r.id+" "+r.entrypoint).toLowerCase().includes(q)) return true;
    return Object.entries(r.fields).some(([k,v]) =>
      (k+" "+v).toLowerCase().includes(q));
  }).sort((a,b) => {
    const x = a[sortKey] ?? "", y = b[sortKey] ?? "";
    return (x < y ? -1 : x > y ? 1 : 0) * (sortAsc ? 1 : -1);
  });
}

function render(){
  const rows = document.getElementById("rows");
  const list = visible();
  rows.innerHTML = list.length ? "" :
    '<tr><td colspan="7" class="empty">no runs match</td></tr>';
  for (const r of list){
    const tr = document.createElement("tr");
    if (sel.includes(r.id)) tr.className = "sel";
    const ok = r.status === "completed";
    tr.innerHTML =
      `<td class="mono">${esc(r.id)}</td><td>${esc(r.label||"—")}</td>`+
      `<td><span class="pill ${ok?"ok":"bad"}">${esc(r.status)}</span></td>`+
      `<td class="mono">${esc((r.started||"").replace("T"," ").replace("+00:00",""))}</td>`+
      `<td class="mono">${esc(r.duration||"")}</td>`+
      `<td class="mono">${esc(r.commit||"")}${r.dirty==="true"?" *":""}</td>`+
      `<td class="mono">${Object.keys(r.fields).length}</td>`;
    tr.onclick = () => toggle(r.id);
    rows.appendChild(tr);
  }
  const n = sel.length;
  document.getElementById("cmp").disabled = n !== 2;
  document.getElementById("clr").disabled = n === 0;
  document.getElementById("hint").textContent =
    n === 0 ? "select two runs to diff" :
    n === 1 ? "select one more" : "ready — press compare";
}

function toggle(id){
  const i = sel.indexOf(id);
  if (i >= 0) sel.splice(i,1); else { if (sel.length===2) sel.shift(); sel.push(id); }
  render();
}

function diff(a, b){
  const keys = new Set([...Object.keys(a.fields), ...Object.keys(b.fields)]);
  const changes = [];
  let same = 0;
  for (const k of keys){
    if (DATA.ignore.includes(k)) continue;
    const x = a.fields[k], y = b.fields[k];
    if (x === y) same++; else changes.push({k, a:x, b:y, kind:kind(k)});
  }
  changes.sort((p,q) => p.k < q.k ? -1 : 1);
  return {changes, same};
}

function verdict(changes){
  const c = changes.filter(x => x.kind==="cause");
  const e = changes.filter(x => x.kind==="effect");
  if (!changes.filter(x => x.kind!=="neutral").length)
    return ["identical","Nothing meaningful moved."];
  if (e.length && !c.length) return ["nondeterministic",
    "Results differ and nothing that could have caused it does. This run is not deterministic — that is a finding, not a glitch."];
  if (e.length && c.length) return ["explained",
    "Results differ, and so do things that could explain it. The candidate causes are where to look."];
  return ["no effect",
    "Inputs or environment differ but the results did not move. Evidence that the result is robust to those changes."];
}

function compare(){
  const a = DATA.runs.find(r => r.id === sel[0]);
  const b = DATA.runs.find(r => r.id === sel[1]);
  const {changes, same} = diff(a,b);
  const [v, text] = verdict(changes);
  const grp = (title, kind) => {
    const items = changes.filter(c => c.kind === kind);
    if (!items.length) return "";
    return `<div class="grp">${title} (${items.length})</div>` + items.map(c =>
      `<div class="chg"><span class="k ${kind}">${esc(c.k)}</span><span>`+
      `<span class="old">${esc(c.a ?? "(absent)")}</span>`+
      `<span class="arrow">→</span>`+
      `<span class="new">${esc(c.b ?? "(absent)")}</span></span></div>`).join("");
  };
  document.getElementById("out").innerHTML =
    `<div class="panel"><h2>${esc(a.id)} → ${esc(b.id)}</h2>`+
    `<div class="sub">${esc(a.label)} · ${esc(a.started)} → ${esc(b.started)}</div>`+
    `<div class="verdict"><b>${v}</b> — ${esc(text)}</div>`+
    grp("candidate causes","cause") + grp("observed effects","effect") +
    grp("other","other") +
    `<div class="sub" style="margin-top:14px">`+
    `${changes.filter(c=>c.kind!=="neutral").length} meaningful field(s) differ, `+
    `${same} identical</div>`+
    `<details><summary>all fields for ${esc(b.id)}</summary>`+
    Object.entries(b.fields).map(([k,val]) =>
      `<div class="kv"><span class="k">${esc(k)}</span><span>${esc(val)}</span></div>`
    ).join("")+`</details></div>`;
  document.getElementById("out").scrollIntoView({behavior:"smooth"});
}

const labels = [...new Set(DATA.runs.map(r => r.label).filter(Boolean))].sort();
document.getElementById("lbl").innerHTML =
  '<option value="">all labels</option>' +
  labels.map(l => `<option>${esc(l)}</option>`).join("");
for (const id of ["q","lbl","st"])
  document.getElementById(id).addEventListener("input", render);
document.getElementById("cmp").onclick = compare;
document.getElementById("clr").onclick = () => {
  sel.length = 0; document.getElementById("out").innerHTML = ""; render(); };
document.querySelectorAll("th[data-k]").forEach(th => th.onclick = () => {
  const k = th.dataset.k;
  if (sortKey === k) sortAsc = !sortAsc; else { sortKey = k; sortAsc = true; }
  render();
});
render();
</script></body></html>
"""
