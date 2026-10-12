"""gdsfactory adapter — provenance for photonic and analog layout.

**What this adapter is, and what it is not.** It records provenance *about*
layout scripts. It contains no design capability, no process design kit, no
foundry data and no device models. The relationship is the same as the MNE
adapter's to patient recordings: it writes down what produced an artifact
without being able to produce one itself.

That distinction matters here more than elsewhere, and the project roadmap
originally excluded chip design for good reasons. Those reasons were about
*digital EDA flows* — synthesis, place-and-route, commercial PDKs under
foundry NDA, and the export-controlled toolchains that go with advanced-node
logic. gdsfactory is a different thing: an open-source Python library for
scripting **layout geometry**, used mostly in silicon photonics research, and
shipping an open generic PDK. See ROADMAP.md for the full reasoning; the short
version is that recording provenance is not designing, and photonics research
groups are an ordinary academic community.

What a generic tracker misses about a layout run:

* **The GDS file has no idea what made it.** A GDSII stream is geometry and
  nothing else — no script, no parameters, no library version. It is the
  artifact that goes to a foundry, and mask changes are expensive. The content
  hash of the written GDS is the only thing that identifies *which* mask, and
  this adapter records it alongside everything that produced it.

* **The gdsfactory version silently changes geometry.** Component generators
  are library code; a default bend radius or a router heuristic changing between
  releases moves polygons. The same script on two machines can produce different
  masks, and nothing in the output says so.

* **The PDK is the process.** gdsfactory refuses to build without an activated
  PDK, but which one — and which *version* of it — determines layers, cross
  sections and every device. Recorded by name, version, and a hash of the layer
  map.

* **Design parameters live in a decorator cache.** ``@gf.cell`` derives a cell
  name from the function and its arguments, so the name carries a settings hash
  suffix. That is useful and not readable; the settings themselves are recorded
  as fields you can diff.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from typing import Any

from ..run import Run
from .base import AVAILABLE, probe_import, safe

name = "gdsfactory"


def availability() -> tuple[str, str]:
    """``(status, reason)`` -- see ``adapters.base.probe_import``."""
    return probe_import("gdsfactory")


def is_available() -> bool:
    return availability()[0] == AVAILABLE


def _sha(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8", "replace")).hexdigest()[:16]


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------

def describe_environment(run: Run, prefix: str = "gdsfactory") -> None:
    """Record the library stack. Versions here move polygons.

    gdsfactory sits on kfactory which sits on KLayout. A change anywhere in that
    stack can alter generated geometry, and the resulting GDS carries no record
    of any of it.
    """
    from importlib.metadata import version

    for dist, key in (("gdsfactory", "version"),
                      ("kfactory", "kfactory_version"),
                      ("klayout", "klayout_version")):
        run.log_param(f"{prefix}.{key}", safe(lambda d=dist: version(d), "unknown"))


# --------------------------------------------------------------------------
# PDK
# --------------------------------------------------------------------------

def describe_pdk(run: Run, pdk: Any = None, prefix: str = "pdk") -> None:
    """Record which process design kit is active, and its shape.

    gdsfactory raises rather than guessing when no PDK is activated, so a run
    that got this far has one. Which one, and which version, decides the layer
    stack, the cross sections and every device the script can call.
    """
    import gdsfactory as gf

    pdk = pdk if pdk is not None else safe(lambda: gf.get_active_pdk())
    if pdk is None:
        run.log_param(f"{prefix}.active", False)
        return

    run.log_param(f"{prefix}.active", True)
    run.log_param(f"{prefix}.name", safe(lambda: str(pdk.name), "unknown"))
    run.log_param(f"{prefix}.version", safe(lambda: str(pdk.version), "unknown"))

    for attr, key in (("cells", "n_cells"),
                      ("cross_sections", "n_cross_sections"),
                      ("layers", "n_layers")):
        run.log_param(f"{prefix}.{key}",
                      safe(lambda a=attr: len(getattr(pdk, a)), "unknown"))

    # The layer map is the process contract: layer names to (layer, datatype).
    # A changed map means the same script writes different GDS layers.
    def _layer_digest():
        layers = pdk.layers
        items = sorted((str(m.name), int(m.value)) for m in layers)
        return _sha(items)

    digest = safe(_layer_digest)
    if digest:
        run.log_param(f"{prefix}.layer_map_sha256", digest)


# --------------------------------------------------------------------------
# component
# --------------------------------------------------------------------------

def describe_component(component: Any, run: Run, prefix: str = "layout") -> None:
    """Record the design: its generator, settings, ports and geometry summary."""
    run.log_param(f"{prefix}.name", safe(lambda: str(component.name), "unknown"))
    run.log_param(f"{prefix}.function", safe(lambda: str(component.function_name),
                                             "unknown"))

    # The design parameters. gdsfactory hashes these into the cell name, which
    # makes the name a fingerprint but not something a human can read or diff.
    settings = safe(lambda: dict(component.settings))
    if settings:
        for key, value in sorted(settings.items()):
            if value is None:
                continue
            if hasattr(value, "__name__"):
                value = value.__name__
            run.log_param(f"{prefix}.setting.{key}", value)
        run.log_param(f"{prefix}.n_settings", len(settings))

    info = safe(lambda: dict(component.info))
    for key, value in sorted((info or {}).items()):
        run.log_param(f"{prefix}.info.{key}", value)

    # Ports are the interface. A changed port count or position breaks every
    # circuit that instantiates this cell.
    ports = safe(lambda: list(component.ports), [])
    run.log_param(f"{prefix}.n_ports", len(ports))
    if ports:
        run.log_param(f"{prefix}.port_names",
                      safe(lambda: sorted(str(p.name) for p in ports), []))

    bbox = safe(lambda: component.bbox_np().tolist())
    if bbox:
        run.log_param(f"{prefix}.bbox_um",
                      [[round(float(v), 4) for v in row] for row in bbox])
        width = safe(lambda: round(float(bbox[1][0] - bbox[0][0]), 4))
        height = safe(lambda: round(float(bbox[1][1] - bbox[0][1]), 4))
        run.log_result(f"{prefix}.width_um", width)
        run.log_result(f"{prefix}.height_um", height)

    run.log_result(f"{prefix}.n_child_cells",
                   safe(lambda: len(list(component.kdb_cell.each_child_cell())),
                        "unknown"))

    _describe_geometry(component, run, prefix)


def _describe_geometry(component: Any, run: Run, prefix: str) -> None:
    """Polygon counts and area per layer, from a flattened copy.

    Flattening a *copy*: the counts are meaningless at the top level of a
    hierarchical cell, where the polygons live in referenced children, and
    flattening the original would mutate the user's design.
    """
    def _stats():
        flat = component.copy()
        flat.flatten()
        layout = flat.kcl.layout
        per_layer: dict[str, int] = {}
        total = 0
        for index in layout.layer_indexes():
            count = flat.kdb_cell.shapes(index).size()
            if count:
                per_layer[str(layout.get_info(index))] = int(count)
                total += int(count)
        return per_layer, total

    result = safe(_stats)
    if result is None:
        return
    per_layer, total = result

    run.log_result(f"{prefix}.n_polygons", total)
    run.log_result(f"{prefix}.n_layers_used", len(per_layer))
    if per_layer:
        run.log_param(f"{prefix}.layers_used", sorted(per_layer))
        for layer, count in sorted(per_layer.items()):
            run.log_result(f"{prefix}.polygons.{layer}", count)


# --------------------------------------------------------------------------
# netlist
# --------------------------------------------------------------------------

def describe_netlist(component: Any, run: Run, prefix: str = "layout") -> None:
    """Hash the netlist: connectivity, independently of geometry.

    Two layouts can have identical polygon counts and different connectivity —
    a route that changed which ports it joins. The netlist hash catches that
    where a geometry hash alone would only say "something moved".
    """
    def _digest():
        import json
        netlist = component.get_netlist()
        return _sha(json.dumps(netlist, sort_keys=True, default=str))

    digest = safe(_digest)
    if digest:
        run.log_param(f"{prefix}.netlist_sha256", digest)


# --------------------------------------------------------------------------
# the artifact
# --------------------------------------------------------------------------

def gds_digest(component: Any) -> tuple[str, int] | None:
    """``(sha256_prefix, n_bytes)`` of the component's GDS stream.

    Written to a temporary file and hashed. gdsfactory's GDS output is
    deterministic for a given component, so this is a stable identity for the
    mask itself — which is the thing that goes to a foundry and the thing a
    GDSII file cannot tell you anything about.
    """
    def _hash():
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "daftar_probe.gds")
        component.write_gds(path)
        with open(path, "rb") as fh:
            raw = fh.read()
        os.remove(path)
        os.rmdir(directory)
        return hashlib.sha256(raw).hexdigest()[:16], len(raw)

    return safe(_hash)


def write_gds(component: Any, path: Any, run: Run, prefix: str = "layout",
              **kwargs: Any):
    """``component.write_gds()`` with the full design provenance recorded.

    ::

        with daftar.track("mzi-sweep", seed=42) as run:
            c = gf.components.mzi(delta_length=20.0)
            gdsfactory_adapter.write_gds(c, "mzi.gds", run)

    Records the library stack, the active PDK, the design settings, the geometry
    summary and the content hash of the written file. A GDSII stream carries
    none of that, so without this the mask on disk is an orphan.
    """
    describe_environment(run)
    describe_pdk(run)
    describe_component(component, run, prefix)
    describe_netlist(component, run, prefix)

    out = component.write_gds(path, **kwargs)

    def _written():
        with open(str(path), "rb") as fh:
            raw = fh.read()
        return hashlib.sha256(raw).hexdigest()[:16], len(raw)

    written = safe(_written)
    if written:
        digest, size = written
        run.log_result(f"{prefix}.gds_sha256", digest)
        run.log_result(f"{prefix}.gds_bytes", size)
    run.add_output(str(path))
    return out


def build(factory: Any, run: Run, prefix: str = "layout", **kwargs: Any):
    """Call a component factory, recording the arguments and the result.

    ::

        with daftar.track("mzi", seed=0) as run:
            c = gdsfactory_adapter.build(gf.components.mzi, run, delta_length=20.0)

    The arguments you passed are recorded separately from the settings the
    component ended up with, because they are not the same thing: gdsfactory
    fills in defaults from the PDK and from the generator signature, and those
    defaults are exactly what changes between library versions.
    """
    describe_environment(run)
    describe_pdk(run)

    for key, value in sorted(kwargs.items()):
        if hasattr(value, "__name__"):
            value = value.__name__
        run.log_param(f"{prefix}.arg.{key}", value)
    run.log_param(f"{prefix}.factory",
                  safe(lambda: getattr(factory, "__name__", str(factory)),
                       "unknown"))

    component = factory(**kwargs)
    describe_component(component, run, prefix)

    digest = gds_digest(component)
    if digest:
        run.log_result(f"{prefix}.gds_sha256", digest[0])
        run.log_result(f"{prefix}.gds_bytes", digest[1])
    return component


def describe(obj: Any, run: Run, prefix: str | None = None) -> None:
    """Dispatch on whatever gdsfactory object is handed in."""
    cls = type(obj).__name__
    if cls in ("Pdk", "PDK"):
        describe_pdk(run, pdk=obj, prefix=prefix or "pdk")
    else:
        describe_component(obj, run, prefix or "layout")
