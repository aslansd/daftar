"""Generate the daftar logo kit.

The mark: three ruled lines of a ledger, the middle one diverging at a marked
point. It is simultaneously the two things daftar is about -- a record book
(دفتر) and a diff, where two runs share a history and then split. The amber dot
sits exactly at the divergence, which is the field daftar exists to identify.

Everything is geometry on a 64-unit grid and text converted to outlines, so the
SVGs render identically anywhere with no font installed.
"""

from __future__ import annotations

import os

import cairosvg

INK = "#16222E"      # near-black with a blue cast; reads as ink, not pure black
AMBER = "#E0913A"    # the divergence point; warm, ledger-ish, works on both grounds
CREAM = "#FBFAF7"    # off-white; paper rather than screen-white

STROKE = 'fill="none" stroke-linecap="round" stroke-linejoin="round"'
HERE = os.path.dirname(os.path.abspath(__file__))


def _word(name: str) -> tuple[float, str]:
    with open(os.path.join(HERE, f"word-{name}.txt")) as fh:
        width = float(fh.readline().strip())
        return width, fh.read().strip()


# --------------------------------------------------------------------------
# the mark
# --------------------------------------------------------------------------

def mark(ink: str = INK, amber: str = AMBER, sw: float = 4.6) -> str:
    """Primary mark. Three ledger rules; the middle one diverges."""
    return (
        f'<g {STROKE} stroke="{ink}" stroke-width="{sw}">'
        '<path d="M12 15 H44"/>'
        '<path d="M12 49 H44"/>'
        '<path d="M12 32 H27"/>'
        '<path d="M35 32 C41 32 47 24 53 24"/>'
        "</g>"
        f'<circle cx="31" cy="32" r="5" fill="{amber}"/>'
    )


def mark_compact(ink: str = INK, amber: str = AMBER, sw: float = 8.0) -> str:
    """Favicon mark. The outer rules vanish below ~32px, so they are dropped."""
    return (
        f'<g {STROKE} stroke="{ink}" stroke-width="{sw}">'
        '<path d="M10 32 H24"/>'
        '<path d="M37 32 C43 32 49 21 55 21"/>'
        "</g>"
        f'<circle cx="30" cy="32" r="7" fill="{amber}"/>'
    )


def mark_mono(colour: str = INK, sw: float = 4.6) -> str:
    """One colour, for stamps, embroidery, print, and anywhere colour is lost.

    The node becomes a ring so the divergence point survives without the amber
    doing the work.
    """
    return (
        f'<g {STROKE} stroke="{colour}" stroke-width="{sw}">'
        '<path d="M12 15 H44"/>'
        '<path d="M12 49 H44"/>'
        '<path d="M12 32 H26"/>'
        '<path d="M36 32 C42 32 47 24 53 24"/>'
        "</g>"
        f'<circle cx="31" cy="32" r="4.4" fill="none" stroke="{colour}" stroke-width="3"/>'
    )


def square(body: str, bg: str | None = None, size: int = 512) -> str:
    plate = f'<rect width="64" height="64" rx="14" fill="{bg}"/>' if bg else ""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" '
        f'width="{size}" height="{size}">{plate}{body}</svg>'
    )


# --------------------------------------------------------------------------
# lockups
# --------------------------------------------------------------------------

def lockup(ink: str = INK, amber: str = AMBER, bg: str | None = None,
           persian: bool = False) -> str:
    """Mark plus wordmark, horizontally. Optionally the Persian دفتر beneath."""
    cap = 26.0
    lat_w, lat_d = _word("latin")
    scale = cap / 100.0
    word_w = lat_w * scale

    mark_box = 64.0
    gap = 14.0
    text_x = mark_box + gap
    baseline = 40.0

    total_w = text_x + word_w + 12
    total_h = 64.0

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {total_w:.1f} {total_h:.1f}" '
        f'width="{total_w * 4:.0f}" height="{total_h * 4:.0f}">'
    ]
    if bg:
        parts.append(f'<rect width="{total_w:.1f}" height="{total_h:.1f}" fill="{bg}"/>')
    parts.append(mark(ink, amber))
    parts.append(
        f'<g fill="{ink}" transform="translate({text_x:.2f},{baseline:.2f}) '
        f'scale({scale:.6f})">'
        + lat_d.replace(f'scale({scale:.6f},{-scale:.6f})', "scale(1,-1)")
        + "</g>"
    )

    if persian:
        fa_cap = 15.0
        fa_w, fa_d = _word("fa")
        fa_scale = fa_cap / 100.0
        parts.append(
            f'<g fill="{ink}" opacity="0.45" '
            f'transform="translate({text_x:.2f},{baseline + 17:.2f}) '
            f'scale({fa_scale:.6f})">'
            + fa_d.replace(f'scale({fa_scale:.6f},{-fa_scale:.6f})', "scale(1,-1)")
            + "</g>"
        )
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------
# write everything
# --------------------------------------------------------------------------

def main() -> None:
    out = os.path.join(HERE, "kit")
    os.makedirs(out, exist_ok=True)

    files = {
        # marks
        "daftar-mark.svg": square(mark()),
        "daftar-mark-on-light.svg": square(mark(), CREAM),
        "daftar-mark-on-dark.svg": square(mark(CREAM, AMBER), INK),
        "daftar-mark-compact.svg": square(mark_compact()),
        "daftar-mark-compact-on-dark.svg": square(mark_compact(CREAM, AMBER), INK),
        "daftar-mark-mono-black.svg": square(mark_mono("#000000")),
        "daftar-mark-mono-white.svg": square(mark_mono("#FFFFFF")),
        # lockups
        "daftar-lockup.svg": lockup(),
        "daftar-lockup-on-dark.svg": lockup(CREAM, AMBER, INK),
        "daftar-lockup-persian.svg": lockup(persian=True),
    }
    for name, content in files.items():
        with open(os.path.join(out, name), "w") as fh:
            fh.write(content)

    # Raster exports people actually need.
    raster = {
        "favicon-16.png": ("daftar-mark-compact.svg", 16),
        "favicon-32.png": ("daftar-mark-compact.svg", 32),
        "favicon-48.png": ("daftar-mark-compact.svg", 48),
        "apple-touch-icon-180.png": ("daftar-mark-compact-on-dark.svg", 180),
        "daftar-mark-256.png": ("daftar-mark-on-light.svg", 256),
        "daftar-avatar-460.png": ("daftar-mark-on-dark.svg", 460),
        "daftar-mark-1024.png": ("daftar-mark.svg", 1024),
    }
    for name, (src, size) in raster.items():
        cairosvg.svg2png(url=os.path.join(out, src),
                         write_to=os.path.join(out, name),
                         output_width=size, output_height=size)

    for name, src, w in (("daftar-lockup-1200.png", "daftar-lockup.svg", 1200),
                         ("daftar-lockup-on-dark-1200.png",
                          "daftar-lockup-on-dark.svg", 1200)):
        cairosvg.svg2png(url=os.path.join(out, src),
                         write_to=os.path.join(out, name), output_width=w)

    print(f"wrote {len(files)} svg + {len(raster) + 2} png to {out}")


if __name__ == "__main__":
    main()
