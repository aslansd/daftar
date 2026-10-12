# daftar — logo

## The idea

Three ruled lines of a ledger. The middle one diverges at a marked point.

That is the two things daftar is at once: a **record book** (دفتر) and a
**diff** — two runs that share a history and then split. The amber dot sits
exactly at the divergence, which is the field daftar exists to identify.

It is not a generic fork or branch glyph: the outer rules make it a page, and
the divergence is asymmetric because in a diff one run is the baseline and the
other is the one that moved.

## Files

| File | Use |
|---|---|
| `daftar-mark.svg` | Transparent. Default for docs and slides. |
| `daftar-mark-on-light.svg` | Cream rounded square. |
| `daftar-mark-on-dark.svg` | Ink rounded square. **GitHub/PyPI avatar.** |
| `daftar-mark-compact.svg` | Simplified. **Use below 32 px** — the outer rules disappear at favicon size. |
| `daftar-mark-compact-on-dark.svg` | Same, on ink. |
| `daftar-mark-mono-black.svg` / `-white.svg` | One colour. The node becomes a ring so the divergence survives without the amber. Print, stamps, embroidery. |
| `daftar-lockup.svg` | Mark + wordmark. README headers. |
| `daftar-lockup-on-dark.svg` | Same, for dark backgrounds. |
| `daftar-lockup-persian.svg` | With دفتر beneath. Good for the Persian-language contexts and the startup form. |

PNG exports: `favicon-16/32/48`, `apple-touch-icon-180`,
`daftar-mark-256/1024`, `daftar-avatar-460`, `daftar-lockup-1200`.

## Palette

| | Hex | Why |
|---|---|---|
| Ink | `#16222E` | Near-black with a blue cast. Reads as ink rather than screen-black. |
| Amber | `#E0913A` | The divergence point. Warm and ledger-ish; legible on both light and dark. |
| Cream | `#FBFAF7` | Paper, not screen-white. |

Amber is used for exactly one thing: the point where two runs diverge. Do not
use it decoratively — its whole job is to mark that spot.

## Rules

- **Clear space**: keep at least the height of one ruled line (8 units on the
  64-grid) clear on every side.
- **Minimum sizes**: primary mark 32 px; compact mark 16 px.
- **Do not** recolour the node to anything but amber, stretch the mark
  non-uniformly, add a shadow, or rotate it. The horizontal rules have to read
  as ruled lines.
- **Do not** place the transparent mark on a mid-tone background — use the
  on-light or on-dark plate instead.

## Wordmark

Set in **Vazirmatn SemiBold**, converted to outlines, so the SVGs render
identically anywhere with no font installed. Vazirmatn was chosen because it is
a well-drawn modern face with both Latin and Persian, which suits a name that
exists in both.

The Persian دفتر is shaped with HarfBuzz — connected contextual forms, correct
right-to-left order. If you regenerate it, do not shape it by hand: naive
character-by-character lookup produces disconnected letters in reverse order.

## Regenerating

```bash
python build_kit.py     # needs cairosvg, fontTools, uharfbuzz, Vazirmatn
```

Everything is geometry on a 64-unit grid; there are no raster sources.

## Licence

Same as daftar: Apache 2.0.
