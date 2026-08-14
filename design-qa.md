# Design QA — memory virtualization diagram

## Visual target

- Source: local Codex attachment `codex-clipboard-c59ef03e-aa05-4682-a1c8-164d7e97f288.png`
- Source dimensions: 1586 × 992 px
- Implemented page viewport: 1600 × 960 CSS px
- Desktop capture: local Codex visualization `moevm-route-fix-final/desktop-e01-aligned-final.jpg`
- Mobile capture: local Codex visualization `moevm-route-fix-final/mobile-aligned-final.jpg`
- Full-view comparison: local Codex visualization `moevm-route-fix-final/reference-vs-final.png`
- State: the E07 transfer sequence repeats every 4.6 seconds and can be paused with the visible motion control.

The source attachment and QA captures stay outside the repository because they are local review artifacts.

## Latest correction

1. A P1 mismatch remained after the first animation pass: the cyan route visibly targeted E01 while E07 lit in VRAM. The active pulse now belongs to E01, matching the accepted reference and the route endpoint; E07 stays unlit in VRAM.
2. The connector elbows previously overlapped by one radius, creating a visible hook, while the coral source leg ended above the NVMe E07 cell. The bridge now stops at the source elbow, both routes terminate on their intended expert cells, and the coral source reaches E07 exactly.
3. The small caret was replaced with the official Phosphor bold full-arrow asset. Desktop target/source center deltas are at most 0.0313 px; the promote leg ends 1.1875 px before the RAM cell to preserve its border, and the prefetch leg meets the NVMe E07 edge exactly.
4. The continuous 4.6-second loop, pause/play control, reduced-motion behavior, and exact user-supplied GPU/RAM/NVMe assets remain unchanged.

## Fidelity surfaces

| Surface | Result |
| --- | --- |
| Typography | Hero line breaks, tier hierarchy, labels, and legend visually match the reference at 1600 × 960. |
| Layout and spacing | Three hardware cards, expert rows, connector gaps, and legend align with the accepted composition. |
| Color | Dark navy background, cyan VRAM/promote path, coral E07/prefetch path, and restrained gray borders match the target palette. |
| Icons and imagery | Exact user-supplied GPU, RAM, and NVMe PNG files are used. Asset SHA-256 values are pinned by tests; attribution/provenance is documented. |
| Motion | Source, cache, active expert, route, and packet phases repeat on a shared 4.6-second timeline; pause/play and reduced-motion behavior are available. |
| Copy and content | Tier names, speed/capacity labels, expert IDs, route legend, and technical hierarchy follow the accepted reference. Existing product copy and safety claims remain unchanged. |

## Responsive and accessibility checks

- Desktop checked at 1600 × 960 with no horizontal overflow.
- Mobile checked at 390 × 844 with cards stacked safely and all hardware icons scaled without distortion.
- At 390 px, both 24 px arrow assets and their three-pixel target lines share the same 187.5 px center; there is no horizontal overflow.
- The figure has an accessible caption and description; decorative diagram components are hidden from assistive technology where redundant.
- The pause/play mechanism is a native checkbox with an accessible label and visible keyboard focus state.
- `prefers-reduced-motion` disables expert pulses and route animation and hides the unnecessary motion control.
- No browser console errors or warnings were observed.

## Automated checks

- Static-site tests: 13 passed.
- Full repository suite: 175 passed, 30 skipped.
- Ruff check: passed.
- Git whitespace check: passed.

final result: passed
