# Design QA — memory virtualization diagram

## Visual target

- Source: local Codex attachment `codex-clipboard-c59ef03e-aa05-4682-a1c8-164d7e97f288.png`
- Source dimensions: 1586 × 992 px
- Implemented page viewport: 1600 × 960 CSS px
- Desktop capture: local Codex visualization `desktop-user-icons-final.jpg`
- Mobile capture: local Codex visualization `mobile-user-icons-final.jpg`
- State: the E07 transfer sequence repeats every 4.6 seconds and can be paused with the visible motion control.

The source attachment and QA captures stay outside the repository because they are local review artifacts.

## Latest correction

1. Replaced the one-shot motion with a continuous loop while preserving `prefers-reduced-motion` and adding a native pause/play control.
2. Replaced the former arrow asset with the official Phosphor bold caret, corrected accidental multi-sided dashed borders, and centered both arrowheads on their three-pixel routes. The measured center delta is at most 0.008 px.
3. Rounded the orthogonal connector elbows to follow the accepted visual direction more closely.
4. Replaced the generic hardware glyphs with the exact user-supplied PNG assets for GPU, RAM, and NVMe. The GPU retains its supplied color; monochrome RAM and NVMe artwork receive only a contrast filter on the dark background.

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
