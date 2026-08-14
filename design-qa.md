# Design QA — memory virtualization diagram

## Visual target

- Source: local Codex attachment `codex-clipboard-c59ef03e-aa05-4682-a1c8-164d7e97f288.png`
- Source dimensions: 1600 × 960 px
- Implemented page viewport: 1600 × 960 CSS px
- Implemented capture: local Codex visualization `desktop-1600x960-v4.png`
- Focused comparison: local Codex visualization `diagram-reference-vs-implementation-final.png`
- Mobile capture: local Codex visualization `mobile-390x844-diagram-centered.png`
- State: the static reference state is preserved; the E07 transfer sequence animates once after load.

The source attachment and QA captures stay outside the repository because they are local review artifacts.

## Comparison history

1. The first implementation lacked hardware icons, used simplified card anatomy, and placed labels inside crooked transfer arrows. Replaced it with three two-column hardware cards, library icons, and clean orthogonal routes.
2. The diagram was initially too narrow and inset relative to the accepted reference. Increased the page canvas, tightened the outer gutter, aligned the hero columns, and matched the 180 px cards with approximately 38 px inter-card gaps.
3. Inner typography and arrowheads were initially too small. Increased tier labels and expert cells, strengthened the three-pixel routes, and made destination arrowheads visible below E01 and E07.

## Fidelity surfaces

| Surface | Result |
| --- | --- |
| Typography | Hero line breaks, tier hierarchy, labels, and legend visually match the reference at 1600 × 960. |
| Layout and spacing | Three 180 px cards, two-column tier anatomy, expert rows, connector gaps, and legend align with the accepted composition. |
| Color | Dark navy background, cyan VRAM/promote path, coral E07/prefetch path, and restrained gray borders match the target palette. |
| Icons and imagery | GPU, CPU, NVMe, and arrow assets come from Phosphor Icons under the bundled MIT license; no CSS or inline-SVG placeholder art is used. |
| Copy and content | Tier names, speed/capacity labels, expert IDs, route legend, and technical hierarchy follow the accepted reference. Existing product copy and safety claims remain unchanged. |

## Responsive and accessibility checks

- Desktop checked at 1600 × 960 with no horizontal overflow.
- Mobile checked at 390 × 844 with cards stacked safely, E07 visible in all three tiers, and in-flow vertical routes.
- The figure has an accessible text description; decorative icons, expert cells, routes, and legend are hidden from assistive technology where redundant.
- `prefers-reduced-motion` disables expert pulses and route animation.
- Navigation anchors and the primary page-top/evidence interactions were exercised successfully.
- No browser console errors or warnings were observed.

## Intentional small differences

- The Phosphor GPU card has two fan circles rather than the single-fan outline in the visual concept.
- The NVMe icon uses a real chip-board library glyph rather than a hand-drawn copy.
- Route elbows are square for crisp CSS geometry instead of the subtly rounded concept path.

These are low-severity icon/connector details and do not change the information hierarchy, motion story, or technical meaning.

## Automated checks

- Static-site tests: 12 passed.
- Full repository suite: 174 passed, 30 skipped.
- Ruff check and format check: passed.
- Git whitespace check: passed.

final result: passed
