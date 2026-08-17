# Design QA — generated artwork and H2D label fix

## Visual targets

- Style reference: user attachment `codex-clipboard-253cd83d-656e-49f9-84ff-d20eac57f675.png`.
  It established the desired premium technical-infographic feel; the third-party
  character, branding, layout, and claims were intentionally not copied.
- Defect reference: user attachment
  `codex-clipboard-cf7ef5cd-bcde-4aaa-a409-602912a38fde.png`.
  It showed the vertical H2D route passing through the `Dedicated H2D stream`
  label.
- Implementation: an original MoEVM hero illustration, an original sparse-expert
  illustration, and an opaque, foreground H2D label that masks the route behind
  it.

The source attachments and browser captures stay outside the repository because
they are local review artifacts.

## Fidelity and intentional differences

| Surface | Result |
| --- | --- |
| Art direction | Passed. Both generated images use the site's dark navy, cyan, and coral hardware language. |
| Originality | Passed. No SpongeBob, third-party character, logo, brand, or in-image text is present. |
| Technical copy | Passed. All claims and labels remain editable HTML rather than generated pixels. |
| H2D label | Passed. The label has an opaque page-colored background, foreground stacking, and a surrounding mask; the vertical route no longer draws through the text. |
| Desktop layout | Passed at 1440 × 1000 with no horizontal overflow, both local images loading, and no console warning/error. |
| Mobile layout | Passed at 390 × 844 with no horizontal overflow; hero art becomes a separate 16:9 panel and the supporting visual stacks above its caption. |
| Motion control | Passed. `Pause motion` changes to `Play motion`, stops the native checkbox state, and restores it on the next click. |

## Accessibility and performance boundaries

- The generated artwork is decorative because the adjacent HTML explains the
  same concepts; both images use empty alternative text to avoid duplicate
  narration.
- Intrinsic dimensions are declared to limit layout shift, and the below-fold
  sparse-expert image is lazy loaded.
- Existing reduced-motion behavior remains intact.
- Assets are self-hosted and permitted by the existing fail-closed CSP; no
  script, analytics, tracking, external image host, or new form was added.
- The PNGs favor source quality and currently add about 3.4 MiB combined. A
  future modern-format optimization can reduce transfer size without changing
  layout.

## Automated checks

- Static-site tests: 14 passed.
- Full repository suite: 282 passed, 73 skipped.
- Ruff check and format check: passed.
- Git whitespace check: passed.
- Browser console: no relevant warnings or errors.

final result: passed
