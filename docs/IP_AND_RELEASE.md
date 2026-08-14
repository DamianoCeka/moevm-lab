# IP and release notes

MoEVM Lab's community source tree is licensed under the **Apache License,
Version 2.0**. The choice provides permissive copyright terms and an explicit
patent license while keeping possible support, hosted services and separately
developed operational products commercially viable.

This document is project hygiene, not legal advice.

## License boundary

- Apache-2.0 covers project-owned source, documentation and original benchmark
  organization in the current development tree and future distributions made
  from it.
- Version 0.3.0 is the first release prepared under Apache-2.0. Historical tags
  through v0.2.0 and previously built archives retain the license file with
  which they were distributed. Do not rename an old proprietary wheel, sdist,
  ZIP or bundle and present it as an Apache-2.0 artifact.
- The v0.3.0 release artifacts must be rebuilt from the licensed source tree so
  package metadata, `LICENSE`, `NOTICE` and the source tag agree.
- Model weights, third-party code, model-derived material and referenced tools
  retain their own copyrights and licenses. Apache-2.0 does not relicense them.
- Contributions intentionally submitted for inclusion are accepted under
  Apache-2.0 unless a separate written agreement says otherwise.

## Public-release checklist

- keep dated design notes and benchmark artifacts;
- identify existing practice separately from genuinely new techniques;
- document contributors and the origin of every major idea;
- preserve third-party copyright, license and attribution notices;
- remove weights, credentials, private traces and proprietary data;
- publish only rebuilt artifacts whose metadata says `Apache-2.0`;
- obtain qualified legal advice before public disclosure when patent strategy
  matters.

## Staged release

1. Publish the reproducible simulator, trace tooling and synchronous community
   runtime under Apache-2.0.
2. Publish design notes and benchmark evidence with explicit limitations.
3. Accept community contributions under the same inbound license.
4. Keep optional commercial layers separate only where they add operational
   value rather than hiding reproducibility.
