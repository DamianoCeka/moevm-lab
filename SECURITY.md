# Security policy

## Supported versions

Security fixes are applied to the latest tagged pre-release and the `main` branch. Older research snapshots are not maintained.

Do not commit:

- API keys, access tokens or model-provider credentials;
- private prompts, user content or unredacted routing traces;
- model weights whose license does not permit redistribution;
- arbitrary checkpoint loaders that execute remote code;
- benchmark scripts that download and run untrusted artifacts automatically.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Contact the repository owner through the [DamianoCeka GitHub profile](https://github.com/DamianoCeka), or use [private vulnerability reporting](https://github.com/DamianoCeka/moevm-lab/security/advisories/new) when that GitHub feature is available.

Include affected version or commit, impact, reproduction steps and any suggested mitigation. The target is to acknowledge a complete report within seven days, agree on a disclosure plan, and publish a fix before technical details are made public.
