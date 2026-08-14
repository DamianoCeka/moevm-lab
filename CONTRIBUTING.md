# Contributing

Reproducible bug reports, research proposals and focused pull requests are
welcome. By intentionally submitting a contribution for inclusion in MoEVM
Lab, you agree to license it under the Apache License, Version 2.0, without
additional terms or conditions, as described in Section 5 of that license.
Only submit work that you wrote or have the legal right to contribute. Identify
and preserve the provenance and license of any third-party material.

Use the issue forms according to intent: bug reports and research proposals are
community contributions; paid feasibility or integration requests use the
[commercial pilot form](https://github.com/DamianoCeka/moevm-lab/issues/new?template=commercial_inquiry.yml).
That form is public. Never include credentials, private weights, customer data
or confidential prompts. Security reports must follow [SECURITY.md](SECURITY.md)
instead of any public issue form.

For development:

1. create a focused branch;
2. add or update tests;
3. run `python -m unittest discover -s tests -v`;
4. run the reference comparison;
5. include before/after traffic and stall metrics;
6. never describe a simulation as an end-to-end benchmark.

Commit messages should be short and describe one coherent change.
