# Contributing to SynapticOS

Thanks for your interest. SynapticOS is in active early development
and the process below is intentionally lightweight; expect it to
firm up as the project approaches 1.0.

## Ground rules

- License: Apache-2.0. Every source file carries an SPDX header
  (`/* SPDX-License-Identifier: Apache-2.0 */`). By contributing
  you agree your changes are provided under the same license.
- Branches: `main` carries releases; day-to-day development happens
  on `dev`. Open pull requests against `dev`.
- Warnings are errors (`-Werror`); a change that introduces any
  compiler warning does not merge.
- Building an application ON TOP of SynapticOS rather than changing
  it? Start from `template/` — an out-of-tree app workspace with its
  own west manifest; you do not need to fork this repo.

## Code style

- Zephyr's C style: tabs for indentation, K&R braces, 80-column
  target, `snake_case`. When in doubt, match the file you are in.
- Public API lives in `include/synaptic/` and is versioned;
  breaking changes only land with a version bump. Private headers
  stay next to their subsystem under `src/`.
- Comments explain constraints the code cannot express (hardware
  errata, ordering requirements, protocol invariants) — not what
  the next line does.
- Log strings must not contain em-dash characters; the twister
  console parser treats them as unexpected bytes.

## Commits

Conventional commits, matching the existing history:

```
feat(infer): deadline-aware dispatch and layer-boundary preemption
fix(ota): drain the in-flight cross-core inference before begin()
docs(readme): phase 4 results
test(store): power-loss injection during registry commit
```

Keep one logical change per commit. The body explains what and why;
measured numbers beat adjectives, and honest misses are recorded as
misses.

## Testing

- Every feature or fix comes with tests. Unit tests run on QEMU:

  ```
  west twister -T synaptic-os/tests -p qemu_cortex_m3
  ```

  Both test applications (`tests/unit`, `tests/unit_store`) must
  pass — the suite is green at 100% or it does not merge.
- QEMU constraints: 64 KB RAM (arenas of 4-8 KB, watch new statics),
  no `k_busy_wait` (use a volatile loop), timing driven by icount.
- Changes to flash, dual-core boot, OTA, or the watchdog need a
  hardware pass on the FRDM-MCXN947 before release; note in the PR
  if you could not run one and it will be covered in the next board
  session.
- Line coverage of `src/core` is tracked with
  `tools/syn_coverage.sh`; do not regress it materially.

## Hardware lessons (do not relearn these)

- Never release CPU1 into erased flash; blank-check first. The
  failure wedges the whole chip including the debug port.
- Never issue one multi-sector flash erase; erase sector-by-sector
  with breathing room. Same failure mode.
- Anything streaming into the shell needs the enlarged RX ring; the
  default 64-byte ring drops bytes at UART rate.

## Pull requests

- Describe what changed, why, and how it was verified (twister
  output, board transcript when relevant).
- PRs that change public behavior update the matching docs
  (`README.md`, `CHANGELOG.md` under Unreleased).
- CI equivalent today is the twister suite plus pristine builds of
  the samples for both targets (and the `template/` app when the
  public API or module glue changed); run them locally before
  requesting review.

## Reporting issues

Open a GitHub issue with the board or QEMU target, the commit hash,
what you expected, what happened, and a serial log if you have one.
For suspected security issues in the OTA or update path, please use
GitHub's private vulnerability reporting instead of a public issue.
