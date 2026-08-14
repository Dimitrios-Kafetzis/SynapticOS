<!-- Target the dev branch; main carries releases. -->

## What & why

<!-- One logical change. What changed, why, and any measured numbers.
     Honest results: misses recorded as misses, stub-NPU numbers
     labeled as such. -->

## Verification

<!-- Paste the relevant tail of the twister run / build output, and a
     board transcript when the change touches flash, dual-core boot,
     OTA, or the watchdog. -->

- [ ] `west twister -T synaptic-os/tests -p qemu_cortex_m3` green (both test apps, 100%)
- [ ] Pristine sample builds pass for QEMU and FRDM (`-Werror`; include the Neutron variants if the NPU/DSP path changed)
- [ ] `template/` app still builds if the public API or module glue changed
- [ ] Commits follow the conventional style in CONTRIBUTING.md (one logical change per commit)
- [ ] Docs updated where public behavior changed (`README.md`, `CHANGELOG.md` under Unreleased)
- [ ] Board pass done, or noted here as pending the next board session (flash / dual-core / OTA / watchdog changes only)

## Notes for reviewers

<!-- Anything non-obvious: constraints honored (frozen public headers,
     QEMU 64 KB RAM, em-dash-free log strings), known limitations,
     follow-ups deliberately left out. -->
