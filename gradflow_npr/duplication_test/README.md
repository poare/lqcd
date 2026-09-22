# Duplication test: reproducing the 2020 QLUA NPR on cfg 1600

Everything here exists to answer one question — does the Chroma port reproduce
`compute_npr_through_op` from the 2020 QLUA pipeline, entry by entry?

**It does.** Worst relative deviation `4.4e-13` over 25 (tag, momentum)
comparisons at 17 momenta, for `prop` and all four `O_mumu`. That is the level
of the CG residual.

Full reasoning, including what each test rules out and the two traps that cost
real time, is in the research hub at `projects/gf-pdf-npr/notes/old-pipeline.md`
under Stage 2b. This file is just how to re-run it.

## The binary

The measurement is a regular inline measurement in
`lib/meas/inline/hadron/inline_npr_momfrac_w.{h,cc}`, so it runs from a **stock
`chroma` binary that has been rebuilt to include it** — there is no separate
driver any more. The commands below use `$CHROMA`; set it to the binary you
rebuilt:

```bash
export CHROMA=$HOME/libraries/src/chroma/mainprogs/main/chroma
```

To get that binary, from `../chroma_inline/`: copy the two files into
`lib/meas/inline/hadron/`, register them in
`inline_hadron_aggregate.cc`, add them to the two lists in `lib/Makefile.am`,
and **link `-lfftw3`** (Chroma's `configure` has no FFTW option, so the link
fails without it).

## Run everything from this directory

The checking scripts are in `../python_scripts/`. Output paths inside the XML
are relative, so running from here keeps results here.

```bash
cd duplication_test
```

## Tasks 1-5: free field, no reference data, ~1 minute total

These run on a **unit gauge field**. They need neither the 113 MB configuration
nor a machine whose memory has been vindicated, which is why they come first.

```bash
$CHROMA -i test_unit_skeleton.ini.xml -o skeleton.out.xml
$CHROMA -i test_gamma.ini.xml     -o gamma.out.xml   | python3 ../python_scripts/check_gamma.py
$CHROMA -i test_project.ini.xml   -o project.out.xml | python3 ../python_scripts/check_project.py
$CHROMA -i test_seqsrc.ini.xml    -o seqsrc.out.xml  | python3 ../python_scripts/check_seqsrc.py
$CHROMA -i test_fft.ini.xml       -o fft.out.xml     | grep check_fft
$CHROMA -i test_freefield.ini.xml -o freefield.out.xml && python3 ../python_scripts/check_freefield.py freefield.txt
```

`test_unit_skeleton` is the one that "fails": it reaches the measurement, prints
`measurement reached`, then aborts on the absent `<FermionAction>`. That is
expected — the log line is the whole test.

For the record, `test_gamma`, `test_project` and `test_seqsrc` aborted with
`give exactly one of <mom_list> and <ksq_cut>` between `55ae22b` and the fix:
the momentum rework demanded a momentum form even from self-tests that project
nothing. It is now required only when something will actually project, which is
the measurement proper or `test_fft`.

What each one pins down:

| test | establishes |
|---|---|
| `test_unit_skeleton` | the measurement is registered and reached by Chroma's driver |
| `check_gamma` | `Gamma(1 << mu)` equals `analysis.py`'s `gamma[mu]`, bit-for-bit |
| `check_project` | the projection phase, the `b_4 = 1/2` twist and the `y` offset |
| `check_seqsrc` | shift directions, gamma placement, relative sign |
| `check_freefield` | the whole chain: vertex is exactly `-2i sin(p_mu) gamma_mu` |
| `test_fft` | FFT projection against the phase sum on a Gaussian random field, off-origin `tsrc`, negative and wrapped `k`; expect `~1e-13` |

Two of these run **periodic in time with `bvec = 0`, deliberately.** A twisted
plane wave is antiperiodic while QDP++'s `shift` is periodic, so the analytic
reference fails on the boundary slices and `mu=3` misses. That boundary
treatment is *correct* — it is what QLUA did — but it makes the free-field
formula inapplicable, so the twist is tested in `check_project` where it
belongs, and the boundary is tested against real data in tasks 6-7.

## Tasks 6-7: against the reference data, ~52 minutes

These were last run in full with the **old standalone binary**. The in-tree
build has been checked on `test_fft` (identical numbers) and `check_freefield`
(pass), but the cfg 1600 comparison has not been repeated since the measurement
moved into `libchroma`.

**The configuration must be passed through `fix_lime_xml.py` first.** QDP++
parses the LIME user-metadata records and QLUA writes bare strings there, so
the read aborts before touching any field data:

```bash
python3 ../python_scripts/fix_lime_xml.py \
  "$DATA/gf-pdf-npr/npr_conv_tests/cl3_16_48_b6p1_m0p2450_cfg_1600_smeared_gf.lime" \
  "$DATA/gf-pdf-npr/npr_conv_tests/cl3_16_48_b6p1_m0p2450_cfg_1600_smeared_gf_xmlfix.lime"
```

That patched copy already exists; this is only needed for a new configuration.

```bash
$CHROMA -i cfg1600_17mom.ini.xml -o cfg1600_17mom.out.xml > cfg1600_17mom.log 2>&1
```

**`prop` is comparable about 10 minutes in**, roughly 40 minutes before the run
finishes — it is written and flushed straight after the point solve, because it
needs one inversion and nothing from `J_mu`. Watch for
`prop projected and written` in the log, then:

```bash
python3 ../python_scripts/compare_cfg1600.py cfg1600_chroma_17mom.txt \
  "$DATA/gf-pdf-npr/npr_conv_tests/cfg1600.h5" x14y6z7t37 prop mom_assignment.json
```

When all five solves are done:

```bash
python3 ../python_scripts/check_trace.py cfg1600_chroma_17mom.txt
python3 ../python_scripts/compare_cfg1600.py cfg1600_chroma_17mom.txt \
  "$DATA/gf-pdf-npr/npr_conv_tests/cfg1600.h5" x14y6z7t37 prop,O11,O22,O33,O44 mom_assignment.json
```

Run `check_trace.py` **before** reading anything into a disagreement in `O*`. A
trace-subtracted sequential source would pass every downstream `Pi` and `Z`
comparison while failing the raw operator comparison in a way that looks like a
`J_mu` bug. The ratio `|sum_mu O_mumu| / max|O_mumu|` must be O(1), not zero.

## The momenta

`mom_assignment.json` fixes which momenta each tag is compared at: `(1,0,0,0)`
and `(2,2,2,2)` for every tag, plus three drawn at random **per tag** from the
reference file's own 13^4 box, seeded `default_rng(22560)` so the choice is
reproducible. `compare_cfg1600.py` honours that assignment and **fails if a
tag's assigned momenta are missing** rather than reporting a pass over zero
entries.

The sample deliberately includes negative components in all four directions
(the HDF5 keys are bare `%d` concatenation, so `p-6-60-3` is exactly where a
read/write mismatch would hide), the corners of the box where `(a p~)^2` is
largest, and degenerate momenta with zero components.

## Files

| | |
|---|---|
| `test_*.ini.xml` | free-field inputs, tasks 1-5, plus `test_fft.ini.xml` |
| `cfg1600_17mom.ini.xml` | the reference comparison, 17 momenta |
| `cfg1600.ini.xml` | the original two-momentum version |
| `mom_assignment.json` | which momenta each tag is compared at |
| `*.log`, `*.out.xml`, `*.txt` | results, all gitignored |

The production template is **not** here — it is `../npr_momfrac.ini.xml`, since
it is the deliverable rather than a test artifact. Retargeting to a different
fermion action means editing its `<FermionAction>` block and nothing else.
