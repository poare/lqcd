# RI/MOM for the quark EMT — Chroma measurement

Computes the momentum-projected propagator `S(p)` and the four diagonal
operators `O_mumu` needed for the RI/MOM renormalisation of the quark EMT.
This is a port of the QLUA code used in Detmold et al. 2020, and it reproduces
that code's output to 4e-13 (see *Duplication test* below).

## The two files you need

| | |
|---|---|
| `npr_momfrac_chroma.cc` | the whole measurement, one translation unit |
| `npr_momfrac.ini.xml` | the input template, one configuration per run |

## Compiling

```bash
make            # uses chroma-config from your Chroma install
```

That builds a standalone binary which registers `NPR_MOMFRAC` into a **stock**
Chroma at runtime — no Chroma fork and no rebuild of Chroma itself.

If the fermion action you want lives in another tree (LALIBE, say), that route
will not see it. In that case compile this file into *that* binary with
`-DNPR_MOMFRAC_NO_MAIN` and add `NprMomfrac::registerAll()` to its linkage
hack. The XML is unchanged either way.

```bash
./npr_momfrac -i npr_momfrac.ini.xml -o out.xml
```

## The switches

**Configuration** — the `<Cfg>` block at the bottom:

```xml
<Cfg>
  <cfg_type>SCIDAC</cfg_type>
  <cfg_file>/path/to/cfg.lime</cfg_file>
</Cfg>
```

The configuration must already be smeared and Landau gauge-fixed; this code
does neither. (Chroma can do the gauge fixing itself via `COULOMB_GAUGEFIX`
with `j_decay = -3`, if you need it.) Set `<nrow>` to match.

*One gotcha.* If the file was written by QLUA, QDP++ will refuse to read it —
QLUA puts bare strings in the LIME user-metadata records where XML is
required, and the reader aborts before touching any field data. Fix with

```bash
python3 python_scripts/fix_lime_xml.py in.lime out.lime
```

which leaves the binary data and the SciDAC checksum untouched.

**Action and inverter** — `<FermionAction>` and `<InvertParam>`. These are
built from Chroma's own XML factories, so **swapping the action is an edit to
this block and nothing else** — no code change. For the exponential clover,
replace `<FermAct>CLOVER</FermAct>` and its parameters with yours, and keep
`<FermionBC>` consistent with `<bvec>` (see below).

```xml
<FermionAction>
  <FermAct>CLOVER</FermAct>
  <Mass>-0.2450</Mass>
  <clovCoeff>1.2493</clovCoeff>
  ...
  <FermionBC>
    <FermBC>SIMPLE_FERMBC</FermBC>
    <boundary>1 1 1 -1</boundary>
  </FermionBC>
</FermionAction>
```

**Momenta** — `<mom_range>`, an inclusive per-direction box. This is a max norm
(L-infinity hypercube), *not* a taxicab ball: a symmetric box of half-width `n`
gives `(2n+1)^4` momenta. The template ships with `n = 1`, i.e. 81 momenta;
`n = 6` is the 28561 the 2020 paper used. `<mom_list>` takes an explicit list
instead, for testing. Giving both is refused rather than silently resolved.

Start small and scale up deliberately: **the momentum loop, not the solver, is
the cost.** The five inversions do not depend on the momenta at all (~50 min on
one node at `16^3 x 48`), while each momentum adds a full-volume phase field
and five lattice sums.

**`<bvec>`** is the momentum twist, `(0,0,0,1/2)` for antiperiodic time. It must
agree with the time component of `<FermionBC>/<boundary>`; they are two halves
of one convention and disagreeing silently is a real risk.

## Output

Text, one line per entry:

```
<tag> k0 k1 k2 k3 spin_row spin_col colour_row colour_col re im
```

with `<tag>` one of `prop`, `O11`, `O22`, `O33`, `O44`. No `1/V` normalisation
— the 2020 analysis assumes it is absent.

`prop` is written and flushed straight after the point solve, before the four
sequential solves start, so it is comparable about 40 minutes before a run
finishes. That matters because `prop` needs one inversion and nothing from the
operator, so it isolates the action, the gauge field and the phase convention
from `J_mu` entirely — it is the first thing to check when something disagrees.

## Duplication test

`duplication_test/` reproduces the 2020 QLUA result on configuration 1600 of
the `cl3_16_48_b6p1_m0p2450` ensemble, and is the evidence that this code is
correct. Worst relative deviation **4.4e-13** across 25 (tag, momentum)
comparisons at 17 momenta, for `prop` and all four operators — the level of the
CG residual. Its `README.md` says how to re-run it.

Five of the seven checks need no reference data at all: they run on a unit
gauge field in about a minute and verify the gamma basis (exact, bit-for-bit
against the 2020 analysis convention), the projection phase and twist, the
sequential source, and the full free-field vertex against its analytic tree
form. **Run those first after any change to the action** — they are fast,
deterministic, and they localise a problem far better than a disagreement
against reference data will.

Worth knowing if you adapt the physics: the sequential source uses the **raw**
gauge links and a periodic shift, with the antiperiodic boundary living in the
Dirac operator alone. That is what QLUA did, and `O44` — the entry that would
expose it if wrong — agrees to the same 4e-13 as everything else. Do not pass
`FermState::getLinks()` to `seqSource`.
