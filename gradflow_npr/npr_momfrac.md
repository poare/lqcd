# RI'/MOM for the quark EMT on exponential clover configuratrions

## Pipeline
The pipeline has a few parts. I've got the first one working so far, and am in the process of testing the others.
1. Computation of momentum-projected two- and three-point point functions $S(p)$ and $G_{\mu\nu}(p)$. Note that I'm only looking at the symmetric, traceless $H(4)$ irrep, so only computing the three-point function for $\mu = \nu\in \{1, 2, 3, 4\}$. 
2. Projection onto the continuum tensor structure and computation of $\mathcal Z_q(p)$ and $\mathcal Z_{\mathcal O}(p)$ for each ensemble, described in https://arxiv.org/abs/2009.05522. 
3. Running and matching of the RCs to $\mu = 2\, \mathrm{GeV}$ for conversion from RI'/MOM to $\overline{\mathrm{MS}}$. 
4. Fitting of hypercubic discretization artifacts via the fitting procedure described in https://arxiv.org/abs/2009.05522. This part will probably be the most difficult. 

## 1. Computation of correlators via Chroma port of QLUA code

Computes the momentum-projected propagator `S(p)` and the four diagonal
operators $O_{\mu\mu}$ needed for the RI'/MOM renormalisation of the quark EMT. This is a port of the QLUA code used in Detmold et al. 2020, and it reproduces that code's output to 4e-13 (see *Duplication test* below). The two files are the measurement code and an XML template for Chroma input.

| Filename | Purpose |
|---|---|
| `npr_momfrac_chroma.cc` | Measurement functions |
| `npr_momfrac.ini.xml` | the input template, one configuration per run |

#### Compiling

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

#### Switches

**Configuration** — the `<Cfg>` block at the bottom:

```xml
<Cfg>
  <cfg_type>SCIDAC</cfg_type>
  <cfg_file>/path/to/cfg.lime</cfg_file>
</Cfg>
```

The configuration must already be smeared and Landau gauge-fixed; this code does neither (Chroma can do the gauge fixing itself via `COULOMB_GAUGEFIX` with `j_decay = -3`.) In 2020 I used GLU to do this, I think you even showed me how.

**Action and inverter** — `<FermionAction>` and `<InvertParam>`. These are built from Chroma's own XML factories, so **swapping the action is an edit to this block and nothing else** — no code change. For the exponential clover, replace `<FermAct>CLOVER</FermAct>` and its parameters with yours, and keep `<FermionBC>` consistent with `<bvec>` (see below).

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

**Momenta** — `<mom_range>`, an inclusive per-direction box. This is a max norm $p^2\leq \mathrm{mom\_range}$, *not* a taxicab ball: a symmetric box of half-width `n` gives `(2n+1)^4` momenta. The template ships with `n = 1`, i.e. 81 momenta; `n = 6` is the 28561 the 2020 paper used. `<mom_list>` takes an explicit list instead, for testing. Giving both is refused rather than silently resolved.

**`<bvec>`** is the momentum twist, `(0,0,0,1/2)` for antiperiodic time, which has to agree with the time component of `<FermionBC>/<boundary>`.

#### Output

Text, one line per entry:

```
<tag> k0 k1 k2 k3 spin_row spin_col colour_row colour_col re im
```

with `<tag>` one of `prop`, `O11`, `O22`, `O33`, `O44`. 

#### Duplication test

From the 2020 project I was able to find some output HDF5 files with values of the momentum-projected propagator $S(p^2)$ and the three-point correlators $G_{11}(p^2), G_{22}(p^2), G_{33}(p^2), G_{44}(p^2)$ that were used in the project. These are all the outputs of the original QLUA script that I used for the project. I used configuration 1600 from ensemble `cl3_16_48_b6p1_m0p2450` for comparison, since I'm having some trouble getting the original QLUA script up and running. The original data was also computed with a point source, which luckily had been stored per configuration with the data, so the Chroma reproduction code uses this point source value as well, which is $y^\mu = (14, 6, 7, 37)$. I ran a few tests on this configuration, which are stored in the `duplication_test/` directory and reproduce the 2020 QLUA result in all cases:
1. 5 momenta per $S(p)$ and $G_{\mu\mu}(p)$. Two momenta set to $(1, 0, 0, 0)$ and $(2, 2, 2, 2)$ to test the unit vector and diagonal case, and 3 momenta randomly chosen for each operator. The worst relative deviation here between the ported Chroma code and the old 2020 QLUA output is *4.4e-13* across all 25 (operator tag, momentum) comparisons.
2. All momenta with components $p_i\in \{-1, 0, 1\}$, for a total of $3^4 = 81$ unique momenta. 

This has me confident that Claude ported the code to Chroma correctly, and the `README.md` says how to re-run it if you want to tinker with anything.

#### Next steps

Next we need to run this on the expontial clover configurations. It computes correlators through the operator, so we might need a fair number of configurations. In the 2020 project I used 101 configs. 

We also might need to be careful with the momentum budget, since your lattices are significantly larger than the $16^3\times 48$ and $24^3\times 24$ clover configuration we used in 2020. In that project, the lattice was small enough we were able to just compute the RCs at each $p^2\leq $ some cutoff. Your ensembles have many, many more available momenta, so we might need to be more judicious with the momenta and restrict the fitting procedure. The advantage of the 2020 approach was that we could fit higher order discretization artifacts like $p^{[4]} / (p^2)^2$: instead we might want to consider only using the most democratic momenta, i.e. $h(p^2) := p^{[4]}/(p^2)^2 \leq 0.3$ and making the discretization artifact fit simpler. 

I think this is the only part of the pipeline that actively needs to be changed: the rest should factor right through. Once we modify the chroma script to use the exponential clover action and your ensembles we can get some timings and decide which momenta we want to run. 

## 2. Projection onto the continuum irrep structure

TODO

## 3. Running and matching to $\mu = 2\,\mathrm{GeV}$

TODO

## 4. Fitting of discretization artifacts

TODO
