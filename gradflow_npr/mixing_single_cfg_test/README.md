# Single-configuration check of the Pi / Z analysis chain

```bash
python3 single_cfg_test.py [chroma_txt]        # defaults to the 17-momentum run
```

Runs `npr_analysis.run_mixing_chain` -- the same chain the production path
uses, itself built on `utilities/pytools.py` -- on one configuration instead of
an ensemble of bootstrap samples. Only the loader differs.

## What it does and does not show

It does **not** validate the Chroma port. The port's correlators already agree
with the 2020 QLUA correlators to 4e-13, so any deterministic function of them
agrees too; a Pi/Z comparison on cfg 1600 cannot fail if the correlator
comparison passed.

It **does** check our implementation of the analysis chain -- the amputation,
the tau_1^(3) projection, the A_ab mixing matrix, and the normalisation --
against the published quantities in `Pi_subset.h5`.

The reference is a bootstrap over 67 configurations, so one configuration
scatters around it by roughly sqrt(67) times the bootstrap width. This catches
factor-level and structural errors, not percent-level ones.

## Result

| | cfg 1600 | ensemble | ratio |
|---|---|---|---|
| `Zq` | 0.8639 | 0.8291 | 1.042 |
| `Pi11` | 0.6036 | 0.5680 | 1.063 |
| `Pi12` | -0.1061 | -0.0957 | 1.109 |

Per momentum `Pi11` often agrees to under 1%. `Pi12` is an order of magnitude
smaller and its sign flips at a few individual momenta -- expected scatter for
a small mixing coefficient on one configuration, and the number to trust least
here.

Use the 17-momentum input rather than the `[-1,1]^4` box: the box sits at
`(a p~)^2 < 0.5`, far below the paper's fit window, where contamination and
config-to-config scatter are both much larger. Agreement visibly improves with
momentum.

## Normalisation, the easy thing to get wrong

`analysis.py` multiplies by the hypervolume in three places -- the point-source
loader, `amputate`, and `quark_renorm` -- and those factors cancel exactly.
`pytools` carries no V factors, so feeding it the **raw** correlators
reproduces `analysis.py`'s Gamma and Zq identically. Do not reintroduce the
factor. `Pi11 ~ 0.6` rather than `0.6 * V` is the check that it is right.
