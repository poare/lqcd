JOBNUM ||| PURPOSE ||| DATE

PION MASS JOBS:
25262 ||| Testing pion 2 point on new smeared / gauge fixed configurations at /data/d10b/ensembles/isoClover/cl3_16_48_b6p1_m0p2450_smeared_gf/landau/ ||| 7/17
25263 ||| Testing pion 2 point on David's smeared / gauge fixed configurations at /data/d10b/users/poare/gf//data/d10b/users/poare/gf/cl3_16_48_b6p1_m0p2450_smeared_gf/landau/ ||| 7/17
EFFECTIVE MASS ON EACH JOB OUTPUTS THE SAME, SO THE CONFIGURATIONS SHOULD BE ABOUT THE SAME.
25278 ||| Testing pion 2 point on coulomb gauge fixed configurations: want to see how the pion two point changes ||| 7/19

NPR MOMFRAC JOBS:
17860 ||| Running test of code through the operator by running code with momentum sources ||| 3/25
17809 ||| Same as 17860 ||| 3/25
17948 ||| (KILLED) Running through the operator code with multiple points sources. Trying to see if the imaginary components are artifacts of the lack of translational invariance on the lattice. ||| 3/26
17951 ||| Same as 17948 but with slight formatting change to make points more readable, cfgs 500 - 800 ||| 3/26
17952 ||| Same as 17951, cfgs 810 - 1100 ||| 3/26
17999 ||| Through the operator, averaged over 20 base points ||| 4/1    NOTE: LIKELY BAD DATA
18759 ||| Born term determination (Three point function is normalized by 1 / 2sqrt(2) ||| 4/4
19045 ||| Running at more momenta ||| 4/6
19134 - 19138 ||| Running untied sources ||| 4/7
19214 ||| Running new wall sources to see the error in Zq ||| 4/7
20361 ||| Running just a point propagator at a large number of momenta to get Zq ||| 4/7
19910 ||| Running npr_momfrac on O11, ..., O44 with randomized seeds ||| 4/7
19970 ||| Born term determination ||| 4/7
20356, 20715 ||| Crosschecking point sources with Phiala ||| 4/8
20401 ||| Rerunning with random seed ||| 4/10
20726, 20730 ||| Crosschecking with Phiala's code again: bug found and fixed ||| 4/13
20733 - 20735 ||| (NOT ACCURATE, THEY DON'T DO A RANDOM POINT, THEY USE THE ORIGIN) Trying to figure out why averaging over points is causing the point sources to overapproximate the wall sources. Running @ 5 momentum (1, 1, 1, 1), ..., (5, 5, 5, 5) that we have wall source values at to try to see what's going on ||| 4/13
20737, 20740 ||| Same as above ||| 4/13
21529 ||| Running my Zq code against Phiala's to try to fix the issue with the single point ||| 4/17
21590 ||| Same, but trying to fix a bug. Have determined that there's an issue somewhere in mine. The current change is to the FTarg (bug still not fixed) ||| 4/17
21592 ||| Current change is to the random numbers ||| 4/18
21664 ||| Same as above, also rerunning Phiala's code (Use this for Zq) ||| 4/18
{ 22454, 22458 - 22460 ||| Running gpu npr_momfrac on 200 configs each with momenta from -6 to 6 in each component ||| 4/24
22556, 22559 - 22560 ||| Same as 22454, but those ran into memory issues because I wasn't saving to d10b ||| 4/27
22606, 22612 - 22616 ||| Same as above, because SLURM killed those jobs ||| 4/28 }
22812 ||| Trying to debug 3 point function: running irreps on QLUA instead of in the analysis ||| 4/29
22855 ||| Running Born term for irreps ||| 4/29
23014, 23015 ||| Vector and axial currents ||| 5/2
23076, 23077 ||| Wilson-Clover current ||| 5/3
23220 ||| Free field currents ||| 5/6
23274 ||| Free field Born term (trying to figure out how to get this projected onto the space spanned by \Lambda^1 and \Lambda^2) ||| 5/8
23278 ||| Free field Born, using simpler contractions before trying to project this onto tensor structure ||| 5/8
23283 ||| Free field Born with full operator structure including trace subtraction. (that works. It shows that the born term is about equal to 2 \Lambda^1, with small corrections in the form of \Lambda^2) ||| 5/8
23294 - 23296 ||| Running npr_momfrac with -1/4 * Dslash in the operator to see if we get any change. ||| 5/8
23476 ||| Running current NPR at different nonzero q vectors (code here is wrong) ||| 5/10
23619 ||| Running current NPR at different q vectors (implementation hopefully fixed) ||| 5/20
23621 - 23624 ||| Running to get ZA from -6 to 6 (THIS IS BUGGY, I WAS RUNNING ON FREE FIELD CONFIGURATIONS) ||| 5/20
23660 ||| Same as above, but hopefully with accurate code ||| 6/1

25291 - 25293 ||| Running npr_momfrac on 24^3 x 24 lattice at 10 configurations and all momenta to get out another estimate of ZO ||| 7/20
25350 - 25352, 25362, 25363 ||| Running same as previous, trying to get more configurations run ||| 7/22
28229 ||| Running npr_momfrac on 32^3 x 48 lattice ||| 9/24
28390 ||| Running current_npr on the lattice to see what a reasonable output for ZV / ZA should be ||| 10/1

Storing all the npr_momfrac output from -6 to 6 in 22454 (/data/d10b/users/poare/lqcd/npr_momfrac/output/cl3_16_48_b6p1_m0p2450_22454).
Useful data sets:
- 20401 has quark renormalization and (vector) operator renormalization for momenta from 0 to 5
- 22454 has quark renormalization and (vector) operator renormalization for momenta from -6 to 6
- 22812 has quark renormalization and (irrep) operator renormalization for momenta from 0 to 5
- 23014 has vector and axial currents for momenta from 0 to 5
- 23476 has vector and axial currents run at nonzero q insertion (CODE IS BAD).
- 23660 should have ZA

GAUGE FIXING JOBS
25277 ||| Smearing cl3_24_24_b6p1_m0p2450 for configs 1000 to 4400 ||| 7/19
25284 ||| Gauge fixing cl3_24_24_b6p1_m0p2450 for configs 1000 to 4400 ||| 7/19

0NUBB JOBS
28215 ||| Running just color-unmixed bilinears to see how fast they run: they run significantly faster (almost 100x faster) than if we include the color mixed operators ||| 9/22
28216 ||| Running all operators (including color-mixed): runs VERY slow ||| 9/22
28230 ||| All color-unmixed operators needed for current renormalization on cl3_16_48_b6p1_m0p2450, cfgs 1000 - 1200 ||| 9/24
28231 ||| Correlators needed for current renormalization on cl3_16_48_b6p1_m0p2450, cfgs 1000 - 1500 ||| 9/24
28337 ||| Running vector and current renormalization. Analysis output isn't great; Zq matches the old Zq factors from RI'-MOM which is good, but ZV and ZA are much too small. Trying to debug now and seeing what part of the code the error is in ||| 9/28
28357 ||| Rerunning and being a bit more careful about the momentum projection steps; I may have previously amputated with the wrong propagators ||| 9/30
28357 ||| Same as before, but changing up the momenta a bit ||| 10/1
28521 ||| Testing q = 0 RI/sMOM vs RI'-MOM. Potential bug fix: remove bvec from momentum projection ||| 10/5
28528 ||| Bug potentially found by removing bvec from momentum projection. Running currents in RI/sMOM ||| 10/5
29053 - 29055 (stored in 29053, BUG HERE) ||| Running 0nubb on cl3_16_48_b6p1_m0p2450 configuration to begin writing / testing the analysis code ||| 11/2
29552 - 29554 ||| Same as before, but trying to fix the bug-- I believe it was in the computation of the Green's functions, I was summing them all up instead of creating separate ones ||| 11/16
29913 - 29916 ||| Still trying to fix the bug, now no longer using e^{-2iqx} for the momentum projection (as in the paper) ||| 11/30

TEST 0NUBB JOBS
30457 ||| Running test script to check what is outputs ||| 12/20
30643 ||| Running free field on QLUA w/ a 4^4 lattice to see if the bug is fixed ||| 12/23
ONCE THIS IS DONE RUNNING, FEED IT THOUGH TO SEE IF WE GET THE CORRECT TREE LEVEL VALUE. THEN DO THE SAME ON THE 16^3 X 48 LATTICE, AND IF THESE TESTS ALL WORK THEN USE ACTUAL CONFIGURATIONS.
30661 ||| Something is wrong with the QLUA code, it's not matching up with the Python output for the 4^4 lattice. Running QLUA code again but extracting the FT factor and the A_gamma factor to see if either of those is the problem. If they all match, then the issue is with the tensor contraction.
30792 ||| Running one last free field job with the 16^3 x 48 lattice to see if it works with the Python code. If it does, then we're done with testing and can revisit the clover configurations ||| 12/29
