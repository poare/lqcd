JOBNUM ||| PURPOSE ||| DATE

PION MASS JOBS:
 ||| Testing pion 2 point on new smeared / gauge fixed configurations at /data/d10b/ensembles/isoClover/cl3_16_48_b6p1_m0p2450_smeared_gf/landau/ ||| 7/17
 ||| Testing pion 2 point on David's smeared / gauge fixed configurations at /data/d10b/users/poare/gf//data/d10b/users/poare/gf/cl3_16_48_b6p1_m0p2450_smeared_gf/landau/ ||| 7/17

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

Storing all the npr_momfrac output from -6 to 6 in 22454 (/data/d10b/users/poare/lqcd/npr_momfrac/output/cl3_16_48_b6p1_m0p2450_22454).
Useful data sets:
- 20401 has quark renormalization and (vector) operator renormalization for momenta from 0 to 5
- 22454 has quark renormalization and (vector) operator renormalization for momenta from -6 to 6
- 22812 has quark renormalization and (irrep) operator renormalization for momenta from 0 to 5
- 23014 has vector and axial currents for momenta from 0 to 5
- 23476 has vector and axial currents run at nonzero q insertion (CODE IS BAD).
- 23660 should have ZA
