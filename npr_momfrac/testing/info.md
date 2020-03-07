This directory is for local testing with QLUA, and running very small configurations.
- Job 14432: Did a gauge transformation on the field and saw what the new propagator was. Doesn't work-- the two propagators are different.
- Job 14470: Computed again with a new source. Also changed plist so there is only one momentum.
- Job 14474: Changed way we're computing an element of SU(n).
- Upon running the analysis, the output looks good. The diagonal propagator elements tend to be around order 10^4 and very similar to one another. The propagators differ in their $\Theta(1)$ terms which are off diagonal, but when we divide through by a common normalization like the trace these differences appear to be uncertainties associated with inverting the equation. The traces on all the color and Dirac parts appear to be the same (within precision for this calculation), and job 14487 is computing the propagator twice from the same gauge field to see if I get those uncertainties.
- Next up: Have to figure out the propagator stuff. 
- Job 14502: Added in group.togroup on the gauge field, see if that helps anything.
- Job 14503: Added in extra group projection in gauge transformation \Omega(n) to see if that helps.
- Job 15890: Propagators match with David's code. Running this job to compare Green's functions with David.
- Job 16636: Running through the operator with a whole lot of momenta. Job 16637 is also running with momenta from 1000 to 1500
