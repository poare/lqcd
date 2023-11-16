# Some quick notes about the numerical details and tests to do

Parameter notes (so far, all $N_c = 2$):
- $8\times 8$: 
    - `DEFAULT_STEP_SIZE = 0.005` and `DEFAULT_RHMC_ITERS = 25`: For 1000 configurations, ran with an acceptance rate of about 0.6. Each config takes $\approx 10$ seconds to run on my desktop.
- $16\times 16$:
    - `DEFAULT_STEP_SIZE = 0.001` and `DEFAULT_RHMC_ITERS = 25`: For 100 configurations, ran with acceptance rate of 0.84, and the plaquette is clearly not thermalized. Each configuration took about $\approx 50$ seconds to run on the desktop.
    - TODO try `DEFAULT_STEP_SIZE = 0.005` and `DEFAULT_RHMC_ITERS = 25`

TODO / tests to run:
- Change up some things about how we're storing the data. Store each configuration individually in an HDF5 file to make it easier to interrupt the code when it's running (so we can keep going from cfg 300 if we need to interrupt it)
- Speed up the code
- $8\times 8$ lattice, compare 5 different hot starts and a cold start. See what the plaquette thermalizes to and make sure it looks statistically the same. Plot all streams against one another to see.
- Run some experiments to determine optimal step sizes for different lattice geometries
- Try to run $SU(3)$