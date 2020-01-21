- 12/30: Job ID 4056, run failed because of some issues with the bash script. 
- 1/9: Job fail because of time limit. Determine how to run on the longer queues.
- 1/21: Everything up and running, code is reformatted to have parameters be input in the bash script. Code still buggy, going to try to debug with the pure gauge quenched QCD.
  - Job 5623: Changed parameters on solver to make it identical to Phiala's code. UPDATE: Job failed because of error.
  - Job 5683: Edited code to make sure that the computation code Phiala has works with my framework. If this works, then there's gotta be a bug somewhere in the correlation function 
    computation. If it doesn't work, then we might have bigger issues with the framework.
    - If this doesn't work, then maybe it's the clover solver. Swap out solver with the one that Phiala's code uses in mosaic.
