# RI/MOM for the Quark EMT on Exponential Clover Ensembles

Rough pipeline:
1. Smear / gauge fix all configurations per ensemble. In 2021 I used GLU, not sure what is the state of the art now. 
2. Compute propagators, preferably without needing to write them all out to disk.
3. Perform contractions and save correlators out. 


Notes
- My QLUA code 