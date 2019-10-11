Last time: the run broke when attempting to evaluate configuration 1620, right after evaluating 290 and 2460. Try again and see if it breaks on the same one, and also edit the code to make it save a number after each run.

The run just failed again at configuration 1620, so I'm going to delete that configuration and see if that file specifically is causing the issue.

Questions 10/7:
 - What is the twist of an operator, and why is it important?
    - Twist is the mass dimension minus the spin dimension. In the OPE, you can Taylor expand an operator a fixed distance away in powers of the distance and the twist, so lowest twist gives largest contribution.
 - How difficult is it to compute 3 point functions compared to 2 point? Are there any easy computations I should do (like the pion mass) for 3-pt functions to get the computation under my belt?
    - Yes, compute a 3-pt function. Determine different ways to do this-- through the source and through the sink, which are discussed in some of the theses that Phiala sent over.
 - Worth it to consider learning chroma?
    - At some point I'll probably have to be able to parse chroma code to write my own stuff in QLUA, but not necessary to do now.


Group meeting 10/10 on QLUA:
- type($\cdot$) will give you the type of a variable.
- Every variable is nil before you assign it to something.
- Namespaces in QLUA can be substituted for tables; qcd is a table, not a library
- To see the keys and values:
for i, d in pairs(qcd) do
  ...
end
- Tables in LUA have different parts: integer indexed and key indexed. ipairs(...) gets the integer indexed parts
- To create a table: x = {0, 10, 20, a = 15} has x[[0]] = 0, x[[1]] = 10, ...
  - pairs(x) will return all the keys / values pairs, and ipairs will return only the integer keys
  - Note x[var] = 50 adds the key / value pair (var, 50) to x. Double [[]] is used to get back values.
- To find the source code:
  cd /home/agrebe/wombat/qlua-quda/src/qlua/sources
  - Might be useful for figuring out what the signature of different funtions is

To run on wombat and write to the log while running, use this command:
/opt/software/openmpi-2.1.1/bin/mpirun -n 6 /opt/software/qlua-20170804/qlua/bin/qlua /home/poare/lqcd/pion_mass/pion_mass.qlua > logs/pion_mass_log${PBS_JOBID}.txt

