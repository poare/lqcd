Last time: the run broke when attempting to evaluate configuration 1620, right after evaluating 290 and 2460. Try again and see if it breaks on the same one, and also edit the code to make it save a number after each run.

The run just failed again at configuration 1620, so I'm going to delete that configuration and see if that file specifically is causing the issue.

Questions 10/7:
 - What is the twist of an operator, and why is it important?
    - Twist is the mass dimension minus the spin dimension. In the OPE, you can Taylor expand an operator a fixed distance away in powers of the distance and the twist, so lowest twist gives largest contribution.
 - How difficult is it to compute 3 point functions compared to 2 point? Are there any easy computations I should do (like the pion mass) for 3-pt functions to get the computation under my belt?
    - Yes, compute a 3-pt function. Determine different ways to do this-- through the source and through the sink, which are discussed in some of the theses that Phiala sent over.
 - Worth it to consider learning chroma?
    - At some point I'll probably have to be able to parse chroma code to write my own stuff in QLUA, but not necessary to do now.


Group meeting 10/10 and 10/17 on QLUA:
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
  - Might be useful for figuring out what the signature of different functions is
- L:Real(3) creates a real valued lattice field with the value 3. Similarly, can do L:Int(), L:Complex(), L:ColorVector(Nc), L:ColorMatrix(Nc), L:DiracFermion(Nc), and L:DiracPropagator(Nc)
- Array indexing is like M[{x, y, z, t}]
- #L will return the number of dimensions of the object, L[0] will give you the size of that dimension
- L:ColorMatrix[{a = 3}] will access a L:ColorVector. You can also write L:ColorMatrix[{x, y, z, t, a = 1, b = 2}] to index one element of the color matrix. Similarly L:DiracFermion[{x, y, z, t, c = 1, d = 2}] should give a number because it indexes a Dirac Matrix.
- To create a point source to invert to get a propagator, create prop = L:DiracPropagator(). Create the source at src_pos = {4, 0, 6, 12}. Then write the following:
for ic = 0, 3 do
  for is = 0, 4 do
    src_pos.c = c    -- adds in key value c with the number c
    src_pos.d = d    -- adds in key value d for spinor index
    prop[src_pos] = complex(1, 0)
  end
end
- Calling field:shift($\mu$, "from_forward") rolls the field forward in the $\mu$ direction

To run on wombat and write to the log while running, use this command:
/opt/software/openmpi-2.1.1/bin/mpirun -n 6 /opt/software/qlua-20170804/qlua/bin/qlua /home/poare/lqcd/pion_mass/pion_mass.qlua > logs/pion_mass_log${PBS_JOBID}.txt

Notes 10/15:
- Does anyone have pion mass code that I can compare this to so I can see if there's a bug in my code? Is it worth trying to run my code again to play around on a different data set?
- Is the only place that $m_u$ and $m_d$ come in the value of $\kappa$?
- Write up exactly what I'm calculating, and send it to Phiala. Rerun the code as well.
- I'm going to run two more configurations today to see if there's anything I can do to the solver to fix the errors with the pion mass. These are:
  - Job 230701. Using two clover solvers with solveU = CL_u:solver(1e-10, 1000, 1e-25, 2000)
  - Job 230706. Using two clover solvers with the single inverter, solveU = CL_u:solver(1e-22, 10000)
- Tried to fix and hermitian conjugate the propagator instead of just conjugate, output stored in 5799.

Notes 10/21:
- Statistical analysis seems to be correct, I checked with Anthony's program. At this point I'm waiting on the server to come back up to keep running tests, and will probably see if it's gauge covariant soon.
- Next up: Time to actually parse the 3-pt function stuff.
- How should the error be scaling when I change the bootstrap number? Mine is relatively constant right now.
- Renormalization on the lattice: Quarks and gluon operators have the same symmetry, so can mix into each other under renormalization. That's why in a lot of papers there's a mixing matrix with the PDFs:
$$
  \begin{pmatrix} f_q(x) \\ f_g(x)\end{pmatrix} =
  \begin{pmatrix} Z_{qq} & Z_{qg} \\ Z_{gq} & Z_{gg} \end{pmatrix}
  \begin{pmatrix} f_q^0 \\ f_g^0\end{pmatrix}
$$
  - I should try to learn about how renormalization works with things like this, it seems very important.
- Symmetry breaking to hypercubic group $H(4)$. When we break Lorentz symmetry by putting a lattice in, we break the Lorentz symmetry down to the hypercubic group. When we tensor irreps of the Lorentz group together (for example, the operator $\mathcal O_{\mu_1\mu_2\mu_3\mu_4}$ lives in a representation of the Lorentz group tensored together 4 times) we don't mix subspaces of "different powers". In the hypercubic group, this isn't necessarily true and we instead get mixing from different powers, which is an issue
  - Breaking down to $H(4)$ means that there are less symmetry elements, so more operators have the same symmetries. This implies that more operators on the lattice are allowed to mix in the continuum under renormalization.
  - Can only calculate moments of GPDs up to order 4 because of this

Math subjects to learn:
- Differential topology: Read Tu's textbook, supplement with Lee smooth manifold
  - Differential forms and integration
- Fiber bundles: Read the lecture notes on Mendeley
- Homology: Type up 18.905 lecture notes, read Hatcher
- Cohomology: TODO
- Representation theory:
  - Weights and roots, representations of $sl(3; \mathbb C)$
  - Young tableaus and Dynkin diagrams
