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
  - Read chapters from Differential Forms and Algebraic Topology: Learn de Rham Theory and Chern Simons theory (universality classes)
- Fiber bundles: Read the lecture notes on Mendeley
- Homology: Type up 18.905 lecture notes, read Hatcher
- Cohomology: TODO
- Representation theory:
  - Weights and roots, representations of $sl(3; \mathbb C)$
  - Young tableaus and Dynkin diagrams

11/18 NUPAT Seminar on the sign problem in lattice field theories:
- Sign problem arises when Euclidean action is complex, because $e^{-S[\phi]} / \int D\phi e^{-S[\phi]}$ is no longer a valid probability distribution
  - Arise in theories like finite density QCD and nuclear EFTs on the lattice
  - Any real time path integral, since weighting is $e^{iS}$ and not $e^{-S}$
- Reweighting: Split the action into real and imaginary parts, then sample with respect to the probability defined by the real part of the action.
  - Issue: Need to calculate $\langle e^{-i Im[S]}\rangle_{Re[S]}$ in the denominator of $\langle O\rangle$, and this is very noisy and blows up the calculation.
- This talk: Like Cauchy's theorem in complex analysis, we can deform the domain of integration of the path integral to an arbitrary manifold. Then we can pick a nicer manifold to integrate over which suppresses the noise from the sign problem.

Questions 11/18:
- In the parton model, the Bjorken variable $x$ is the same as the momentum fraction $\xi$. When we consider full QCD, the parton model breaks down to some extent because of interactions between the quark and gluon constituents of the proton. In this case, to what extent does $x$ differ from $\xi$, and is it still reasonable to call $x$ the momentum fraction of the proton? What is the difference between the momentum fraction and the Bjorken variable, specifically when we do not assume the parton model? That seems to be how a lot of sources do it.
- Do I understand DGLAP? Last time we talked you said that "the quantum numbers of gluons and quarks are the same, so the PDFs can mix". How does this work with the DGLAP derivation, where I go through and calculate the possible partonic influences on my hadron?
  - Do you need to do this calculation for a general process each time you do a calculation to get the right splitting functions to describe the flow of the PDFs?
- Walk through the ratio stuff and figure out what exactly is going on-- specifically with the projection matrices and different correlations.
- For a general calculation, where do you usually start? At the ratio and whittle it down by choosing the right $\mathcal J$ and projection matrices to get what you want?
  - Where do the projection matrices come in?

11/18 next steps:
- Debug code once the server is up and running
- Calculate a pion three point function for the electromagnetic form factor $F_1(p^2)$. Will need to:
  - Read the textbook about what exactly to calculate. Work through the calculations myself for an arbitrary value of $p$ to get $F_1(p^2)$
  - Calculate the three point function and two point function, and form the ratio to extract the form factor
    - Do this both through the sink and through the source to make sure I understand what I'm doing.
- Read the section on GPDs and determine how to calculate the moments of a GPD. Read the theses to supplement this as well.
- Read 32.4 and 32.5 on the OPE and lightcone coordinates.

1/6 questions:
- Figure out how bash works. When I run "bash", the whole command line lights up green. Not sure why it doesn't do this by default anymore.

Group meeting 1/9: Task management, Taxi code.

Meeting with Phiala 1/13: Good start for debugging pion code.
- The folder /data/d10a/projects/playingwithpions should have relevant code for computing a pion effective mass curve
  - test_pion.qlua should be the QLUA file that computes the correlator
  - submit_pion.sh is the bash script. It's written much more succinctly than mine, and loops through the configurations and file names before calling the QLUA script.
  - Check out Mike's mosaic repository: /data/wombat/users/mlwagman/mosaic. This should have a lot of fermion initialization code for QLUA.
- File naming conventions: Starting with 1 and going to whatever it ends up at. Will either be sequential or every 10, depending on its output in an HMC stream when the gauge field configurations are being generated.

Meeting with Phiala 1/23: Pion code debugged!
- Server possibly going down tomorrow, so starting 1/24 may need to figure out smaller things to run locally. Phiala suggested:
  - Figure out where the issue was with
- Big next step: Learn renormalization on the lattice. Read the papers that I've printed out, including Rome-Southampton and Phiala's computation of gluon gravitational form factors. Then, can begin to work on operator renormalization, specifically on the off diagonal elements of the mixing matrix:
$$
  \begin{pmatrix}\mathcal O_{R, q} \\ \mathcal O_{R, g} \end{pmatrix} = \begin{pmatrix} Z_{qq} & Z_{qg} \\ Z_{gq} & Z_{gg}\end{pmatrix}\begin{pmatrix} \mathcal O_{latt, q} \\
  \mathcal O_{\latt, g}\end{pmatrix}
$$
- If I can before server shuts down, try to figure out how to run with momentum and generate a dispersion curve.

Meeting with Phiala 2/3:
- Try to run the pion dispersion on code in the gluon form factors paper. Don't run on all configurations, just need around 20ish (maybe do 50ish?). These are the ensembles at /data/d10b/ensembles/isoClover/cl21_32_96_b6p1_m0p2800_m0p2450_mn3-ec/cfgs/0. Should look like Figure 2a in the paper. Do the lime files with ec-0 in the name, there are 48 of them.
  - Definitely something weird going on, but might just be artifacts of a smaller lattice size-- Phiala says it looks decent
  - Will need to restore the units, there's a value of a (lattice spacing) in the paper to use.
- After that start thinking about how to do renormalization of the quark operators
  - Start looking into gluon gauge-fixed propagators. See how to calculate them, maybe try to calculate a few by myself, and see what they're supposed to look like.

Meeting with Phiala 2/10:
1. Pion dispersion code:
  - Go over scaling and units-- right now the effective mass isn't even close to what it is in the paper.
2. Representations of H(4) and the OPE
  - Walk through what I think I understand:
    - 4 of Lorentz corresponds to $\tau_1^{(4)}$ of H(4), and we want the Clebsch-Gordan decomposition of this representation.
  - Do we always consider symmetric representations? (i.e. for 3 index tensors?)
3. Rome-Southampton:
  - Is Z_{qg} << Z_{qq} just like in the gluon mixing case?
  - Can we walk through an example? The paper is very abstract and it would be nice to have a more concrete idea of what all of the letters in those master equations mean (also look at Phiala's gluon ff and NPR papers, and the other paper with the overview)?
- NPR code notes:
  - Edit out the operator and put the finite difference quark operator in-- should be reasonably easy to do that. The code should basically run once that's done.
  - Try to get that working by the end of the week, and coordinate with David.
  - The Born term is just the tree level matrix element. It should be in the Gockeler paper, along with the R(\mu) term to convert from the RI-MOM scheme to the MS bar scheme.
    - That term is in the Gracey paper to 3 loops, so if the fitting isn't working very well then use the more precise version.
    - This $R(\mu)$ term is the "matching coefficient" to convert between schemes to get it into MS bar.
  - Phiala should be sending over the staples paper. The original code implements that, so it should make it easy to see what each part of the code is doing.
  - This is pretty much the simplest NPR calculation we can do, so it should help to get me started before considering mixing and uglier things in the proton calculation.
  - The figure of the curvy plot in Figure 1 should turn flat once you multiply by the R(\mu) matching coefficients. 
