# General considerations
- Inverse Laplace transform is not well defined (i.e. have a non-trivial kernel which is quite large), but that's for an arbitrary Laplace transform. In reality, there's some structure there that we also want to impose
- **Carlson's theorem**: Given certain constraints on a function (assuming it's sufficiently well-behaved), if we know the function on a countably infinite number of points, there's a unique analytic continuation to the entire complex plane. 
  - Assuming the function is nicely behaved seems like a reasonable thing to be doing for spectral functions, since we know they satisfy a lot of nice analytic properties. 
  - **Question to think about**: given $N$ data points we've sampled a function at (i.e. we know $\{\tilde C(\omega_n)\}_{n = 1}^N$, where $\tilde C$ is the Fourier transform of the correlator), how different can different analytic continuations of these points be, especially given analytic information that we want the final continuation $\tilde C(\omega)$ to satisfy? 
  - Basically: given extra assumptions about the initial function $C(\omega)$, how ill-posed is the reconstruction problem? 
    - Define the inverse map to be the map $\mathcal L^{-1}_N : \{\{\tilde C(\omega_n)\}_{n=1}^N\}\rightarrow \{C(\omega)\}$. A measure of how ill-posed the problem is is how large the kernel of $\mathcal L_N^{-1}$ is: the larger $\mathrm{ker}\mathcal L_N^{-1}$ is, the more solutions we can reconstruct from the same set of data. 
- Different norms for minimization: are there physics informed quantities that we can add to the norms?
  - Note that for the Nevanlinna paper, this shows up in the functional $F[\rho(\omega)]$ that we minimize at the end. Currently, it's just an expansion in Hardy space with a smoothness prior, but there are a lot of other things we can try. 
  - Sum rules?
    - Weinberg sum rules are a thing that we can impose, i.e. the normalization
      $$
      \int d\omega\,\rho(\omega) = 1
      $$
  - Dispersion relations?
    - https://www2.ph.ed.ac.uk/~rzwicky2/disp-lec.pdf
- Things to try with nevanlinna:
  - Extended precision
  - Prior to minimize on spectral functions which picks out the best, subject to physical constraints like those discussed above. 
  - Implementing derivatives to do the optimization at the end of the day.
    - Will has been thinking of using a spline, I feel like doing some sort of reverse-mode AD could be helpful.

## General notes
- Rewrite code in C++, there seems to be some issues with extended precision numbers in Python
- For Monte Carlo data: is there a way to partition each errorbar into N segments, then compute the Pick matrix of each set of data points and see which ones have a solution that is allowed? Presumably could test this with data that has a good Pick matrix, then adding some noise, then partitioning up the errorbar and pushing it through
- Next: H5Writer and read back into Python, plot recon
- Write up notes on optimizing the Hardy function basis and implement


## Simulation file format
- Input and output stored in same folder `$dir` in h5 files. 
  - Input file is `${dir}/data.h5`, output file is `${dir}/recon.h5`
- For extended precision data, store in h5 files as strings with separate datasets for real and imaginary parts. 
- Input file datasets:
  - 'beta': Temporal extent of data.
  - 'freqs/imag': Imaginary part of Matsubara frequencies (should be purely imaginary)
  - 'ng/real': Real part of Green's function.
  - 'ng/imag': Imaginary part of Green's function.
- Output file datasets: all input file datasets, and in addition:
  - 'start' : Start of the real domain data (int)
  - 'stop' : End of the real domain data (int)
  - 'npts' : Number of points for real domain data (int)
  - 'beta' : Temporal size of lattice (int)
  - 'eta' : eta value (string, i.e. extended precision number)
  - 'recon_real' : Reconstructed (real part of) spectral function.
  - 'recon_imag' : Reconstructed (imaginary part of) spectral function.
  <!-- - 'phis' : phi values used for reconstruction. -->