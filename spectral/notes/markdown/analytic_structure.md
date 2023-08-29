# Analytic structure of spectral functions
- What constraints on the reconstruction problem can we impose with knowledge of how well-behaved the spectral function is? 

#### Notation
- Suppose we evaluate the correlation function $C(t)$ at $N$ data points, $\{C(t_n)\}_{n = 1}^N$. The reconstruction problem is to take the Fourier transform $\{\tilde C(\theta_n)\} = \mathcal F[\{C(t_n)\}]$, and to analytically continue $\tilde C(\theta_n)$ to the complex plane, $\tilde C(\theta)$. The spectral function is just $\rho(\omega) = \tilde C(i\omega)$, where $\omega\in\mathbb R$. 
- Assume we're working with fixed data points $t_n$ and $\omega_n$ (although the reconstruction likely differs depending on what the actual values of $t_n$ and $\omega_n$ are). Let:
  1. $\mathcal C_N$ denote the space of all possible sets of data points $C(t_n)$, i.e. $\mathcal C_N$ is isomorphic to $\mathbb C^N$ as a vector space and any set of data points $\{C(t_n)\}_{n = 1}^N\in\mathcal C^N$ . 
  2. $\tilde{\mathcal{C}}_N$ be the space of all possible data points in frequency space, so $\{\tilde C(\theta_n)\}_{n = 1}^N\in \tilde{\mathcal{C}}_N$. 
  3. $C^\infty(\mathbb C)$ be the space of smooth $\mathbb C$-valued functions. 
- Formally, the inverse problem is a map:
    $$
        \mathcal L^{-1}_N : \mathcal C_N\rightarrow C^\infty(\mathbb C)
    $$
  Likewise, analytic continuation is denoted as the map:
    $$
        A_N : \tilde{\mathcal{C}}_N\rightarrow C^\infty(\mathbb C)
    $$
  hence $\mathcal L^{-1}_N = A_N\circ \mathcal F$. 


#### Analytic constraints
- **Carlson's theorem** (https://en.wikipedia.org/wiki/Carlson%27s_theorem): Given certain constraints on a function (assuming it's sufficiently well-behaved), if we know the function's value on the natural numbers $\mathbb Z$, there's a unique analytic continuation to the entire complex plane. 
  - Assuming the function is nicely behaved seems like a reasonable thing to be doing for spectral functions, since we know they satisfy a lot of nice analytic properties. 
  - **Question to think about**: given $N$ data points we've sampled a function at (i.e. we know $\{\tilde C(\omega_n)\}_{n = 1}^N$, where $\tilde C$ is the Fourier transform of the correlator), how different can different analytic continuations of these points be, especially given analytic information that we want the final continuation $\tilde C(\omega)$ to satisfy? 
  - Basically: given extra assumptions about the initial function $C(\omega)$, how ill-posed is the reconstruction problem? 
    - A measure of how ill-posed the problem is is how large the kernel of $\mathcal L_N^{-1}$ is: the larger $\mathrm{ker}\mathcal L_N^{-1}$ is, the more solutions we can reconstruct from the same set of data. 
- An extension of this idea is **Rubel's theorem**. 
  - The **upper density** of a subset $A\subset\mathbb N$ is defined as 
    $$
    \overline D(A) := \limsup_{n\in\mathbb N} \frac{|A\cap \{0, 1, ..., n\}|}{n}
    $$
    Essentially, one can think of the upper density of a subset $A$ as the fraction of the natural numbers that $A$ contains as we look at larger and larger slices of $\mathbb N$. $A$ has upper density 1 if as we increase $n$, $A$ contains more and more natural numbers. 
  - https://math.stackexchange.com/questions/126745/references-on-density-of-subsets-of-mathbbn
  - The examples on Wikipedia are helpful for understanding upper density a bit better: https://en.wikipedia.org/wiki/Natural_density
  - **Theorem (Rubel)**: If $f$ is specified on a subset $A\subseteq\mathbb N$ of upper density 1, then $f$ has a unique analytic continuation.
- Check out this link: https://math.stackexchange.com/questions/3291778/when-analytic-continuation-from-discrete-set-is-unique
- 