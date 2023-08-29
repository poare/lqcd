# Hybrid Monte Carlo (HMC)
We want to simulate a chiral gauge theory which circumvents the chiral fermion problem. To start, we'll study a theory of chiral bosons, summarized in the next section. First, let's review the setup for HMC, as we'll be hoping to simulate field configurations with HMC. We wish to perform the path integral,
$$
    \langle\mathcal O\rangle = \frac{1}{Z} \int D\phi\, e^{iS[\phi]}\mathcal O(\phi)
$$
Note that for our case, the action $S_k[\phi]$ is invariant under Wick rotation (it goes as $\int dt ... \partial_t ...$), but it comes with an extra factor of $i$ in the front, hence we can write it as:
$$
    \langle\mathcal O\rangle_k = \frac{1}{Z_k} \int D\phi\, e^{-S_{k}^{(R)}[\phi]}\mathcal O(\phi)
$$
where we've decomposed the action as $S_k[\phi] = i S_k^{(R)}[\phi]$. Note that I have no reason to suspect that $S_k^{(R)}[\phi]$ is positive-semidefinite and forms a valid probability distribution; this may be a problem. There also may be a chance it's positive for left-chiral particles and negative for right-chiral particles, since $d\lfloor d\phi\rceil\propto \partial_x \partial_t \phi$. Regardless, let's assume we have a valid probability distribution $D\phi e^{-S[\phi]}$, and we want to use Monte Carlo to evaluate the path integral. 

To generate a field configuration, we'll need to follow 

# Chiral field theory
The theory we want to simulate in $(1+1)d$ has the action,
$$
    S_k = \frac{ik}{2\pi} \int_{\mathcal{N}^3} d\phi\,\cup d\lfloor d\phi\rceil = 2\pi i k \int_{\mathcal{M}^2} \phi\, d\lfloor d\phi\rceil
$$
where $\mathcal M^2$ is the $(1+1)d$ spacetime manifold and $\mathcal{N}^3$ is its bulk, i.e. $\partial\mathcal N^3 = \mathcal M^2$. Here the field $\phi$ is valued in $\mathfrak{u}(1)\cong \mathbb R$ and has gauge redundancies so that it is effectively valued in $\mathbb R / \mathbb Z$. The $\lfloor\cdot\rceil$ is a map $\mathbb R\rightarrow\mathbb R / \mathbb Z$ which floors $\cdot$ to the nearest integer. $\phi$ represents the phase of the $U(1)$ field, i.e. $\Phi = e^{2\pi i\phi}$ is our dynamical boson field. 

#### Continuum model
The continuum model can be recase relatively simply as a bosonic $\phi$ coupled to vortices. 