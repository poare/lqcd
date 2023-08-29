# Notes about the comments from CalLat

### Renormalization
This comment is correct-- we used the wrong scheme for $\mathcal Z_q$, and our result uses $\mathcal Z_q^{({\gamma_\mu q^\mu})}$ instead of $\mathcal Z_q^{(\gamma^\mu)}$, i.e. we have the wrong projector for our quark-field renormalization. 

We had done this because you can only determine both $\mathcal Z_V$ and $\mathcal Z_q$ using the $\gamma_\mu q^\mu$ scheme, not the $\gamma^\mu$ scheme. The definition from our paper is:
$$
    \mathcal Z_q = \frac{i}{12\tilde p^2} \mathrm{Tr} [S^{-1}(p) \gamma^\mu \tilde{p}_\mu]\bigg|_{p^2 = \mu^2}.
$$
I believe this is consistent with the $\gamma^\mu q_\mu$-scheme definition of $\mathcal Z_q / \mathcal Z_V$, which is:
$$
    \frac{\mathcal Z_q^{\gamma^\mu q_\mu}}{\mathcal Z_V} = \frac{q^\mu}{12 q^2} \mathrm{Tr}[\gamma^\alpha q_\alpha \Pi_V^\mu ]
$$

Instead, what we should be doing is using $\mathcal Z_V$ pre-determined by the Ward identities, which can be found in https://arxiv.org/pdf/1411.7017.pdf. It looks like the correct results are on page 109:
$$
    \mathcal Z_V(24I) = 0.71273(26) \\
    \mathcal Z_V(32I) = 0.74404(181)
$$
which is consistent with what we found as well in our analysis. 

So, I should change the analysis code to:
- Compute $\mathcal Z_q^{(\gamma^\mu)}$ using the formula
    $$
    \frac{\mathcal Z_q^{(\gamma^\mu)}}{\mathcal Z_V} = \frac{1}{48} \mathrm{Tr}[\gamma^\mu\Pi_V^\mu]
    $$
  where $\Pi_V^\mu = S^{-1}(p_1) G_V^\mu(p_1, p_2) S^{-1}(p_2)$ is the amputated vector 3-point function ($\Lambda_V^\mu$ in our notation)
  - Note that now we don't need to use the $\mathcal Z_V$ factors until we've extrapolated everything to the chiral limit. We'll be computing:
    $$
    \frac{\mathcal Z_{nm}}{\mathcal Z_V^2} = \left(\frac{\mathcal Z_q}{\mathcal Z_V}\right)^2 [F_{nr}^{(\mathrm{tree})} F_{rm}^{-1}]
    $$
- Edit the code to only save $\mathcal Z_q / \mathcal Z_V$ and $\mathcal Z_{nm} / \mathcal Z_V^2$. **Also, edit the chiral extrapolation code** in `amell_extrap_final.py`, it should only extrapolate quantities that are normalized with factors of $\mathcal Z_V$.
- Be careful with correlations, as now $\mathcal Z_{nm}$ and $\mathcal Z_V$ should be uncorrelated since they're computed differently. 

# Changes to make to the paper
Most things are untouched, there are only a few places we need to clean up.

#### Responses to their direct comments
1. Added in a bit more about the pion mass specifics.
2. Not really sure if we need to address this? 
3. Added in some extra lattice spacing details.
4. Might want to address this by talking about the fact that there is still a discrepancy between our results and theirs, so clearly there are some systematics that aren't being accounted for and you can only make an apples-to-apples comparison between our results.

##### Renormalization section
1. Equation 22 with the definition of $\mathcal Z_q$ should become:
    $$
    \frac{\mathcal Z_q^{(\gamma^\mu)}}{\mathcal Z_V} = \frac{1}{48} \mathrm{Tr}[\gamma^\mu\Lambda_V^\mu]
    $$
   and we need to define $\Lambda_V$ in the main text.
2. Discussion between Eqs 22 and 23 needs to be updated because we are no longer including $\mathcal Z_V$ or $\mathcal Z_A$ in the fit, and we can update the notation.
3. Eq (24) should be updated with extra $\mathcal Z_V$ factors
    $$
    \frac{\mathcal Z_{nm}}{\mathcal Z_V^2} = \left(\frac{\mathcal Z_q}{\mathcal Z_V}\right)^2 [F_{nr}^{(\mathrm{tree})} F_{rm}^{-1}]
    $$
4. Fix the sign on the anomalous dimension.
5. Renormalization coefficient results.

##### Appendix C
1. We still have a valid determination of $\mathcal Z_V$ and $\mathcal Z_A$, but we need to phrase it a bit differently since working in the $(\gamma^\mu, \gamma^\mu)$ scheme only determines $\mathcal Z_q / \mathcal Z_V$. We can likely pull more closely from the original RI/sMOM paper, 
2. (independent of CalLat comments) Add in band for the $\mathcal Z_V$ values from https://arxiv.org/abs/1411.7017, which are:
    $$
        \mathcal Z_V(24I) = 0.71273(26) \\
        \mathcal Z_V(32I) = 0.74404(181)
    $$
3. 