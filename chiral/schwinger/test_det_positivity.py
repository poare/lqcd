from schwinger import *

def dirac_op_screen_mu(cfg, kappa, sign=1, mu=0):
    assert(NS == 2)
    # print("Making dirac op...")
    start = time.time()
    Nd, L = cfg.shape[0], cfg.shape[1:]
    assert(Nd == 2 and len(L) == 2)
    V = np.prod(L)

    # preprocess cfg
    cfg[1,1:,0] = 0 # cut t-links at t = 0 (except x = 0)
    cfg[1,:,0] *= np.exp(1j * mu) # chemical potential (t = 0)
    
    cfg0, cfg1 = cfg[0], cfg[1]

    indptr = []
    indices = []
    data = []
    for i in range(V):
        x = index_to_coord(i, L)
        fwd = list(x)
        fwd[0] = (fwd[0] + 1) % L[0]
        fwd_sign = -1 if fwd[0] == 0 else 1
        bwd = list(x)
        bwd[0] = (bwd[0] - 1) % L[0]
        bwd_sign = -1 if bwd[0] == L[0]-1 else 1
        up = list(x)
        up[1] = (up[1] + 1) % L[1]
        up_sign = -1 if up[1] == 0 else 1
        down = list(x)
        down[1] = (down[1] - 1) % L[1]
        down_sign = -1 if down[1] == L[1]-1 else 1
        link_fwd = fwd_sign*cfg0[x]
        link_bwd = bwd_sign*np.conj(cfg0[tuple(bwd)])
        link_up = up_sign*cfg1[x]
        link_down = down_sign*np.conj(cfg1[tuple(down)])
        j_fwd = get_coord_index(fwd, L)
        j_bwd = get_coord_index(bwd ,L)
        j_up = get_coord_index(up, L)
        j_down = get_coord_index(down, L)

        j_blocks = [(i, pauli(0)),
                    (j_fwd, -kappa * link_fwd * (pauli(0) - sign*pauli(1))),
                    (j_bwd, -kappa * link_bwd * (pauli(0) + sign*pauli(1))),
                    (j_up, -kappa * link_up * (pauli(0) - sign*pauli(2))),
                    (j_down, -kappa * link_down * (pauli(0) + sign*pauli(2)))]
        j_blocks.sort(key=lambda x: x[0])
        indptr.append(len(indices))
        for j,block in j_blocks:
            indices.append(j)
            data.append(block)
    indptr.append(len(indices))
    data = np.array(data, dtype=np.complex128)
    indptr = np.array(indptr)
    indices = np.array(indices)
    out = sp.sparse.bsr_matrix((data, indices, indptr), shape=(NS*V,NS*V))
    
    # print("TIME dirac op {:.2f}s".format(time.time() - start))
    rescale = 1/(2*kappa)
    return rescale*out

def ...
