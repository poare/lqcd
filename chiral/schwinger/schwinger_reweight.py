### Run reweighting with forces for schwinger.

assert(False) # Not done!
def main(L, n_step, n_skip, n_therm, beta, kappa, eps=0.1):
    Nd = len(L)
    V = np.prod(L)
    shape = tuple([Nd] + list(L))
    init_cfg_A = 0.4*np.random.normal(size=shape)
    cfg = np.exp(1j * init_cfg_A)

    np.set_printoptions(precision=4, linewidth=1000)

    # MC updates
    total_acc = 0
    cfgs = []
    plaqs = []
    for i in range(-n_therm, n_step):
        print("MC step {} / {}".format(i+1, n_step))
        old_cfg = np.copy(cfg)
        M = dirac_op(cfg, kappa, sign=1)
        Mdag = dirac_op(cfg, kappa, sign=-1)
        ## Direct det method
        # detMdagM = np.real(det_sparse(Mdag * M))
        ## Pseudofermion heatbath
        phi = sample_pf(Mdag)
        old_S = -beta * np.sum(np.real(ensemble_plaqs(cfg))) + pf_action(M, Mdag, phi)
        old_pi = np.random.normal(size=cfg.shape)
        old_K = np.sum(old_pi*old_pi) / 2
        old_H = old_S + old_K

        cfg = np.copy(cfg)
        new_pi = np.copy(old_pi)
        leapfrog_update(cfg, new_pi, phi, beta, kappa, tau=0.5, n_step=15)

        M = dirac_op(cfg, kappa, sign=1)
        Mdag = dirac_op(cfg, kappa, sign=-1)
        ## Direct det method
        # detMdagM = np.real(det_sparse(Mdag * M))
        new_S = -beta * np.sum(np.real(ensemble_plaqs(cfg))) + pf_action(M, Mdag, phi)
        new_K = np.sum(new_pi*new_pi) / 2
        new_H = new_S + new_K

        delta_H = new_H - old_H
        print("Delta H = {:.5g} - {:.5g} = {:.5g}".format(new_H, old_H, delta_H))
        print("Delta S = {:.5g} - {:.5g} = {:.5g}".format(new_S, old_S, new_S - old_S))
        print("Delta K = {:.5g} - {:.5g} = {:.5g}".format(new_K, old_K, new_K - old_K))
        
        acc = 0
        if np.random.random() < np.exp(-delta_H):
            acc = 1
        else:
            cfg = old_cfg
        print("Acc {:.5g} (changed {})".format(min(1.0, np.exp(-delta_H)), acc))
        total_acc += acc
        
        # Eval avg plaq
        plaq = np.sum(np.real(ensemble_plaqs(cfg))) / V
        print("Average plaq = {:.6g}".format(plaq))
        
        # Save cfg
        if i >= 0 and i % n_skip == 0:
            print("Saving cfg!")
            cfgs.append(cfg)
            plaqs.append(plaq)
            
    print("MC finished.")
    print("Total acc {:.1f}".format(total_acc / n_step))
    return cfgs, plaqs
