from schwinger_hmc import *
import time

def robbins_monroe_polya(action, shape, a,
                         N_step, N_outer_therm, N_therm, N_batch_cfg, N_batch_skip,
                         tau, n_leap, rm_c, alpha=0.75):
    # S_range = action.S_range
    theta_i, delta = action.theta_i, action.delta
    print('Running Robbins-Monroe for theta_i,delta = {},{}'.format(theta_i, delta))
    # d_S = S_range[1] - S_range[0]
    # mid_S = np.mean(S_range)
    all_polyas = []
    all_as = []
    all_plaqs = []
    all_topos = []
    # cfg = action.draw_cfg(shape)
    # all_plaqs.append(np.real(np.mean(ensemble_plaqs(cfg))))
    # all_topos.append(np.sum(compute_topo(cfg)))
    # for i in range(N_outer_therm):
    #     cfg, _, _ = hmc_update(cfg, action, tau, n_leap, verbose=False)
    #     all_plaqs.append(np.real(np.mean(ensemble_plaqs(cfg))))
    #     all_topos.append(np.sum(compute_topo(cfg)))
        
    for n in range(N_step):
        action.a = a
        # FORNOW: reinit cfg
        cfg = action.draw_cfg(shape)
        for i in range(N_therm):
            cfg, S, acc = hmc_update(cfg, action, tau, n_leap, verbose=False)
            all_plaqs.append(np.real(np.mean(ensemble_plaqs(cfg))))
            all_topos.append(np.sum(compute_topo(cfg)))
        polyas = []
        tot_acc = 0.0
        for i in range(N_batch_cfg):
            print('rm iter n = {}, mc step i = {}'.format(n, i))
            for j in range(N_batch_skip):
                cfg, S, acc = hmc_update(cfg, action, tau, n_leap, verbose=False)
                tot_acc += acc / float(N_batch_cfg * N_batch_skip)
                all_plaqs.append(np.real(np.mean(ensemble_plaqs(cfg))))
                all_topos.append(np.sum(compute_topo(cfg)))
            np.array(all_plaqs).tofile('tmp.plaq.dat')
            np.array(all_topos).tofile('tmp.topo.dat')
            print('Wrote tmp topo / plaq')
            polyas.append(polya_uwtheta(cfg, action.polya_x))
        all_polyas.append(np.array(polyas))
        print('mean polya(a={:.8f}) = {}'.format(a, np.mean(polyas)))
        print('acc rate = {:.3f}'.format(tot_acc))
        delta_polya = np.mean(polyas) - theta_i
        da = delta_polya * rm_c / (delta**2 * (n+1)**alpha)
        a += da
        print('new a = {:.8f}'.format(a))
        all_as.append(a)
    return {
        'polyas': np.array(all_polyas),
        'as': np.array(all_as),
        'topos': np.array(all_topos),
        'plaqs': np.array(all_plaqs),
        'cfg': cfg
    }

if __name__ == "__main__":
    # Model
    parser = argparse.ArgumentParser()
    # binned physics
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--polya_x', type=int)
    parser.add_argument('--theta_i', type=float)
    parser.add_argument('--delta', type=float)
    # lattice
    parser.add_argument('--Lx', type=int, required=True)
    parser.add_argument('--Lt', type=int, required=True)
    # Robbins / HMC
    parser.add_argument('--N_outer_therm', type=int, default=500)
    parser.add_argument('--N_therm', type=int, default=10)
    parser.add_argument('--N_mc_iter', type=int, default=100)
    parser.add_argument('--N_rm_iter', type=int, default=1000)
    parser.add_argument('--N_mc_skip', type=int, default=2)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--n_leap', type=int, default=20)
    parser.add_argument('--seed', type=int, default=int(1000*time.time()))
    parser.add_argument('--rm_c', type=float, default=12.0)
    # etc
    parser.add_argument('--out_prefix', type=str, required=True)
    args = parser.parse_args()
    print('Running with args = {}'.format(args))
    start = time.time()
    L = (args.Lx, args.Lt)
    Nd = len(L)
    beta, polya_x, theta_i, delta = args.beta, args.polya_x, args.theta_i, args.delta
    shape = tuple([Nd] + list(L))
    # shape = (2,L,tE+2*tM)
    # Vp = Vm = tM * L
    # min_SM, max_SM = -2*beta*Vp, 2*beta*Vp
    # assert(args.min_S >= min_SM)
    # assert(args.max_S <= max_SM)
    action = PolyaGaussianBinnedPureGaugeAction(beta, polya_x, theta_i, delta)
    # action = SchwingerHBOnSKActionRestricted(beta, [None, None], tE, tM)

    # Robbins-Monroe on given bin
    # S_range = [args.min_S, args.max_S]
    # action.S_range = S_range
    result = robbins_monroe_polya(
        action, shape, 0.0, args.N_rm_iter, args.N_outer_therm, args.N_therm,
        args.N_mc_iter, args.N_mc_skip, args.tau, args.n_leap, args.rm_c)
    all_polyas = result['polyas']
    all_as = result['as']
    all_topos = result['topos']
    all_plaqs = result['plaqs']
    cfg = result['cfg']
    fname = '{}.uwtheta.dat'.format(args.out_prefix)
    np.array(all_polyas).tofile(fname)
    print('all_polyas.shape = {}'.format(all_polyas.shape))
    print('Wrote all uw thetas to {}'.format(fname))
    fname = '{}.a.dat'.format(args.out_prefix)
    np.array(all_as).tofile(fname)
    print('all_as.shape = {}'.format(all_as.shape))
    print('Wrote all as to {}'.format(fname))
    fname = '{}.topo.dat'.format(args.out_prefix)
    np.array(all_topos).tofile(fname)
    print('all_topos.shape = {}'.format(all_topos.shape))
    print('Wrote all topos to {}'.format(fname))
    fname = '{}.plaq.dat'.format(args.out_prefix)
    np.array(all_plaqs).tofile(fname)
    print('all_plaqs.shape = {}'.format(all_plaqs.shape))
    print('Wrote all plaqs to {}'.format(fname))
    fname = '{}.cfg.dat'.format(args.out_prefix)
    cfg.tofile(fname)
    print('Wrote final cfg to {}'.format(fname))
    
    print('Total runtime = {} s'.format(time.time() - start))
