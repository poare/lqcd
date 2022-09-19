"""
doc here
"""
import scipy
import numpy as np
import gvar as gv
import lsqfit
import itertools
import pandas as pd
# from lqcd_analysis import visualize as plt
import visualize as plt


def check_duplicate_keys(dict_list):
    """
    Checks for duplicate keys in a list of dicts.
    Raises:
        ValueError, when non-unique keys are found
    """
    all_keys = [list(adict.keys()) for adict in dict_list]  # list of lists
    flat_keys = [key for sublist in all_keys for key in sublist]  # flatten
    unique_keys = np.unique(flat_keys)
    if len(unique_keys) != len(flat_keys):
        raise ValueError("Non-unique keys found in dict_list")


def get_value(dict_list, key):
    """
    Gets a value from a list of dicts.
    """
    for adict in dict_list:
        value = adict.get(key)
        if value is not None:
            return value
        value = adict.get(f"log({key})")
        if value is not None:
            return np.exp(value)
    raise KeyError(f"Key '{key}' not found within dict_list.")


def F(x, p):
    check_duplicate_keys([x, p])
    alpha = get_value([x, p], 'alpha')
    beta = get_value([x, p], 'beta')
    c = get_value([x, p], 'c')
    # c = 0.0 if (type(beta) == np.float64 or type(beta) == float) else gv.gvar('0(0)')
    mpi = get_value([x, p], 'mpi')
    fpi = get_value([x, p], 'fpi')
    a2 = get_value([x, p], 'a2')
    a = np.sqrt(a2)
    L = get_value([x, p], 'L')
    lam_chi = np.sqrt(2) * 2 * np.pi * fpi
    eps_pi = mpi / lam_chi
    result = (
        1
        + eps_pi**2 * (np.log(eps_pi**2) - 1 + c - f0(mpi*a*L, L) + 2*f1(mpi*a*L, L))
        + alpha * a2
    )
    result *= beta * lam_chi**4 / (4*np.pi)**2
    return result


def Fbreakdown(x, p):
    check_duplicate_keys([x, p])
    alpha = get_value([x, p], 'alpha')
    beta = get_value([x, p], 'beta')
    c = get_value([x, p], 'c')
    # c = 0.0 if (type(beta) == np.float64 or type(beta) == float) else gv.gvar('0(0)')
    mpi = get_value([x, p], 'mpi')
    fpi = get_value([x, p], 'fpi')
    a2 = get_value([x, p], 'a2')
    a = np.sqrt(a2)
    L = get_value([x, p], 'L')
    # derived variables
    mpiL = mpi*a*L
    lam_chi = np.sqrt(2) * 2 * np.pi * fpi
    eps_pi = mpi / lam_chi
    eps2 = eps_pi**2
    # pieces
    prefactor = beta * lam_chi**4 / (4*np.pi)**2
    analytic = prefactor * (1 + eps2 * (c-1))
    logs = prefactor * eps2 * np.log(eps2)
    finite_volume = prefactor * eps2 * (2 * f1(mpiL, L) - f0(mpiL, L))
    discretization = prefactor * alpha * a2
    return {
        'full': analytic + logs + finite_volume + discretization,
        'analytic': analytic,
        'logs': logs,
        'finite_volume': finite_volume,
        'discretization': discretization,
    }


def Fctm(x, p):
    check_duplicate_keys([x, p])
    # alpha = get_value([x, p], 'alpha')
    beta = get_value([x, p], 'beta')
    c = get_value([x, p], 'c')
    # c = 0.0 if (type(beta) == np.float64 or type(beta) == float) else gv.gvar('0(0)')
    lam_chi = get_value([x, p], 'lam_chi')
    eps_pi = get_value([x, p], 'eps_pi')
    result = (1 + eps_pi**2 * (np.log(eps_pi**2) - 1 + c))
    result *= beta * lam_chi**4 / (4*np.pi)**2
    return result


def F3(x, p):
    check_duplicate_keys([x, p])
    alpha = get_value([x, p], 'alpha')
    beta = get_value([x, p], 'beta')
    c = get_value([x, p], 'c')
    # c = 0.0 if (type(beta) == np.float64 or type(beta) == float) else gv.gvar('0(0)')
    mpi = get_value([x, p], 'mpi')
    fpi = get_value([x, p], 'fpi')
    a2 = get_value([x, p], 'a2')
    a = np.sqrt(a2)
    L = get_value([x, p], 'L')
    # derived variables
    mpiL = mpi*a*L
    lam_chi = np.sqrt(2) * 2 * np.pi * fpi
    eps_pi = mpi / lam_chi
    eps2 = eps_pi**2
    result = (
        1
        + eps2 * (3*np.log(eps2) - 1 + c - f0(mpiL, L) + 2*f1(mpiL, L))
        + alpha * a2
    )
    result *= eps2 * beta * lam_chi**4 / (4*np.pi)**2
    # TODO: be careful with factor of eps_pi**2
    # Does it naturally belong on the LHS or RHS of (82) in the notes?
    return result


def F3ctm(x, p):
    check_duplicate_keys([x, p])
    beta = get_value([x, p], 'beta')
    c = get_value([x, p], 'c')
    # c = 0.0 if (type(beta) == np.float64 or type(beta) == float) else gv.gvar('0(0)')
    lam_chi = get_value([x, p], 'lam_chi')
    eps_pi = get_value([x, p], 'eps_pi')
    eps2 = eps_pi**2
    result = 1 + eps2 * (3*np.log(eps2) - 1 + c)
    result *= eps2 * beta * lam_chi**4 / (4*np.pi)**2
    return result


def besselk0(x):
    """Bessel function of second kind K_0(x)."""
    if isinstance(x, gv.GVar):
        f = scipy.special.kn(0, x.mean)
        dfdx = -scipy.special.kn(1, x.mean)
        return gv.gvar_function(x, f, dfdx)
    else:
        return scipy.special.kn(0, x)


def besselk1(x):
    """Bessel function of second kind K_1(x)."""
    if isinstance(x, gv.GVar):
        f = scipy.special.kn(1, x.mean)
        dfdx = -0.5*(scipy.special.kn(0, x.mean) + scipy.special.kn(2, x.mean))
        return gv.gvar_function(x, f, dfdx)
    else:
        return scipy.special.kn(1, x)


def f0(x, ns):
    """
    f_0(m L) = -2 \sum_{\vec{n}, \abs{n}\neq 0} K_0(mL \abs{n})
    """
    test = hasattr(x, '__len__') + hasattr(ns, '__len__')
    tmp_x = x
    tmp_ns = ns
    result = 0.0*x
    if np.all(ns < 0):
        return result

    if test == 2:
        # Vector data - fine as it is
        pass
    elif test == 0:
        # Scalar data - temporarily wrap as iterable
        tmp_x = [tmp_x]
        tmp_ns = [tmp_ns]
        result = [result]
    else:
        raise ValueError("Incomensurate x and ns", x, ns)

    for idx in range(len(result)):
        for n_vec in itertools.product(range(tmp_ns[idx]), repeat=3):
            n = np.linalg.norm(n_vec)
            if n == 0:
                continue
            result[idx] += besselk0(tmp_x[idx]*n)
            if n > 10:
                break
        result[idx] *= -2

    if test == 0:
        # Unpackage scalar data
        result = result[0]

    return result


def f1(x, ns):
    """
    f_1(m L) = 4 \sum_{\vec{n}, \abs{n}\neq 0} \frac{K_1(mL \abs{n})}{m L \abs{n}}
    """

    test = hasattr(x, '__len__') + hasattr(ns, '__len__')
    tmp_x = x
    tmp_ns = ns
    result = 0.0*x
    if np.all(ns < 0):
        return result

    if test == 2:
        # Vector data - fine as it is
        pass
    elif test == 0:
        # Scalar data - temporarily wrap as iterable
        tmp_x = [tmp_x]
        tmp_ns = [tmp_ns]
        result = [result]
    else:
        raise ValueError("Incomensurate x and ns", x, ns)

    for idx in range(len(result)):
        for n_vec in itertools.product(range(tmp_ns[idx]), repeat=3):
            n = np.linalg.norm(n_vec)
            if n == 0:
                continue
            result[idx] += besselk1(tmp_x[idx]*n)
            if n > 10:
                break
        result[idx] *= 4 / tmp_x[idx]

    if test == 0:
        # Unpackage scalar data
        result = result[0]

    return result


callat = {
    'O1':      gv.gvar("-0.0191(13)"),
    'O1prime': gv.gvar("-0.0722(49)"),
    'O2':      gv.gvar("-0.0368(31)"),
    'O2prime': gv.gvar("0.0116(10)"),
    'O3':      gv.gvar("0.000185(10)"),
}

poare = {
    'O1':      gv.gvar("-0.01451(78)"),
    'O1prime': gv.gvar("-0.0613(24)"),
    'O2':      gv.gvar("-0.0278(13)"),
    'O2prime': gv.gvar("0.00762(41)"),
    'O3':      gv.gvar("0.0000907(30)"),
}

class LatticeMultiplicities:
    def __init__(self, sizes):
        self.data = {}
        for L in sizes:
            tmp = pd.DataFrame(itertools.product(range(L), repeat=3), columns=['x', 'y', 'z'])
            tmp['n'] = tmp[['x','y','z']].apply(lambda triplet: np.linalg.norm(triplet), axis=1)
            tmp = pd.DataFrame(
                tmp.groupby('n').size()).reset_index().rename(columns={0:'multiplicity'}
            )
            self.data[L] = tmp


class ChiralFitter:
    def __init__(self, data):
        self.data = data
        self.fit = None
        self.fit_full = None
        self.models = {
            'O1': F,
            'O2': F,
            'O3': F3,
            'O1prime': F,
            'O2prime': F,
        }
        self._pdg = {
            'mpi': 0.1349768,  # MeV
            'fpi': 0.1302,     # MeV
        }

    def __call__(self, key, prior_width=1.0, **kwargs):
        self.fit = self.run_fit(key, prior_width, **kwargs)
        self.fit_full = self.run_fit_full(key, prior_width, **kwargs)

    def run_fit_full(self, key, prior_width=1.0, **kwargs):
        """
        Fit including errors on x and on y. Note: 18 fit coefficients, 
        """
        y = self.data[key].values
        x = {'L': self.data['ns'].values}
        prior = {
            'mpi': self.data['mpi'].values,
            'fpi': self.data['fpi'].values,
            'a2': 1/self.data['ainv'].values**2,
            'alpha': gv.gvar(0, prior_width),
            'beta': gv.gvar(0, prior_width),
            'c': gv.gvar(0, prior_width),
            # 'c': gv.gvar(0, prior_width * 1e-5),
        }
        return lsqfit.nonlinear_fit(data=(x,y), fcn=self.models[key], prior=prior, **kwargs)

    def run_fit(self, key, prior_width=1.0, **kwargs):
        y = self.data[key].values
        x = {
            'L': self.data['ns'].values,
            'mpi': gv.mean(self.data['mpi'].values),
            'fpi': gv.mean(self.data['fpi'].values),
            'a2': gv.mean(1/self.data['ainv'].values**2),
        }
        prior = {
            'alpha': gv.gvar(0, prior_width),
            'beta': gv.gvar(0, prior_width),
            'c': gv.gvar(0, prior_width),
            # 'c': gv.gvar(0, prior_width * 1e-5),
        }
        return lsqfit.nonlinear_fit(data=(x,y), fcn=self.models[key], prior=prior, **kwargs)

    def evaluate_at_physical_point(self, name):

        if name == 'basic':
            fit = self.fit
        elif name == 'full':
            fit = self.fit_full
        else:
            raise ValueError("Unrecognized name", name)
        x_phys = {
            'L': -1,
            'a2': 0.0,  # Continuum
            'mpi': self._pdg['mpi'],  # MeV
            'fpi': self._pdg['fpi'],  # MeV
        }
        p_phys = {key: fit.p[key] for key in ['alpha', 'beta', 'c']}
        # p_phys = {key: fit.p[key] for key in ['alpha', 'beta']}
        return fit.fcn(x_phys, p_phys)

    def plot_results(self, key, ax):
        if key in ['O1', 'O2', 'O1prime', 'O2prime']:
            fctm = Fctm
        elif key in ['O3']:
            fctm = F3ctm
        else:
            raise NotImplementedError(f"Unrecognized key {key}")

        kwargs = {'markeredgewidth':2, 'capsize': 5, 'ms': 4}
        # Data
        x = (self.data['mpi'] / (np.sqrt(2) * 2 * np.pi * self.data['fpi']))**2
        x = x.values
        y = self.data[key].values
        y = y * (self._pdg['fpi'] / self.data['fpi'])**4
        plt.errorbar(ax, x, y, fmt='o', color='k', label='Data', **kwargs)

        # Fit result (including x and y errors)
        x = (self.fit_full.p['mpi'] / (np.sqrt(2) * 2 * np.pi * self.fit_full.p['fpi']))**2
        y = self.fit_full.fcn(self.fit_full.x, self.fit_full.p)
        y = y * (self._pdg['fpi'] / self.data['fpi'])**4
        plt.errorbar(ax, x, y, fmt='^', color='b', label='Fit (x and y errors)', **kwargs)

        # Interpolation
        x = {
            'lam_chi': (np.sqrt(2) * 2 * np.pi) * (self._pdg['fpi']),
            'eps_pi': np.linspace(0, np.sqrt(0.1)),
        }
        p = {key: self.fit_full.p[key] for key in ['alpha', 'beta', 'c']}
        # p = {key: self.fit_full.p[key] for key in ['alpha', 'beta']}
        y = fctm(x, p)
        plt.errorbar(ax, x['eps_pi']**2, y,
                     color=ax.lines[-1].get_color(), bands=True, alpha=0.5, label='interpolation')

        # Physical point (extrapolated including x and y errors)
        x = np.array([(self._pdg['mpi'] / (np.sqrt(2) * 2 * np.pi * self._pdg['fpi']))**2])
        y = [self.evaluate_at_physical_point('full')]
        plt.errorbar(ax, x, y, fmt='^', color='b', **kwargs)

        # Fit result (including y errors only)
        x = (self.fit.x['mpi'] / (np.sqrt(2) * 2 * np.pi * self.fit.x['fpi']))**2
        y = self.fit.fcn(self.fit.x, self.fit.p)
        y = y * (self._pdg['fpi'] / self.data['fpi'])**4
        plt.errorbar(ax, x, y, fmt='s', color='g', label='Fit (y errors only)', **kwargs)

        # Physical point (extrapolated including y errors only)
        x = np.array([(self._pdg['mpi'] / (np.sqrt(2) * 2 * np.pi * self._pdg['fpi']))**2]) * 1.05
        y = [self.evaluate_at_physical_point('basic')]
        plt.errorbar(ax, x, y, fmt='s', color='g', **kwargs)

        # Interpolation
        x = {
            'lam_chi': (np.sqrt(2) * 2 * np.pi) * (self._pdg['fpi']),
            'eps_pi': np.linspace(0, np.sqrt(0.1)),
        }
        p = {key: self.fit.p[key] for key in ['alpha', 'beta', 'c']}
        y = fctm(x, p)
        plt.errorbar(ax, x['eps_pi']**2, y,
                     color=ax.lines[-1].get_color(), bands=True, alpha=0.5, label='interpolation')


        # CalLat result
        x = np.array([(self._pdg['mpi'] / (np.sqrt(2) * 2 * np.pi * self._pdg['fpi']))**2])
        y = callat[key]
        plt.errorbar(ax, x, y, fmt='x', color='r', label='CalLat result', **kwargs)

        # Patrick Oare result
        x = np.array([(self._pdg['mpi'] / (np.sqrt(2) * 2 * np.pi * self._pdg['fpi']))**2]) * 0.98
        y = poare[key]
        plt.errorbar(ax, x, y, fmt='x', color='y', label='Patrick result', **kwargs)

