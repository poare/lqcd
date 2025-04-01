
import sys
import numpy as np

class KrylovVector:
    """
    Dynamic vector that represents the filtered Krylov space.
    """

    def __init__(self, data, op = 'D', src = 'b'):
        """
        Assumes data is a numpy vector. 
        """
        self.vec = data
        self.dim = len(data)
        self._op = op
        self._src = src

    def op(self):
        """
        Acts the operator on the Krylov vector by shifting the components to the right.
        """
        return KrylovVector(
            np.insert(self.vec, 0, 0),
            op = self._op,
            src = self._src
        )
    
    def copy(self):
        return KrylovVector(self.vec, op = self._op, src = self._src)

    def __add__(self, other):
        """
        Adds two KrylovVectors by padding the smaller vector with zeros. 
        """
        if isinstance(other, (KrylovVector)):
            d1, d2 = self.dim, other.dim
            if d1 >= d2:
                larger, smaller = self.vec, other.vec
            else:
                larger, smaller = other.vec, self.vec
            sum = larger.copy()
            for i in range(min(d1, d2)):
                sum[i] += smaller[i]
            return KrylovVector(sum, op = self._op, src = self._src)
        # elif :
        return KrylovVector(self.vec + other, op = self._op, src = self._src)

    def __sub__(self, other):
        """
        Adds two KrylovVectors by padding the smaller vector with zeros. 
        """
        if isinstance(other, KrylovVector):
            d1, d2 = self.dim, other.dim
            if d1 >= d2:
                larger, smaller = self.vec, -other.vec
            else:
                larger, smaller = -other.vec, self.vec
            diff = larger.copy()
            for i in range(min(d1, d2)):
                diff[i] += smaller[i]
            return KrylovVector(diff, op = self._op, src = self._src)
        return KrylovVector(self.vec - other, op = self._op, src = self._src)           # if not a Krylov vector

    def __mul__(self, other):
        # if isinstance(other, (int, float, np.float32, np.float64, np.complex64, np.complex128)):
        #     return KrylovVector(self.vec * other, op = self._op, src = self._src)
        # d1, d2 = self.dim, other.dim
        # if d1 >= d2:
        #     larger, smaller = self.vec, other.vec
        # else:
        #     larger, smaller = other.vec, self.vec
        # prod = larger.copy()
        # for i in range(min(d1, d2)):
        #     prod[i] *= smaller[i]
        # return KrylovVector(prod, op = self._op, src = self._src)
        if isinstance(other, KrylovVector):
            d1, d2 = self.dim, other.dim
            if d1 >= d2:
                larger, smaller = self.vec, other.vec
            else:
                larger, smaller = other.vec, self.vec
            prod = larger.copy()
            for i in range(min(d1, d2)):
                prod[i] *= smaller[i]
            return KrylovVector(prod, op = self._op, src = self._src)
        # elif :                    # other specific cases
        return KrylovVector(self.vec * other, op = self._op, src = self._src)
    
    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return KrylovVector(-self.vec)

    def __str__(self):
        if self.dim == 0:
            return ''
        tokens = [f'{self._op}^{i}{self._src}' for i in range(self.dim)]
        tokens[0] = self._src
        if self.dim > 1:
            tokens[1] = f'{self._op}{self._src}'
        outstr = ''
        for ci, x in zip(self.vec, tokens):
            outstr += f'{ci} {x} + '
        return outstr[:-3]

    def __repr__(self):
        return str(self.vec)

# for testing
v1 = KrylovVector(np.array([2, 3]), op = 'A', src = 'psi')
v2 = KrylovVector(np.array([8, 4, 6, 1, 3]))

class PolyGCR:
    """
    Constructs the GCR polynomial, given the alpha and beta coefficients. 
    """

    def __init__(self, alphas, betas):
        self.alphas = alphas
        self.betas = betas
        self.N = len(alphas)
        self.init_coeffs()
        self.compute_poly_coefficients()
    
    def init_coeffs(self):
        """Initializes the polynomial recursion."""
        self.psi         = KrylovVector(np.array([], dtype = np.complex64))
        self.r           = KrylovVector(np.array([1.], dtype = np.complex64))
        self.p           = [
            KrylovVector(np.array([1.], dtype = np.complex64))
        ]
    
    def compute_poly_coefficients(self):
        """
        Computes the GCR polynomial coefficients based on the update scheme,
        psi_{j+1} --> psi_j + alpha_j p_j
        r_{j+1}   --> r_j - \alpha_j D p_j
        p_{j+1}   --> r_{j+1} + \sum_{i=0}^j \beta_{ij} p_i
        """
        for j in range(self.N):
            alpha = self.alphas[j]
            self.psi = self.psi + alpha * self.p[j]
            self.r   = self.r - alpha * self.p[j].op()
            if j >= len(self.betas):             # don't need final set of betas
                print(f'Iteration j = {j}. psi = {self.psi}, r = {self.r}.\n')
                break
            beta = self.betas[j]
            pnew = KrylovVector(self.r.vec)
            for i in range(j + 1):
                pnew += beta[i] * self.p[i]
            self.p.append(pnew)
            # print(f'Iteration j = {j}. psi = {self.psi}, r = {self.r}, p = {self.p[-1]}.\n')
        return self.psi

    def eval_poly(self, x):
        """
        Evaluates the stored GCR polynomial at the point x. 
        """
        px = 0.
        # print(f'Evaling poly. psi = {self.psi.vec}')
        for i in range(len(self.psi.vec)):
            px += self.psi.vec[i] * (x**i)
        return px
    
    def eval_poly_stable(self, x, store_all = False):
        """
        Evaluates the stored GCR polynomial at the point x by iterating through the 
        standard set of iterations. This method is numerically stable and not subject 
        to fine cancellations. 

        If store_all is true, returns a vector (q_m(x)) for 1\leq m\leq n, i.e. the polynomial 
        at x for each iteration. 
        """
        psi = 0.
        r = 1.
        p = [1.]
        # print('evaling poly')
        if store_all:
            psi_lst = []
        for j in range(self.N):
            # print(f'iteration {j}')
            alpha = self.alphas[j]
            psi   = psi + alpha * p[j]
            if store_all:
                psi_lst.append(psi)
            r     = r - alpha * x * p[j]
            if j >= len(self.betas):             # don't need final set of betas
                print(f'Iteration j = {j}. psi = {self.psi}, r = {self.r}.\n')
                break
            beta = self.betas[j]
            pnew = r
            for i in range(j + 1):
                pnew += beta[i] * p[i]
            p.append(pnew)
            # print(f'Iteration j = {j}. psi = {psi}, r = {r}, p = {p[-1]}.\n')
        if store_all:
            return psi_lst
        return psi

    def get_psi(self):
        return self.psi
    
    def get_resids(self):
        return self.r

    def get_conj_resids(self):
        return self.p

def main(args):
    print('********** N = 1 RESULTS **********')
    alphas0 = [0.244300601 + 1j*0.00013007545]
    betas0  = [
        [-0.184661284]
    ]
    gcr0 = PolyGCR(alphas0, betas0)
    print(gcr0.get_psi())

    print('Testing polynomial stable vs unstable.')
    x0 = 1.2 + 1j*0.7
    print(f'Unstable q_n(x) = {gcr0.eval_poly(x0)}')
    print(f'Stable q_n(x) = {gcr0.eval_poly_stable(x0)}')

    print('\n********** N = 2 RESULTS **********')
    alphas1 = [0.244300601 + 1j*0.00013007545, 0.285370966 + 1j*-0.000160704478]
    betas1  = [
        [-0.184661284], 
        [-0.149530292, -0.143286162]
    ]
    gcr1 = PolyGCR(alphas1, betas1)
    print(gcr1.get_psi())

    print('\n********** N = 5 RESULTS **********')
    alphas2 = [
        a[0] + 1j*a[1] for a in
        [(0.244300601,0.00013007545), (0.285370966,-0.000160704478), (0.26497512,-7.74209838e-05), (0.290557579,-0.000406907695), (0.271661483,-0.000656724506)]
    ]
    betas2  = [
        [-0.184661284], 
        [-0.149530292, -0.143286162], 
        [-0.253664787, -0.205662175, -0.131759912],
        [-0.175774146, -0.35276841, -0.18673159, -0.108101303], 
        [-0.195023979, -0.347355205, -0.331549312, -0.167088889,-0.104738669],
    ]
    gcr2 = PolyGCR(alphas2, betas2)
    print(gcr2.get_psi())

    print('Testing polynomial stable vs unstable.')
    x0 = 2.3 + 1j*0.48
    print(f'Unstable q_n(x) = {gcr2.eval_poly(x0)}')
    print(f'Stable q_n(x) = {gcr2.eval_poly_stable(x0)}')

    # Dsq for N = 3
    print('\n******* Dsq, N = 3 RESULTS ********')
    alphas3 = [0.0659889805 + 1j*2.23536979e-19, 0.0688501046 + 1j*2.05501011e-19, 0.06022993 + 1j*2.4091912e-20]
    betas3 = [
        [0.265919406], 
        [0.464138416, -5.33778239e-15], 
        [0.621436336, 2.78827548e-14, -7.99838167e-15]
    ]
    gcr3 = PolyGCR(alphas3, betas3)
    print(gcr3.get_psi())

    print('Testing polynomial stable vs unstable.')
    x0 = -1.7 + 1j*4.3
    print(f'Unstable q_n(x) = {gcr3.eval_poly(x0)}')
    print(f'Stable q_n(x) = {gcr3.eval_poly_stable(x0)}')

if __name__ == '__main__':
    main(sys.argv)