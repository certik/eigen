from numpy import array, dot, eye
from scipy.sparse import csc_matrix

from eigen import solve_eig_numpy, solve_eig_newton

def main():
    A = array([
        [2.9766, 0.3945, 0.4198, 1.1159],
        [0.3945, 2.7328, -0.3097, 0.1129],
        [0.4198, -0.3097, 2.5675, 0.6079],
        [1.1159, 0.1129, 0.6079, 1.7231],
        ])
    lam = 4
    x = array([0.7606, 0.1850, 0.3890, 0.4858])
    print A
    print "lam=", lam
    print x
    print "-"*80
    n = len(A)
    one = eye(n)
    Q = solve_eig_numpy(csc_matrix(A), csc_matrix(one))
    lam, x = Q[3]
    x = array(x.flat)
    print lam, x
    x = array([ 1, 2, 3, 4])
    lam, x = solve_eig_newton(csc_matrix(A), x0=x, eps=1e-10, debug=True)
    print "newton:"
    print lam, array(x.flat)

main()
