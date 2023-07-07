import numpy as np

def solve_alpha(x):
    alpha = x[0]
    s = np.power(np.linalg.norm(x[1:]), 2)
    v = x.copy()
    if s == 0:
        t = 0
        print(f's = 0')
        exit
    t = np.sqrt(alpha**2 + s)
    v[0] = alpha - t if np.all(alpha <= 0) else -s / (alpha + t)
    t = 2 * v[0]**2 / (s + v[0]**2)
    v /= v[0]
    return v, t

def householder(A):
    m, n = A.shape
    R = A.copy()
    for j in range(n):
        v, t = solve_alpha(R[j:, j])
        H = np.identity(m)
        H[j:, j:] -= t * np.outer(v, v)
        R = H @ R
        R[j:, j:] *= -1
        display_matrix(R)
    return R

def display_matrix(matrix):
    columns = len(matrix[0])
    max_width = max(max(len(f"{element:.4f}") for element in row) for row in matrix)
    print('\nUpdated Matrix (H):')
    print('-' * ((max_width + 5) * columns + 8))
    for row in matrix:
        for element in row:
            print(f'{element:{max_width+2}.4f}', end="\t")
        print()
    print('-' * ((max_width + 5) * columns + 8))

def main():
    A = np.array([[1, -1, 4], 
                  [1, 4, -2],
                  [1, 4, 2],
                  [1, -1, 0]], dtype=np.longdouble)
    display_matrix(A)
    R = householder(A)


if __name__ == '__main__':
    main()