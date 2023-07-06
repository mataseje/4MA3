import numpy as np

def householder(x, b, j):
    alpha = x[0]
    s = np.power(np.linalg.norm(x[1:]), 2)
    v = x.copy()
    if s == 0:
        tau = 0
        exit
    else:
        t = np.sqrt(alpha**2 + s)
        v[0] = alpha - t if alpha <= 0 else -s / (alpha + t)
        tau = 2 * v[0]**2 / (s + v[0]**2)
        v /= v[0]
    b_hat = b.copy()
    b_hat[j:] -= tau * np.dot(v, b[j:]) * v
    return v, tau, b_hat

def qr_decomposition(A, b):
    m, n = A.shape
    R = A.copy()
    Q = np.identity(m)
    for j in range(n):
        v, tau, b = householder(R[j:, j], b, j)
        H = np.identity(m)
        H[j:, j:] -= tau * np.outer(v, v)
        R = H @ R
        Q = H @ Q
        R[j:, j:] *= -1
        b[j:] *= -1
        display_matrix(R)
    return Q, R, b

def calculate_delta_x(x, back):
    delta_x = []
    for i in range(len(x)):
        delta_x.append(back[i] - x[i])
    return delta_x

def backward_substitution(matrix, vector):
    n = len(matrix[0])
    threshold=1e-10
    output_matrix = [[0.0] * n for _ in range(n)]
    print(f'\n\033[92mBackward Substitution - Results\033[0m')
    for i in range(n-1, -1, -1):
        output_matrix[i] = vector[i]
        print(f"\nx{i+1}:\n{vector[i]} = {' + '.join([f'({k})x{(j+1)}' for j,k in enumerate(matrix[i])])}")
        for j in range(i+1, n):
            print(f"= {output_matrix[i]} - ({matrix[i][j]} * {output_matrix[j]})")
            output_matrix[i] -= matrix[i][j] * output_matrix[j]
        print(f"x{i+1} = {output_matrix[i]} / {matrix[i][i]}")
        if abs(matrix[i][i]) < threshold:
            output_matrix[i] = 0.0
        else:
            output_matrix[i] /= matrix[i][i]
        print(f"x{i+1} = {output_matrix[i]}\n")
    display_x(output_matrix)
    return output_matrix

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

def display_x(b_vector):
    print('\nx:')
    print('-------------')
    for index, item in enumerate(b_vector, start=1):
        print(f'x{index} = {item:.4f}')
    print('-------------')

def main():

    x = [2.95, 1.74, -1.45, 1.32]
    A = np.array([[1, -1, 0, 0], 
                  [1, 0, -1, 0],
                  [1, 0, 0, -1],
                  [0, 1, -1, 0],
                  [0, 1, 0, -1],
                  [0, 0, 1, -1]], dtype=np.longdouble)
    b = np.array([1.23, 4.45, 1.61, 3.21, 0.45, -2.75], dtype=np.longdouble)
    
    Q, R, b_hat = qr_decomposition(A, b)

    print(f'Q:\n{Q}')
    print(f'R:\n{R}')
    print(f'b_hat:\n{b_hat}')

    x_hat = backward_substitution(R, b_hat)
    delta_x = calculate_delta_x(x, x_hat)

    print(f'\nx_hat: [{", ".join([f"{x:.4f}" for x in x_hat])}]')
    print(f'delta_x: [{", ".join([f"{delta:.4f}" for delta in delta_x])}]\n')

if __name__ == '__main__':
    main()