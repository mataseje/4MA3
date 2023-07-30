import numpy as np

def rayleigh_quotient_iteration(A, x, index, output_data):
    tolerance = 0.0001
    max_iterations = 20

    n = len(A[0])
    I = np.eye(n, dtype=np.float64, order='C')

    # Starting Loop Values
    yk_inf = 0
    xk = x
    prev_rq = None

    for k in range(max_iterations):
        new_row = []
        xT = xk.T
        r_quotient = np.dot(xT, np.dot(A, xk)) / np.dot(xT, xk)

        # r_matrix = A - (r_quotient*I)
        r_matrix = np.linalg.inv(A)
        yk = GaussElimination(r_matrix, xk)
        yk_inf = np.linalg.norm(yk, ord=np.inf)
        xk = yk / yk_inf

        if prev_rq is not None:
            if abs(r_quotient - prev_rq) <= tolerance:
                new_row.extend([index+1, f'e{index+1} {str(x)}', r_quotient, k])
                output_data.extend([new_row])
                print("\nRAYLEIGH QUOTIENT ITERATION METHOD:")
                print(f"The difference between rayleigh quotients has reached the tolerance ({tolerance}), see values below:")
                print(f"(Rayleigh Quotient ({r_quotient}) - Previous Rayleigh Quotient({prev_rq}) = {(r_quotient - prev_rq)}")
                break
        prev_rq = r_quotient
    
    if (len(output_data) == 4):
        print_table(output_data)
    return


def GaussElimination(H_matrix, b_vector):
    n = len(H_matrix)
    L = np.eye(n, dtype=np.float64, order='C')
    U = H_matrix.copy()
    b = b_vector.copy()
    for k in range(n):
        max_row = np.argmax(np.abs(U[k:n, k])) + k
        U[[k, max_row]] = U[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]

        if U[k][k] == 0:
            print(f'Pivot is zero')
            break
        
        for i in range(k+1, n):
            factor = U[i][k] / U[k][k]
            L[i][k] = factor
            for j in range(k, n):
                U[i][j] -= (factor * U[k][j])
            b[i] -= (factor * b[k])          
    yk = backward_substitution(U, b, n)
    return yk

def backward_substitution(matrix, vector, n):
    output_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n-1, -1, -1):
        output_matrix[i] = vector[i]
        for j in range(i+1, n):
            output_matrix[i] -= matrix[i][j] * output_matrix[j]
        output_matrix[i] /= matrix[i][i]
    return output_matrix

def print_table(output_data):
    print()
    print("RAYLEIGH QUOTIENT ITERATION METHOD:")
    headers = ['#', 'Starting Eigenvector', 'Eigenvalue', 'Number of Iterations']
    table_data = [headers] + output_data
    column_widths = [max(len(str(row[i])) for row in table_data) for i in range(len(headers))]
    header_row = " | ".join(format(header, f"{width}s") for header, width in zip(headers, column_widths))
    print(header_row)
    print("-" * len(header_row))

    for row in output_data:
        row_str = " | ".join(format(cell, f"{width}") for cell, width in zip(row, column_widths))
        print(row_str)

def qr_iteration(A):
    tolerance = 0.0001
    max_iterations = 100
    Ak = np.copy(A)

    for i in range(max_iterations):
        Q, R = gramSchmidt(Ak)
        Ak = R @ Q
        diag_norm = np.linalg.norm(Ak - np.diag(np.diagonal(Ak)))
        eigenvalues = np.diag(Ak)
        
        if diag_norm <= tolerance:
            output = f"""
----------------------------
QR ITERATION METHOD:

The eigen values are: {eigenvalues}
The number of iterations for the convergence is: {i}
        """
            print(output)
            break
    

def gramSchmidt(A):
    rows = len(A) 
    Q = np.zeros_like(A)
    R = np.zeros((rows, rows))

    for i in range(rows):
        a = A[:, i]
        for j in range(i):
            R[j, i] = np.dot(Q[:, j], A[:, i])
            a -= R[j, i] * Q[:, j]
        R[i, i] = np.linalg.norm(a)
        Q[:, i] = a / R[i, i]
    return Q, R


def main():
    A = np.array([[2.9766, 0.3945, 0.4198, 1.1159],
                  [0.3945, 2.7328, -0.3097, 0.1129],
                  [0.4198, -0.3097, 2.5675, 0.6079],
                  [1.1159, 0.1129, 0.6079, 1.7231]])
    # Starting Eigenvectors
    e1 = np.array([1, 1, 1, 1])
    e2 = np.array([2, 2, 2, 2])
    e3 = np.array([0, 0, 0, 1])
    e4 = np.array([1, 1, 0, 1])

    x_values = [e1, e2, e3, e4]
    output_data = []
    for index, x in enumerate(x_values):
        rayleigh_quotient_iteration(A, x, index, output_data)
    
    qr_iteration(A)


if __name__ == '__main__':
    main()
