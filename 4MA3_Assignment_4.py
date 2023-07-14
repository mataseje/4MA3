import numpy as np


def rayleigh_quotient_iteration(A, x, tolerance, max_iterations):
    
    n = len(A[0])
    I = np.eye(n, dtype=np.float64, order='C')
    for i in range(max_iterations):
        xTAx = np.dot(np.dot(x, A), x.T)
        print(f'xT(A)(x) ={np.dot(np.dot(x, A), x.T)}')
        xTx = np.dot(x, x.T)
        print(f'xT(A)(x) = {np.dot(x, x.T)}')
        r_quotient = xTAx / xTx
        print(f'r_quotient = {xTAx / xTx}')
        next_vector = x / (A - (np.dot(x, I)))
        print(next_vector)

        num_iterations = i
        eigenvalue = r_quotient
        if np.linalg.norm(x - next_vector) < tolerance:
            break

    return eigenvalue, num_iterations


def print_table(output_data):
    print()
    headers = ['#', 'Starting Eigenvector', 'Eigenvalue', 'Number of Iterations']
    table_data = [headers] + output_data
    column_widths = [max(len(str(row[i])) for row in table_data) for i in range(len(headers))]
    header_row = " | ".join(format(header, f"{width}s") for header, width in zip(headers, column_widths))
    print(header_row)
    print("-" * len(header_row))

    for row in output_data:
        row_str = " | ".join(format(cell, f"{width}") for cell, width in zip(row, column_widths))
        print(row_str)


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
    A = Q @ R
    eigen = np.diagonal(A)
    output = f"""
    The eigen values are: {A}
    The number of iterations for the convergence is: {i}
    """
    print(output)

    return Q, R


def main():

    A = np.array([[2.9766, 0.3945, 0.4198, 1.1159],
                  [0.3945, 2.7328, -0.3097, 0.1129],
                  [0.4198, -0.3097, 2.5675, 0.6079],
                  [1.1159, 0.1129, 0.6079, 1.7231]])
    
    x1 = np.array([1, 1, 0, 1])
    x2 = np.array([0, 1, 1, 1])
    x3 = np.array([1, 0, 1, 1])
    x4 = np.array([1, 1, 0, 1])

    tolerance = 0.0001
    max_iterations = 100
    row1 = eigenvalues, num_iterations = rayleigh_quotient_iteration(A, x1, tolerance, max_iterations)
    
    output_data = [
        ["1", f'e1: {x1}', row1[0], row1[1]],
        ["2", f'e2: {x2}', row1[0], row1[1]],
        ["3", f'e3: {x3}', row1[0], row1[1]],
        ["4", f'e4: {x4}', row1[0], row1[1]]]
    table = print_table(output_data)

    qr = gramSchmidt(A)


if __name__ == '__main__':
    main()

