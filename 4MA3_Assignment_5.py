import numpy as np

def newton_method_nonlinear_equation(x):
    # f(x) = (x-2)^2 + 4x -8
    # f'(x) = 2(x)
    conv_tolerance = 1e-5
    max_iterations = 40
    output_data = []
    next_x = x
    for k in range(max_iterations):
        new_row = []
        x = next_x
        fx = ((x-2) ** 2) + 4*(x) - 8
        f_x = 2*(x)
        hk = (fx / f_x)
        next_x = x - hk

        new_row.extend([k, x, fx, f_x, hk])
        new_row = [round(value, 6) for value in new_row]
        output_data.extend([new_row])

        if abs(hk) < conv_tolerance:
            break

    headers = ['k', 'xk','f(xk)', "f'(xk)", 'hk']
    print_table(headers, output_data)

def newton_method_system_of_equations():
    max_iterations = 20
    conv_tolerance = 1e-4
    output_data = []

    # initial guess for x0
    x1 = 1
    x2 = 2
    xn = np.array([x1, x2])
    # ref. initial f(x)  
    fx_origin = np.array([[x1 + 2*x2 - 2], 
                          [(x1**2) + (4*(x2**2)) - 4]])
    # ref. jacobian matrix
    jf_origin = np.array([[(1), (2)], 
                          [(2*x1), (8*x2)]])

    for k in range (0, max_iterations):
        new_row = []
        x = xn
        fx = fxn(x)
        jfx = np.zeros((2, 2))
        jfx_0 = JF_0(x)
        jfx[0] = jfx_0
        jfx_1 = JF_1(x)
        jfx[1] = jfx_1

        neg_fx = []
        for n in fx:
            neg_fx.append(n*(-1))
        so = GaussElimination(jfx, neg_fx)
        xn = so + x
        
        new_row.extend([k, xn[0], xn[1]])
        new_row = [round(value, 6) for value in new_row]
        output_data.extend([new_row])
        
        condition_met = all(abs(value) < conv_tolerance for value in fx)
        if condition_met:
            break

    headers = ['k', 'x1^k','x2^k']
    print_table(headers, output_data)
    return

def GaussElimination(H_matrix, b_vector):
    n = len(H_matrix)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0
    U = H_matrix
    for k in range(n-1):
        for i in range(k+1, n):
            if U[k][k] == 0:
                break
            factor = U[i][k] / U[k][k]
            L[i][k] = factor
            for j in range(k, n):
                U[i][j] -= factor * U[k][j]
            b_vector[i] -= factor * b_vector[k]
    sk = backward_substitution(U, b_vector, n)
    return sk

def backward_substitution(matrix, vector, n):
    output_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n-1, -1, -1):
        output_matrix[i] = vector[i]
        for j in range(i+1, n):
            output_matrix[i] -= matrix[i][j] * output_matrix[j]
        output_matrix[i] /= matrix[i][i]
    print(f'backward_sub results: {output_matrix}')
    return output_matrix

def fxn(x):
    fx_00 = (x[0] + 2*x[1] - 2)
    fx_01 = ((x[0]**2) + 4*((x[1])**2) - 4)
    return np.array([fx_00, fx_01])

def JF_0(x):
    # Calculate first array of Jacobian matrix
    jf_00 = 1
    jf_01 = 2
    return jf_00, jf_01

def JF_1(x):
    # Calculate second array of Jacobian matrix
    jf_10 = (2*x[0])
    jf_11 = (8*x[1])   
    return jf_10, jf_11

def print_table(headers, output_data):
    print()
    table_data = [headers] + output_data
    column_widths = [max(len(str(column)) for column in column_data) for column_data in zip(*table_data)]
    header_row = " | ".join(format(header, f"{width}s") for header, width in zip(headers, column_widths))
    print("-" * len(header_row))
    print(header_row)
    print("-" * len(header_row))

    for row in output_data:
        row_str = " | ".join(format(cell, f"{width}") for cell, width in zip(row, column_widths))
        print(row_str)
    print()

def main():

    for x in range(0, 4):
        print(f'\n\nStarting "x" value = {x}')
        newton_method_nonlinear_equation(x)

    newton_method_system_of_equations()

if __name__ == '__main__':
    main()