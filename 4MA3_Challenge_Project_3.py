import numpy as np


def get_derivatives(a,p,r,n):
    deriv_X = a*(n)*((1+r)**n-1)
    deriv_Y = (p/r)*(n*((1+r)**(n-1)))
    return deriv_X, deriv_Y

def get_functions(a,p,r,n):
    X = a*((1+r)**n)
    Y = p*(((1+r)**n - 1)/r)
    return X, Y
     
def NewtonA():
    n = 10
    a = 100000
    p = 10000
    r = 0.06

    conv_tolerance = 1e-6
    max_iterations = 10000
    output_data = []
    next_x = n
    for k in range(max_iterations):
        new_row = []
        n = next_x
        X, Y = get_functions(a,p,r,n)
        fx = X - Y
        deriv_X, deriv_Y = get_derivatives(a,p,r,n)
        f_x = deriv_X - deriv_Y
        hk = (fx / f_x)
        next_x = n - hk

        new_row.extend([k, n, fx, f_x, hk])
        new_row = [round(value, 6) for value in new_row]
        output_data.extend([new_row])

        if abs(fx) < conv_tolerance:
            break

    headers = ['k', 'n = n + hk','f(xk) = X-Y', "f'(xk) = deriv_X-deriv_Y", "hk = f(xk)/f'(xk)"]
    print_table(headers, output_data)
    print(f'It will take {next_x} years to repay the loan\n')

def NewtonB():
    # rxy - x(1+y) = 0
    # -xy + (d-y)(1+y) = 0
    r = 5
    d = 1
    max_iterations = 1000
    conv_tolerance = 1e-6
    output_data = []

    # initial guess for x0
    x = 7
    y = 5
    xn = np.array([x, y], dtype=np.float64)

    # ref. jacobian matrix
    # [ -y   -2y-x ]
    # [ 4y-1    4x ]
    jf_origin = np.array([[(-x),     (-2*(y))-x], 
                          [((4*y)-1),       4*x]])

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
        
        new_row.extend([k, xn[0], xn[1], fx[0], fx[1]])
        new_row = [round(value, 6) for value in new_row]
        output_data.extend([new_row])
        
        xn = so + x

        condition_met = all(abs(value) < conv_tolerance for value in fx)
        if condition_met:
            break

    headers = ['k', 'x', 'y', '5xy - x(1+y) == 4xy - x ', '-xy + (1-y)(1+y) == -y^2 - xy + 1']
    print_table(headers, output_data)
    return

def fxn(x):
    # 4xy - x
    fx_00 = (4*x[0]*x[1] - x[0])
    # -y^2 - xy - 1
    fx_01 = ((-x[1]**2) - (x[0]*x[1]) + 1)

    return np.array([fx_00, fx_01])

def JF_0(x):
    # 4y-1
    jf_10 = (4*(x[1])-1)
    # 4x
    jf_11 = (4*(x[0]))   
    return jf_10, jf_11

def JF_1(x):
    # -y
    jf_00 = (-x[1])
    # -2y - x
    jf_01 = (-2*(x[1]) - x[0])
    return jf_00, jf_01

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
    return output_matrix

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

    NewtonA()
    NewtonB()

if __name__ == '__main__':
    main()