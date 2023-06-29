def forward_substitution(matrix, vector, n):
    '''
    Algorithm 1: Forward Substitution (Lower Triangle)
    '''
    output_matrix = [[0.0] * n for _ in range(n)]
    print(f'\n\033[92mForward Substitution - Results\033[0m')
    for i in range(n):
        output_matrix[i] = vector[i]
        print(f"\nx{i+1}:\n{vector[i]} = {' + '.join([f'({k:.4f})x{(j+1)}' for j,k in enumerate(matrix[i])])}")
        for j in range(i):
            print(f"= {output_matrix[i]:.4f} - ({matrix[i][j]:.4f} * {output_matrix[j]:.4f})")
            output_matrix[i] -= matrix[i][j] * output_matrix[j]
        print(f"x{i+1} = {output_matrix[i]:.4f} / {matrix[i][i]:.4f}")
        output_matrix[i] /= matrix[i][i]
        print(f"x{i+1} = {output_matrix[i]:.4f}")
    display_x(output_matrix)
    return output_matrix


def backward_substitution(matrix, vector, n):
    '''
    Algorithm 2: Backward Substitution (Upper Triangle)
    '''
    output_matrix = [[0.0] * n for _ in range(n)]
    print(f'\n\033[92mBackward Substitution - Results\033[0m')
    for i in range(n-1, -1, -1):
        output_matrix[i] = vector[i]
        print(f"\nx{i+1}:\n{vector[i]} = {' + '.join([f'({k:.4f})x{(j+1)}' for j,k in enumerate(matrix[i])])}")
        for j in range(i+1, n):
            print(f"= {output_matrix[i]:.4f} - ({matrix[i][j]:.4f} * {output_matrix[j]:.4f})")
            output_matrix[i] -= matrix[i][j] * output_matrix[j]
        print(f"x{i+1} = {output_matrix[i]:.4f} / {matrix[i][i]:.4f}")
        output_matrix[i] /= matrix[i][i]
        print(f"x{i+1} = {output_matrix[i]:.4f}\n")
    display_x(output_matrix)
    return output_matrix


def GaussElimination(H_matrix, b_vector):
    '''
    Algorithm 3: Gaussian Elimination
    '''
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
                display_matrix(U)
                print()
            b_vector[i] -= factor * b_vector[k]
            display_vector(b_vector)
    print(f'\n\033[92mFINAL MATRIX:  H\033[0m')
    display_matrix(H_matrix)
    print(f'\n\033[92mFINAL VECTOR:  b\033[0m')
    display_vector(b_vector)
    
    return L, U


def display_matrix(matrix):
    columns = len(matrix[0])
    max_width = max(max(len(f"{element:.4f}") for element in row) for row in matrix)

    print('Updated Matrix (H):')
    print('-' * ((max_width + 3) * columns + 8))
    for row in matrix:
        for element in row:
            print(f'{element:{max_width+2}.4f}', end="\t")
        print()
    print('-' * ((max_width + 3) * columns + 8))

def display_vector(b_vector):
    print('Updated Vector (b):')
    print('-------------')
    for item in b_vector:
        print(f'{item:.4f}')
    print('-------------')

def display_x(b_vector):
    print('\nx:')
    print('-------------')
    for index, item in enumerate(b_vector, start=1):
        print(f'x{index} = {item:.4f}')
    print('-------------')

def generateHB(n):
    H = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            H[i][j] = 1/((i+1) + (j+1) - 1)
    return H

def calculate_residual(A, b, back, fore):
    # Get the dimensions of the matrix
    rows = len(A)
    cols = len(A[0])
    # Calculate residual (r1) for backward substitution results
    Ax_r1 = [0] * rows
    for i in range(rows):
        for j in range(cols):
            Ax_r1[i] += A[i][j] * back[j]
    r1 = []
    for i in range(len(b)):
        r1.append(b[i] - Ax_r1[i])
    # Calculate residual (r2) for foreward substitution results
    Ax_r2 = [0] * rows
    for i in range(rows):
        for j in range(cols):
            Ax_r2[i] += A[i][j] * fore[j]
    r2 = []
    for i in range(len(b)):
        r2.append(b[i] - Ax_r2[i])
    return r1, r2, back, fore


def calculate_delta_x(x, back, fore):
    # Calculate delta x for backward substitution results
    delta_x1 = []
    for i in range(len(x)):
        delta_x1.append(back[i] - x[i])
    delta_x2 = []
    for i in range(len(x)):
        delta_x2.append(fore[i] - x[i]) 
    return delta_x1, delta_x2

def calculate_b_vector(H_matrix, n_vector):
    rows = len(H_matrix)
    cols = len(H_matrix[0])
    # Multiply H matrix by n_vector
    result = [0] * rows
    for i in range(rows):
        for j in range(cols):
            result[i] += H_matrix[i][j] * n_vector[j]
    return result

def find_infinitynorm(vector):
    infinity_norm = max(vector, key=abs)
    return infinity_norm

def monitor_condH(H_matrix):
    H = H_matrix

def gauss_jordan(n):
    init_matrix = generateHB(n)
    aug_matrix = [row + [int(i == j) for j in range(n)] for i, row in enumerate(init_matrix)]
    for col in range(n):
        max_row = max(range(col, n), key=lambda r: abs(aug_matrix[r][col]))
        aug_matrix[col], aug_matrix[max_row] = aug_matrix[max_row], aug_matrix[col]
        pivot = aug_matrix[col][col]
        aug_matrix[col] = [element / pivot for element in aug_matrix[col]]
        for row in range(n):
            if row != col:
                multiplier = aug_matrix[row][col]
                aug_matrix[row] = [element - multiplier * aug_matrix[col][i] for i, element in enumerate(aug_matrix[row])]
    
    return [row[n:] for row in aug_matrix], init_matrix

def matrix_norm(matrix):
    infinity_norm = max(sum(abs(element) for element in row) for row in matrix)
    return infinity_norm


def main():
    n = 2
    n_vector = [1]*n

    # Generate Hilbert Matrix
    H_matrix = generateHB(n)
    print(f'\nn = {n}')
    print(f'H = {H_matrix}')
    print(f'n-vector = {n_vector}')
    
    # Calculate b-vector given n-vector entries equal 1
    b_vector = calculate_b_vector(H_matrix, n_vector)

    print(f'\n\033[91mStarting matrix (H)\033[0m')
    display_matrix(H_matrix)
    print(f'\n\033[91mStarting Vector (b)\033[0m')
    display_vector(b_vector)

    # Use Gaussian Elimination to solve for 'approximate_x'
    print(f'\n\033[91mStart Gaussian Elimination\033[0m')
    L, U = GaussElimination(H_matrix, b_vector)
    print(f'\n\033[92mLower Matrix (L)\033[0m')   
    display_matrix(L)
    print(f'\n\033[92mUpper Matrix (U)\033[0m')
    display_matrix(U)

    # Calculate Residual (r = b - Hx)
    r1, r2, back, fore = calculate_residual(H_matrix, b_vector, backward_substitution(U, b_vector, n), forward_substitution(L, b_vector, n))
    
    # Calculate Delta X  (delta_x = approximate_x - x)
    delta_x1, delta_x2 = calculate_delta_x(n_vector, back, fore)

    # Find infinity norm of the residual and delta x
    r_infinitynorm = find_infinitynorm(r2)
    delta_x_infinitynorm = find_infinitynorm(delta_x2)

    # Monitor cond(H)
    H_inverse, H_original = gauss_jordan(n)
    print(f'\n\033[92mInverse Matrix (H^-1)\033[0m')
    display_matrix(H_inverse)
    norm_H = matrix_norm(H_original)
    norm_H_inv = matrix_norm(H_inverse)
    cond_H = norm_H * norm_H_inv

    # SUMMARY RESULTS
    print(f'\n\033[92mSUMMARY\033[0m')
    formatted_b_vector = [f'{element:.4f}' for element in b_vector]
    print(f'b_vector                = {formatted_b_vector}')
    print(f'n_vector                = {n_vector}\n')
    
    approximate_x = [f'{element:.4f}' for element in fore]
    print(f'approximate_x           = {approximate_x}')
    delta_x = [f'{element:.4f}' for element in delta_x2]
    print(f'delta_x                 = {delta_x}')
    formatted_r2 = [f'{element:.4f}' for element in r2]
    print(f'r                       = {formatted_r2}\n')

    print(f'r infinity-norm         = {abs(r_infinitynorm):.4f}')
    print(f'delta_x infinity-norm   = {abs(delta_x_infinitynorm):.4f}')
    print(f'cond(H)                 = {float(cond_H):.4f}\n')

    # Backward Substitution Values
    # print(f'delta_x1                 = {delta_x1}')
    # print(f'r1 (backward sub)       = {r1}')

if __name__ == '__main__':
    main()