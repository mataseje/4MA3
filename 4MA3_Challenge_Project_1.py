def forward_substitution(matrix, vector, n):
    '''
    Algorithm 1: Forward Substitution (Lower Triangle). Ex: 
    1  0  0  0 
    3  1  0  0
    4  1  1  0 
    2  1  1  1 
    '''
    output_matrix = [[0.0] * n for _ in range(n)]
    print(f'\n\033[92mForward Substitution - Results\033[0m')
    for i in range(n):
        output_matrix[i] = vector[i]
        print(f"\nx{i+1}:\n{vector[i]} = {' + '.join([f'({k})x{(j+1)}' for j,k in enumerate(matrix[i])])}")
        for j in range(i):
            print(f"= {output_matrix[i]} - ({matrix[i][j]} * {output_matrix[j]})")
            output_matrix[i] -= matrix[i][j] * output_matrix[j]
        print(f"x{i+1} = {output_matrix[i]} / {matrix[i][i]}")
        output_matrix[i] /= matrix[i][i]
        print(f"x{i+1} = {output_matrix[i]}")
    display_x(output_matrix)


def backward_substitution(matrix, vector, n):
    '''
    Algorithm 2: Backward Substitution (Upper Triangle). Ex:
    1  2  1 -1
    0 -4  1  7
    0  0 -2  1
    0  0  0 -2
    '''
    output_matrix = [[0.0] * n for _ in range(n)]
    print(f'\n\033[92mBackward Substitution - Results\033[0m')
    for i in range(n-1, -1, -1):
        output_matrix[i] = vector[i]
        print(f"\nx{i+1}:\n{vector[i]} = {' + '.join([f'({k})x{(j+1)}' for j,k in enumerate(matrix[i])])}")
        for j in range(i+1, n):
            print(f"= {output_matrix[i]} - ({matrix[i][j]} * {output_matrix[j]})")
            output_matrix[i] -= matrix[i][j] * output_matrix[j]
        print(f"x{i+1} = {output_matrix[i]} / {matrix[i][i]}")
        output_matrix[i] /= matrix[i][i]
        print(f"x{i+1} = {output_matrix[i]:.3f}\n")
    display_x(output_matrix)


def lu_decomposition(a_matrix, b_vector):
    '''
    Algorithm 3: Gaussian Elimination (Upper Triangle). Ex:
    '''
    n = len(a_matrix)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0
    U = a_matrix

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
    
    print(f'\n\033[92mFINAL MATRIX:  A=\033[0m')
    display_matrix(a_matrix)
    print(f'\n\033[92mFINAL VECTOR:  b=\033[0m')
    display_vector(b_vector)
    print(f'\n\033[92mLower Matrix (L) =\033[0m')
    display_matrix(L)
    forward_substitution(L, b_vector, n)

    print(f'\n\033[92mUpper Matrix (U) =\033[0m')
    display_matrix(U)
    backward_substitution(U, b_vector, n)


def display_matrix(matrix):
    print('Updated Matrix (A):')
    print('----------------------------')
    for row in matrix:
        for element in row:
            print(f'{element:.3f}', end="\t")
        print()
    print('----------------------------')


def display_vector(b_vector):
    print('Updated Vector (b):')
    print('--------')
    for item in b_vector:
        print(f'{item:.3f}')
    print('--------\n')


def display_x(b_vector):
    print('\nx:')
    print('--------')
    for index, item in enumerate(b_vector, start=1):
        print(f'x{index} = {item:.3f}')
    print('--------\n')

def generateHB(n):
    H = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            H[i][j] = 1/((i+1) + (j+1) - 1)
    return H


def main():
    n = 2
    A_matrix = generateHB(n)

    b_vector = [1, 1]

    print(f'\n\033[91mStarting matrix (A) =\033[0m')
    display_matrix(A_matrix)

    print(f'\n\033[91mStarting Vector (b) =\033[0m')
    display_vector(b_vector)

    print(f'\n\033[91mStart Gaussian Elimination\033[0m')
    lu_decomposition(A_matrix, b_vector)


if __name__ == '__main__':
    main()