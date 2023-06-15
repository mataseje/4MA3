import math

def Householder(matrix):
    
    columns = len(matrix[0])
    rows = len(matrix)
    for k in range(columns):
        x = [matrix[i][k] for i in range(k, rows)]
        alpha = math.sqrt(sum(xi ** 2 for xi in x[1:]))
        if x[0] <= 0:
            x[0] -= alpha
        else:
            x[0] = -alpha
        beta = math.sqrt(sum(xi ** 2 for xi in x))
        if beta != 0:
            v = [xi / beta for xi in x]
            for j in range(k, columns):
                scalar = 0.0
                for i in range(k, rows):
                    scalar += v[i - k] * matrix[i][j]
                for i in range(k, rows):
                    matrix[i][j] -= 2.0 * v[i - k] * scalar
    return matrix

def backward_substitution(matrix, vector, n):
    print(f'\n\033[92mBackward Substitution - Results\033[0m')
    output_matrix = [[0.0] * n for _ in range(n)]

    for i in range(n-1, -1, -1):
        output_matrix[i] = vector[i]
        print(f"\nx{i+1}:\n{vector[i]:.3f} = {' + '.join([f'({k:.3f})x{(j+1)}' for j,k in enumerate(matrix[i])])}")
        for j in range(i+1, n):
            print(f"= {output_matrix[i]:.3f} - ({matrix[i][j]:.3f} * {output_matrix[j]:.3f})")
            output_matrix[i] -= matrix[i][j] * output_matrix[j]
            print(f"x{i+1} = {output_matrix[i]:.3f} / {matrix[i][i]:.3f}")
        output_matrix[i] /= matrix[i][i]
        print(f"x{i+1} = {output_matrix[i]:.3f}\n")
    display_vector(output_matrix)

def display_matrix(matrix):
    print('----------------------------')
    for row in matrix:
        for element in row:
            print("{:.3f}".format(element), end="\t")
        print()
    print('----------------------------')

def display_vector(b_vector):
    print('--------')
    for item in b_vector:
        print("{:.3f}".format(item))
    print('--------\n')



def main():
    A_matrix = [[1, -1, 0, 0], 
                [1, 0, -1, 0],
                [1, 0, 0, -1],
                [0, 1, -1, 0],
                [0, 1, 0, -1],
                [0, 0, 1, -1],]
    b_vector = [1.23, 4.45, 1.61, 3.21, 0.45, -2.75]
    
    print('\nInitial Matrix (A):\n')
    display_matrix(A_matrix)

    H_matrix = Householder(A_matrix)
    
    print('Updated Matrix (A):')
    display_matrix(H_matrix)
    backward_substitution(H_matrix, b_vector, len(A_matrix[0]))

if __name__ == '__main__':
    main()