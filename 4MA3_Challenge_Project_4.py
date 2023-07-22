import numpy as np

def NewtonsMethod():
    conv_tolerance = 1e-5
    max_iterations = 2000

    # Starting Point
    starting_points = [[-1, 1], [0,1], [2,1]]
    
    for n in starting_points:
        output_data = []
        x0 = n
        print(f'SOLVING THE MINIMUM FOR STARTING POINT {x0}')
        
        # Hessian Matrix
        Hf = np.array([[-1200*((x0[0])**2) - (400*x0[1]) + 2,     (-400*x0[0])], 
                       [-400*x0[0],                                        200]])
        
        x_next = x0

        for k in range(0, max_iterations):
            new_row = []

            x = x_next
            fxk = solve_function(x)
            gf_xk = gradiant_fx(x)
            neg_gfxk = []
            for element in gf_xk:
                neg_gfxk.append(element * -1)
            sk = backward_substitution(Hf, neg_gfxk, len(Hf))
            print(f'x: {x}')
            print(f'gf_xk: {gf_xk}')
            print(f'neg_gfxk: {neg_gfxk}')
            print(f'sk: {sk}')
            x_next = np.array(x) + np.array(sk)
            print(f'x_next: {x_next}')

            if np.all(abs(x_next - x) < conv_tolerance):
                print(f'''
STARTING POINT {x0}
x(k+1) did not change over two iterations within a tolerance of 1e-5:
x({k}) = {x_next}
x({k-1}) = {x}
x({k}) - x({k-1}) = ({x_next - x})
Therefore, we have converged and {x_next} is the minimum)
                ''')
                new_row.extend([k, x, fxk, gf_xk])
                output_data.extend([new_row])
                break

            new_row.extend([k, x, fxk, gf_xk])
            output_data.extend([new_row])
        
        headers = ['k', '(x,y)', 'f(x,y)', 'gradiant_f(x,y)']
        print_table(headers, output_data)
    return

def gradiant_fx(x):
    # [-400x^3 - 400xy + 2x + 2, -200x^2 + 200y]
    gradiant_f = [-400*(x[0]**3) - 400*x[0]*x[1] + 2*x[0] + 2, -200*(x[0]**2) + 200*x[1]]
    return gradiant_f

def solve_function(x):
    # -100x^4 - 200yx^2 + 100y^2 + x^2 - 2x + 1
    fx = ((-100*(x[0]**4)) - (200*(x[0]**2)*(x[1]) + (100*(x[1]**2)) + (x[0]**2)) - 2*(x[0]) + 1)
    return fx


def backward_substitution(matrix, vector, n):
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
        formatted_cells = [format(str(cell), f"{width}") for cell, width in zip(row, column_widths)]
        row_str = " | ".join(formatted_cells)
        print(row_str)
    print()

def main():
    NewtonsMethod()


if __name__ == '__main__':
    main()