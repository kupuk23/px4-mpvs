import numpy as np

K = np.array(
                [
                    [500.0, 0.0, 320.0],  # fx, 0, cx
                    [0.0, 500.0, 240.0],  # 0, fy, cy
                    [0.0, 0.0, 1.0],  # 0, 0, 1
                ]
            )

def calc_interaction_matrix(s, Z):
        
        L = np.zeros((s.shape[0], 6))

        N = int(s.shape[0] / 2)
        for i in range(N):
            x, y = s[i * 2], s[i * 2 + 1]
            depth = Z[i]
            # normalize the point
            x_n = (x - K[0, 2]) / K[0, 0]
            y_n = (y - K[1, 2]) / K[1, 1]

            row = i * 2

            L[row, 0] = -1.0 / depth
            L[row, 1] = 0.0
            L[row, 2] = x_n / depth
            L[row, 3] = x_n * y_n
            L[row, 4] = -(1.0 + x_n * x_n)
            L[row, 5] = y_n

            # For vy
            L[row + 1, 0] = 0.0
            L[row + 1, 1] = -1.0 / depth
            L[row + 1, 2] = y_n / depth
            L[row + 1, 3] = 1.0 + y_n * y_n
            L[row + 1, 4] = -x_n * y_n
            L[row + 1, 5] = -x_n

        return L  # cs.MX((8,6))