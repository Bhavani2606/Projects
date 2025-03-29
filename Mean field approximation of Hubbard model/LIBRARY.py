#to store all the functions used in this project

import numpy as np
import matplotlib.pyplot as plt
import math

def H_con_dia(N, U, t, nr):#to construct the Hamiltonian for up-spin electrons in a diamond lattice
    H_up = np.zeros((N, N))
    for i in range (0, N):
        H_up[i, i] = U*0.5
    mid = int((nr+3)/2)
    
    m = 1
    array = []
    for i in range (1, mid+1):
        if i != 1:
            for k in range (0, 2):
                row = []
                for j in range (0, i):
                    row.append(m)
                    m+=1
                array.append(row)
        else:
            row = []
            for j in range (0, i):
                row.append(m)
                m+=1
            array.append(row)
    
    for i in range (mid-1, 0, -1):
        if i != 1:
            for k in range (0, 2):
                row = []
                for j in range (0, i):
                    row.append(m)
                    m+=1
                array.append(row)
        else:
            row = []
            for j in range (0, i):
                row.append(m)
                m+=1
            array.append(row)
    # print(array)
    for i in range (0, int(len(array)/2)):
        for j in range(0, len(array[i])):
            # print(i, j)
            if i%2 == 0:
                H_up[(array[i][j]-1), (array[i+1][j]-1)] = -t
                H_up[array[i+1][j]-1, array[i][j]-1] = -t
                H_up[array[i][j]-1, array[i+1][j+1]-1] = -t
                H_up[array[i+1][j+1]-1, array[i][j]-1] = -t
            else:
                H_up[array[i][j]-1, array[i+1][j]-1] = -t
                H_up[array[i+1][j]-1, array[i][j]-1] = -t
    
    for i in range (int(len(array)/2), (len(array) - 1)):
        for j in range(0, (len(array[i])-1)):
            # print(i, j)
            if i%2 == 0:
                H_up[array[i+1][j]-1, array[i][j]-1] = -t
                H_up[array[i][j]-1, array[i+1][j]-1] = -t
                H_up[array[i+1][j]-1, array[i][j+1]-1] = -t
                H_up[array[i][j+1]-1, array[i+1][j]-1] = -t
            else:
                H_up[array[i][j]-1, array[i+1][j]-1] = -t
                H_up[array[i+1][j]-1, array[i][j]-1] = -t
                H_up[array[i][j], array[i+1][j]] = -t
                H_up[array[i+1][j], array[i][j]] = -t
            

    return H_up
# print(H)

def H_con_tri(N, U, t):#to construct the Hamiltonian for up-spin electrons fro triangle lattice
    H_up = np.zeros((N, N))
    for i in range (0, N):
        H_up[i, i] = U*0.5

    for i in range (2, N):
        r = N
        for k in range (2, i+1):
            r -= 2*k
        r -= i
        if r == 0:
            mid = i
            break

    array = []
    m = 1
    for k in range (2, mid+1):
        row = []
        for l in range (0, k):
            row.append(m)
            m+=1
        array.append(row)

    row2 = []
    for l in range (0, mid):
        row2.append(m)
        m+=1
    array.append(row2)


    for n in range (mid, 1, -1):
        row = []
        for b in range (0, n):
            row.append(m)
            m+=1
        array.append(row)

    for a in range (0, N):
        for b in range (0, a):
            for i in range (0, len(array)):
                if i == mid-1:
                    for j in range (1, len(array[i]), 2):
                        # print(i, j, 100)
                        if j+1 < len(array[i]):
                            # print(a, b, "ab")
                            if a+1 == array[i][j+1] and b+1 == array[i][j]:
                                H_up[a, b] = -t
                                H_up[b, a] = -t
                                # print(a, b, "ab-added")
                else:
                    for j in range (0, len(array[i]), 2):
                        # print(i, j)
                        if j+1 < len(array[i]):
                            # print(a+1, b+1, "ab", array[i][j+1], array[i][j])
                            if a+1 == array[i][j+1] and b+1 == array[i][j]:
                                H_up[a, b] = -t
                                H_up[b, a] = -t
                                # print(a, b, t, "ab added-2", H[a, b])
            for i in range (0, len(array)-1):
                maxl = min(len(array[i]), len(array[i+1]))
                for j in range (1, maxl+1):
                    if b+1 == array[i][-j] and a+1 == array[i+1][-j]:
                        H_up[a, b] = -t
                        H_up[b, a] = -t
                        # print(a, b)
    return H_up
# print(H)

def H_block(H_up, H_down):#to construct the full hamiltonian using up spin and down spin hamiltonian
    N = len(H_up)
    Z = np.zeros((N, N))
    H_1 = np.concatenate([H_up, Z], axis = 1)
    H_2 = np.concatenate([Z, H_down], axis = 1)
    H_f = np.vstack([H_1, H_2])
    return H_f

def scf(H_f, tol):
    N = int(len(H_f)/2)
    den1 = np.zeros(int(2*N))
    den2 = np.ones_like(den1)
    diff = np.max(abs(den2- den1))
    diff_list = []
    countlist = []

    count = 0
    patience = 0
    while count <= 1000 and patience < 10:
        count +=1
        # eigen_val = eigen_val2
        eigval_2, eigr = np.linalg.eig(H_f)
        # print(eigval_2, eigr, count, "\n")
        eigr = eigr.transpose()
        # print(eigr[0], eigval_2[0])
        # print((H_f @ (eigr[0].transpose()))/eigval_2[0], "res")
        den1 = np.array(den2)
        eigen_val2 = np.array(eigval_2)
        # print("eigen values", eigen_val2)
        eigval_2.sort()
        # print("sort", eigval_2)
        chem_pot = eigval_2[N-1]
        den2 = np.zeros(2*N)
        deg = 0
        for ev in eigen_val2:
            if ev == chem_pot:
                deg += 1
        # print("chemical potential = ", chem_pot, "degeneracy = ", deg)
        nos = 0
        for i in range (0, 2*N):
            if eigen_val2[i] <= chem_pot:
                if eigen_val2[i] != chem_pot:
                    den2 += abs(eigr[i])**2
                else:
                    den2 += (abs(eigr[i])**2)*(1/deg)
                nos +=1
        # print (nos, "n")  
        for i in range (0, N*2):
            if i >= N:
                H_f[i, i] = H_f[i, i]*(den2[i-N])
            else:
                H_f[i, i] = H_f[i, i]*(den2[i+N])
        diff = np.max(abs(den2 - den1))
        if count > 2:
            diff_list.append(np.linalg.norm(den2 - den1))
            countlist.append(count)
        if diff < tol:
            patience += 1
        # print("\nden, ",den1, den2, diff, "\n")
        # print(diff)
        # print(den2)
        # np.set_printoptions(precision=2)
        # np.savetxt('h_f2.txt',H_f, delimiter='\t',fmt='%1.2f')
        # print(np.sum(den2))
    # print(count, "count")
    den_lat = []
    for i in range (0, N):
        den_lat.append([den2[i] + den2[i+N]])
    den_lat = np.array(den_lat)

    spinup_list = []
    spindown_list = []
    lattice_index = []
    for k in range (0, N):
        lattice_index.append([k+1])
        spinup_list.append([den2[k]])
        spindown_list.append([den2[k+N]])
    spinup_list = np.array(spinup_list)
    spindown_list = np.array(spindown_list)
    lattice_index = np.array(lattice_index)

    netspin_list = (spinup_list - spindown_list)*0.5
    return (den2, lattice_index, spinup_list, spindown_list, netspin_list, count, diff, countlist, diff_list)


def generate_hexagon(center, radius):
    """Generate the coordinates for the vertices of a hexagon given a center and radius.

    Args:
        center (tuple): The (x, y) coordinates of the hexagon's center.
        radius (float): The radius of the hexagon.

    Returns:
        list: A list of tuples representing the (x, y) coordinates of the hexagon's vertices.
    """
    angle_offset = np.pi / 6  # 30 degrees in radians for alignment
    points = []
    
    # Calculate the vertices of the hexagon
    for i in range(6):
        angle = angle_offset + i * (2 * np.pi / 6)  # Angle for each vertex
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        points.append((x, y))
    
    return points

def plot_hexagon(ax, hexagon_points):
    """Plot the hexagon on a graph.

    Args:
        hexagon_points (list): List of (x, y) tuples representing the vertices of the hexagon.
    """
    # Convert points to separate x and y lists for plotting
    x_coords = [p[0] for p in hexagon_points]
    y_coords = [p[1] for p in hexagon_points]
    
    # Close the loop to complete the hexagon
    x_coords.append(x_coords[0])
    y_coords.append(y_coords[0])

    # Plot the hexagon
    hexagon = ax.plot(x_coords, y_coords, color = 'blue')


# Example usage
# center = (0, 0)
# radius = 1
# hexagon = generate_hexagon(center, radius)

# plot_hexagon(hexagon)

import matplotlib.pyplot as plt
import numpy as np

# Function to draw circles with given radii and centers
def draw_circles(centers, radii):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot each circle
    for center, radius in zip(centers, radii):
        # Create a circle with the given center and radius
        circle = plt.Circle(center, radius, fill=True, color='r', linewidth=2)
        
        # Add the circle to the plot
        ax.add_artist(circle)
    
    # # Set the axis limits to ensure all circles are visible
    # ax.set_xlim(min([c[0] for c in centers]) - max(radii), max([c[0] for c in centers]) + max(radii))
    # ax.set_ylim(min([c[1] for c in centers]) - max(radii), max([c[1] for c in centers]) + max(radii))
    
    # # Keep the axis ratio to 1 to maintain the circle shapes
    # ax.set_aspect('equal', adjustable='box')
    
    # Show the plot
    # plt.xlabel("X-Axis")
    # plt.ylabel("Y-Axis")
    # plt.title("Circles with Given Radii and Centers")
    # plt.grid(True)
    # plt.show()

# Example usage with a list of centers and radii
centers = [(1, 2), (4, 6), (7, 3)]
radii = [2, 1.5, 1]

draw_circles(centers, radii)
