# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:07:36 2021

@author: Neil McFarlane
"""

"""
Old functions from the cartiesian GSM program are retained here....
Many of the variables are undefined here (they are in the main program), so the functions cannot be successfully called here.
In order to test them, they must be input into the main program.
"""

def hessian_calc(LJ_structure):
    """
    
    Parameters
    ----------
    LJ_structure : string
        Respresents the LJ structure which the user has input for optimisation.

    Returns
    -------
    hess : array
        Gives the Hessian matrix for the LJ structure.

    """
  
    equations = open('eqn_out', 'a')
    equations.truncate(0)
  
    # The total number of atoms is taken from the length of the list x_coords.
    n_atoms = len(x_coords)
    
    # Using the total number of atoms, the shape of the Hessian is initialised in a Numpy array of dimensions 3N x 3N.
    hess = np.zeros([n_atoms * 3, n_atoms * 3], float)
    
    # The individual x, y, and z coordinates for every atom are initialised as labels in the list hess_labels.
    hess_labels = []
    for i in range(1, n_atoms + 1):
        x_label = 'x' + str(i)
        y_label = 'y' + str(i)
        z_label = 'z' + str(i)
        hess_labels.append(x_label)
        hess_labels.append(y_label)
        hess_labels.append(z_label)
  
    # The symbols and expression for the LJ potential with expanded r is initialised.
    xA, yA, zA, xB, yB, zB = sym.symbols('xA yA zA xB yB zB')
    LJ_expr  = (4 * Xe_epsilon * ((Xe_sigma / sym.sqrt((xA - xB)**2 + (yA - yB)**2 + (zA - zB)**2))**12 - (Xe_sigma / sym.sqrt((xA - xB)**2 + (yA - yB)**2 + (zA - zB)**2))**6))
    
    for i in range(0, n_atoms * 3):
        # In order to identify the atoms by number, the list variable atom is created using hess_labels.
        atom_lst = [int(num) for num in re.findall(r'\d+', hess_labels[i])]
        for x in atom_lst:
            atom = int(x)
   
        for j in range(0, n_atoms * 3):
            # Using the number of atoms as an upper bound, for each atom, the distance to every other atom is calculated.
            # If the distance is zero, then it is the same atom and is omitted from further calculation.
            # When the value is not zero and is less than the cut off distance, then r is added to the list r_list.
            interaction_list = []
            for l in range(1, n_atoms + 1):
                r = math.sqrt((float(x_coords[atom-1]) - float(x_coords[l-1]))**2 + (float(y_coords[atom-1]) - float(y_coords[l-1]))**2 + (float(z_coords[atom-1]) - float(z_coords[l-1]))**2)
                if r < cut_off and r > 0.00001:
                    if not r in interaction_list:
                        interaction_list.append(l)
            
            # For each term of the Hessian matrix, using interaction_list, every possible interaction and its corresponding second derivative is calculated.
            out_term = 0
            equations.write('For row ' + str(i) + ' and column ' + str(j) + '...' + '\n')
            for k in interaction_list: 
                symbol_string = 'x' + str(atom) + ' ' + 'y' + str(atom) + ' ' + 'z' + str(atom) + ' ' + 'x' + str(k) + ' ' + 'y' + str(k) + ' ' + 'z' + str(k)
                symbol_list = symbol_string.split()
                for item in symbol_list:
                    item = sym.symbols(item)
                LJ_pot = LJ_expr.subs([(xA, symbol_list[0]), (yA, symbol_list[1]), (zA, symbol_list[2]), (xB, symbol_list[3]), (yB, symbol_list[4]), (zB, symbol_list[5])])
                first_deriv = sym.diff(LJ_pot, hess_labels[i])
                second_deriv = sym.diff(first_deriv, hess_labels[j])
                added_term = second_deriv.subs([(symbol_list[0], x_coords[atom-1]), (symbol_list[1], y_coords[atom-1]), (symbol_list[2], z_coords[atom-1]), (symbol_list[3], x_coords[k-1]), (symbol_list[4], y_coords[k-1]), (symbol_list[5], z_coords[k-1])])
                equations.write(str(added_term) + '\n')
                out_term = out_term + added_term
            # The term which is added to the Hessian is rounded to mitigate errors down the line.
            hess[i,j] = round(out_term, 4)
            
    # The hessian should be a symmetric matrix by definition.
    # This criteria is checked in order to mitigate errors down the line.
    symmetry = (hess.transpose() == hess).all()
    if symmetry == False:
        print("ERROR: Hessian is not symmetric. Exiting program...")
        sys.exit()


    equations.close()
    # It is the initial hessian matrix, so it is saved to a file and returned.
    np.savetxt('init_hess', hess)
    return(hess)


def xyz_to_Zmat(coords):
    """
    
    // Function which generates the primitive internal coordinates of a defined structure using cartesian coordinates. //
    
    Parameters
    ----------
    coords : Numpy array
        Numpy array containing the x, y, and z coordinates of the stucture.

    Returns
    -------
    prim_coords : Numpy array
        Numpy array containing the atom numbers which have been determined to have a primitive internal coordinate.
    init_z_mat : saved file
        File which contains the initial Z-matrix in terms of distance between atoms, angles, and dihedrals.

    """
    
    # All atoms are defined as xenon in this implementation of the code. 
    # This could very easily be changed to another noble gas by altering the lines below as well as the LJ parameters.
    atom_names = []
    for i in range(0, len(coords)):
        atom_names.append('Xe_' + str(i + 1))

    # Using Scipy's cdist, the Numpy distance_matrix containing the distance of every atom to every other atom is created.
    distance_matrix = cdist(coords, coords)
    
    # Sub-function which calculates the angle between three atoms using Numpy's linear algebra routines.
    # It takes the arguments of the coordinate array, and the indices which define the three atoms in the coordinate array.
    def angle(coords, i, j, k):
        # Two vectors of the three coordinates are calculated.
        r_ij = coords[i] - coords[j]
        r_kj = coords[k] - coords[j]
        
        # Numerator and denominator are explicitly written as variables for ease of working.
        numerator = np.dot(r_ij, r_kj)
        denominator = np.linalg.norm(r_ij) * np.linalg.norm(r_kj)

        # Arccos is used to calculate the angle in radians.
        theta = math.acos(numerator / denominator)
        theta = 180.0 * theta / np.pi
            
        return theta
    
    # Sub-function which calculates the dihedral between four atoms using Numpy's linear algebra routines.
    # It takes the arguments of the coordinate array, and the indices which define the four atoms in the coordinate array.
    def dihedral(coords, i, j, k, l):
        # Three vectors of the four coordinates are calculated.
        r_ij = coords[i] - coords[j]
        r_lk = coords[l] - coords[k] 
        r_kj = coords[k] - coords[j]
        
        # Term 1 is written to variable term_1.
        combined_vecs = np.array([r_lk, r_ij, r_kj])
        det = np.linalg.det(combined_vecs)
        term_1 = np.sign(det)
        
        # The terms contained in term 2 are explicitly written.
        e = r_kj / np.linalg.norm(r_kj)
        a1 = r_ij - np.dot(np.dot(r_ij, e), e)
        a2 = r_lk - np.dot(np.dot(r_lk, e), e)
        
        # Numerator and denominator for term 2 are explicitly written as variables for ease of working.
        numerator = np.dot(a1, a2)
        denominator = np.linalg.norm(a1) * np.linalg.norm(a2)
        division = numerator / denominator
        
        # Series of if statements solve the value for term 2.
        # Arccos is used to calculate the angle in radians.
        if (division <= -1):
            term_2 = math.acos(-1)
        elif (division >= 1):
            term_2 = math.acos(1)
        else:
            term_2 = math.acos(division)
        
        # The dihedral angle is calculated.
        chi = term_1 * term_2
        chi = -180.0 - 180.0 * chi / np.pi
        if (chi <= -180.0):
            chi = chi + 360.0

        return chi
    
    # Three lists containing the primitive coordinates of radius, angle, and dihedral are initialised.
    rad_list = []
    angle_list = []
    dihedral_list = []
    
    # The number of atoms and number of dimensions are saved as variables using Numpy's shape.
    n_atoms, n_coords = coords.shape
    
    # The Numpy array containing the Z-matrix is initialised.
    # The Numpy array containing the primitive internal connectivity is initialised.
    z_matrix = np.zeros([len(coords), 7])
    prim_coords = np.zeros([0, 4])
    
    with open('init_z_mat', 'w') as f1:
        if n_atoms > 0:
            # The first atom is written and is defined as the origin of the Z-matrix.
            label = atom_names[0]
            
            # Written to initial Z-matrix file.
            f1.write(label)
            f1.write('\n')
            
            # Saved to Numpy array with Z-matrix.
            z_matrix[0][0] = 1
            
            if n_atoms > 1:
                # The second atom is written, with it's distance to the first.
                label = atom_names[1]
                rad_list.append(distance_matrix[0][1])
                r = '{:>11.5f}'.format(rad_list[0])
                
                # Written to initial Z-matrix file.
                f1.write('{:<3s} {:>4d}  {:11s}'.format(label, 1, r))
                f1.write('\n')
                
                # Saved to Numpy array with Z-matrix.
                z_matrix[1][0] = 2
                z_matrix[1][1] = 1
                z_matrix[1][2] = r

                # Saved to Numpy array with primitive connectivity.
                prim_coords = np.append(prim_coords, np.array([[2, 1, 0, 0]]), axis = 0)
                
                if n_atoms > 2:
                    # The third atom is written, with it's distance to the first, and it's angle between the first, second and third.
                    label = atom_names[2]
                    rad_list.append(distance_matrix[0][2])
                    r = '{:>11.5f}'.format(rad_list[1])
                    angle_list.append(angle(coords, 2, 0, 1))
                    a = '{:>11.5f}'.format(angle_list[0])
                    
                    # Written to initial Z-matrix file.
                    f1.write('{:<3s} {:>4d}  {:11s} {:>4d}  {:11s}'.format(label, 1, r, 2, a))
                    f1.write('\n')
                    
                    # Saved to Numpy array with Z-matrix.
                    z_matrix[2][0] = 3
                    z_matrix[2][1] = 1
                    z_matrix[2][2] = r
                    z_matrix[2][3] = 2
                    z_matrix[2][4] = a
                    
                    # Saved to Numpy array with primitive connectivity.
                    prim_coords = np.append(prim_coords, np.array([[3, 1, 0, 0]]), axis = 0)
                    prim_coords = np.append(prim_coords, np.array([[3, 1, 2, 0]]), axis = 0)

                    if n_atoms > 3:
                        # Beyond atom 3, then we must write the distance, angle and dihedral to the Z-matrix.
                        for i in range(3, n_atoms):
                            label = atom_names[i]
                            rad_list.append(distance_matrix[i-3][i])
                            r = '{:>11.5f}'.format(rad_list[i-1])
                            angle_list.append(angle(coords, i, i-3, i-2))
                            a = '{:>11.5f}'.format(angle_list[i-2])
                            dihedral_list.append(dihedral(coords, i, i-3, i-2, i-1))
                            d = '{:>11.5f}'.format(dihedral_list[i-3])
                            
                            # Written to initial Z-matrix file.
                            f1.write('{:3s} {:>4d}  {:11s} {:>4d}  {:11s} {:>4d}  {:11s}'.format(label, i-2, r, i-1, a, i, d)) 
                            f1.write('\n') 

                            # Saved to Numpy array with Z-matrix.
                            z_matrix[i][0] = i + 1
                            z_matrix[i][1] = i - 2
                            z_matrix[i][2] = r
                            z_matrix[i][3] = i - 1
                            z_matrix[i][4] = a
                            z_matrix[i][5] = i
                            z_matrix[i][6] = d   

                            # Saved to Numpy array with primitive connectivity.
                            prim_coords = np.append(prim_coords, np.array([[i+1, i-2, 0, 0]]), axis = 0)
                            prim_coords = np.append(prim_coords, np.array([[i+1, i-2, i-1, 0]]), axis = 0)
                            prim_coords = np.append(prim_coords, np.array([[i+1, i-2, i-1, i]]), axis = 0)

    return prim_coords  



def initial_B_mat(coords, prim_coords):
    """
    
    // Function which generates the B matrix from a set of defined cartesian and primitive internal coordinates. //
    
    Parameters
    ----------
    coords : Numpy array
        Numpy array containing the x, y, and z coordinates of the stucture.
    prim_coords : Numpy array
        Numpy array containing the atom numbers which have been determined to have a primitive internal coordinate.

    Returns
    -------
    B_mat : Numpy array
        Numpt array containing the B matrix used to convert between cartesian and internal coordinates.

    """
    
    # The B matrix is initialised.
    n_atoms = len(coords)
    n_prims = len(prim_coords)
    B_mat = np.zeros([n_prims, n_atoms * 3])
    
    # Sub-function which calculates analytical derivative expressions for the distance between two atoms using Numpy's linear algebra routines.
    # It takes the arguments of the coordinate array, and the indices which define the two atoms in the coordinate array.
    def rad_grad(coords, i, j):
        # A vector of the two coordinates is calculated.
        r_ij = coords[j] - coords[i]
        
        # The gradient matrix is initialised.
        grad = np.zeros([2, 3])
        
        # The analytical expressions for the derivatives are substituted.
        grad[0][:] = r_ij / np.linalg.norm(r_ij)
        grad[1][:] = -r_ij / np.linalg.norm(r_ij)

        return grad
    
    # Sub-function which calculates analytical derivative expressions for the angle between three atoms using Numpy's linear algebra routines.
    # It takes the arguments of the coordinate array, and the indices which define the three atoms in the coordinate array.
    def angle_grad(coords, i, j, k):
        # Two vectors of the three coordinates are calculated.
        r_ij = coords[i] - coords[j]
        r_kj = coords[k] - coords[j]
        
        # Numerator and denominator are explicitly written as variables for ease of working.
        numerator = np.dot(r_ij, r_kj)
        denominator = np.linalg.norm(r_ij) * np.linalg.norm(r_kj)

        # Arccos is used to calculate the angle in radians.
        theta = math.acos(numerator / denominator)
        
        # The gradient matrix is initialised.
        grad = np.zeros([3, 3])
        
        # The analytical expressions for the derivatives are substituted.
        # Limiting case for theta = 180.
        if (np.linalg.norm(theta) > (math.pi - (1 * 10**-6))):
            grad[0][:] = (math.pi - theta) / (2 * np.linalg.norm(r_ij)**2) * r_ij
            grad[2][:] = (math.pi - theta) / (2 * np.linalg.norm(r_kj)**2) * r_kj
            grad[1][:] = ((1 / np.linalg.norm(r_ij)) - (1 / np.linalg.norm(r_kj))) * (math.pi - theta) / (2 * np.linalg.norm(r_ij)**2) * r_ij
        # All other cases.
        else:
            grad[0][:] = ((mp.cot(theta) * r_ij) / np.linalg.norm(r_ij)**2) - (r_kj / (np.linalg.norm(r_ij) * np.linalg.norm(r_kj) * mp.sin(theta)))
            grad[2][:] = ((mp.cot(theta) * r_kj) / np.linalg.norm(r_kj)) - (r_ij / (np.linalg.norm(r_ij) * np.linalg.norm(r_kj) * mp.sin(theta)))
            grad[1][:] = ((r_ij + r_kj) / (np.linalg.norm(r_ij) * np.linalg.norm(r_kj) * mp.sin(theta))) - (((r_ij / np.linalg.norm(r_ij)**2) + (r_kj / np.linalg.norm(r_kj)**2)) * mp.cot(theta))
        
        return grad
    
    # Sub-function which calculates analytical derivative expressions for the dihedral between four atoms using Numpy's linear algebra routines.
    # It takes the arguments of the coordinate array, and the indices which define the four atoms in the coordinate array.
    def dihedral_grad(coords, i, j, k, l):
        # Three vectors of the four coordinates are calculated.
        r_ij = coords[i] - coords[j]
        r_lk = coords[l] - coords[k] 
        r_kj = coords[k] - coords[j]
        
        # Term 1 is written to variable term_1.
        combined_vecs = np.array([r_lk, r_ij, r_kj])
        det = np.linalg.det(combined_vecs)
        term_1 = np.sign(det)
        
        # The terms contained in term 2 are explicitly written.
        e = r_kj / np.linalg.norm(r_kj)
        a1 = r_ij - np.dot(np.dot(r_ij, e), e)
        a2 = r_lk - np.dot(np.dot(r_lk, e), e)
        
        # Numerator and denominator for term 2 are explicitly written as variables for ease of working.
        numerator = np.dot(a1, a2)
        denominator = np.linalg.norm(a1) * np.linalg.norm(a2)
        division = numerator / denominator
        
        # Series of if statements solve the value for term 2.
        # Arccos is used to calculate the angle in radians.
        if (division <= -1):
            term_2 = math.acos(-1)
        elif (division >= 1):
            term_2 = math.acos(1)
        else:
            term_2 = math.acos(division)
        
        # The dihedral angle is calculated.
        chi = term_1 * term_2
        
        # The gradient matrix is initialised.
        grad = np.zeros([4, 3])
        
        # The analytical expressions for the derivatives are substituted.
        # Limiting case for chi = 180.
        if (np.linalg.norm(chi) > (math.pi - (1 * 10**-6))): 
            g = np.cross(r_kj, a1)
            g = g / np.linalg.norm(g)
            grad[0][:] = g / np.linalg.norm(a1)
            grad[3][:] = g / np.linalg.norm(a2)
            A = (r_ij * e) / np.linalg.norm(r_kj)
            B = (r_lk * e) / np.linalg.norm(r_kj)
            grad[2][:] = (-((1 + B) / np.linalg.norm(a2)) + (A / np.linalg.norm(a1))) * g
            grad[1][:] = (-((1 - A) / np.linalg.norm(a1)) - (B / np.linalg.norm(a2))) * g
        # Limiting case for chi = 0.
        if (np.linalg.norm(chi) < (1 * 10**-6)): 
            g = np.cross(r_kj, a1)
            g = g / np.linalg.norm(g)
            grad[0][:] = g / np.linalg.norm(a1)
            grad[3][:] = - g / np.linalg.norm(a2)
            A = (r_ij * e) / np.linalg.norm(r_kj)
            B = (r_lk * e) / np.linalg.norm(r_kj)
            grad[2][:] = (((1 + B) / np.linalg.norm(a2)) - (A / np.linalg.norm(a1))) * g
            grad[1][:] = (-((1 - A) / np.linalg.norm(a1)) + (B / np.linalg.norm(a2))) * g
        # All other cases.
        else:
            grad[0][:] = ((a1 * mp.cot(chi)) / np.linalg.norm(a1)**2) - (a2 / (np.linalg.norm(a1) * np.linalg.norm(a2) * mp.sin(chi)))
            grad[3][:] = ((a2 * mp.cot(chi)) / np.linalg.norm(a2)**2) - (a1 / (np.linalg.norm(a1) * np.linalg.norm(a2) * mp.sin(chi)))
            A = (r_ij * e) / np.linalg.norm(r_kj)
            B = (r_lk * e) / np.linalg.norm(r_kj)
            grad[2][:] = ((((1 + B) * a1) + (A * a2)) / (np.linalg.norm(a1) * np.linalg.norm(a2) * mp.sin(chi))) - ((((1 + B) * a2) / np.linalg.norm(a2)**2) + ((A * a1) / np.linalg.norm(a1)**2)) * mp.cot(chi)
            grad[1][:] = ((((1 - A) * a2) - (B * a1)) / (np.linalg.norm(a1) * np.linalg.norm(a2) * mp.sin(chi))) - ((((1 - A) * a1) / np.linalg.norm(a1)**2) + ((B * a2) / np.linalg.norm(a2)**2)) * mp.cot(chi)
            
        return grad
    
    # The B matrix is filled using a series of for-loops with if statements.    
    for i in range(0, n_prims):
        if prim_coords[i][2] == 0:
            grad = rad_grad(coords, int(prim_coords[i][0]-1), int(prim_coords[i][1]-1))
            k = 2
        elif prim_coords[i][3] == 0:
            grad = angle_grad(coords, int(prim_coords[i][0]-1), int(prim_coords[i][1]-1), int(prim_coords[i][2]-1))
            k = 3
        else:
            grad = dihedral_grad(coords, int(prim_coords[i][0]-1), int(prim_coords[i][1]-1), int(prim_coords[i][2]-1), int(prim_coords[i][3]-1))
            k = 4
        for j in range(0, k):
            position = int(3*(prim_coords[i][j] - 1))
            B_mat[i][position:position+3] = grad[j][:]
   
    np.savetxt("B_mat", B_mat)
    return B_mat


