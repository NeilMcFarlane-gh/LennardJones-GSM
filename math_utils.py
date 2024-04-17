# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 13:24:51 2021

@author: Neil McFarlane
"""

"""
Math utilities for the growing string method.
The functions within this code are called as m_utils, i.e.,:
    
    import math_utils as m_utils
    
"""

#################################################################################################################################################################################

# Standard library imports.
import math

# Third party imports.
import numpy as np
from numpy import linalg as la
from itertools import combinations

# Local imports.

#################################################################################################################################################################################

def rad_grad(coords, i, j):
    """
    
    // Function which calculates the gradient terms of the bond distance term between two atoms for the Wilson B matrix. //
    
    Parameters
    ----------
    coords : Numpy array
        Numpy array containing the xyz coordinates of the stucture.
    i : integer
        Identifier for the first of the two atoms.
    j : integer
        Identifier for the second of the two atoms.
        
    Returns
    -------
    grad : Numpy array
        Numpy array containing the gradient terms for the Wilson B matrix.

    """
    
    # A vector of the two coordinates is calculated and made to be real numbers.
    r_ij = (coords[j].real - coords[i].real)
  
    # The gradient matrix is initialised.
    grad = np.zeros([2, 3])
        
    # The analytical expressions for the derivatives are substituted.
    grad[0][:] = r_ij / la.norm(r_ij)
    grad[1][:] = -r_ij / la.norm(r_ij)

    return grad

def LJ_potential(coords, epsilon, sigma):
    """
    
    // Function which calculates the LJ potential of a defined structure expressed in cartesian coordinates. //
    
    Parameters
    ----------
    coords : Numpy array
        Numpy array containing the xyz coordinates of the stucture.
    epsilon : float
        Lennard-Jones parameter.
    sigma : float
        Lennard-Jones parameter.
        
    Returns
    -------
    E_LJ : float
        Gives the LJ potential.

    """
    
    # The total LJ potential is reset/initialised.
    E_LJ = 0 # eV.
    
    # The list of atom numbers is initialised so that all interations can be obtained.
    atom_indices = []
    for i in range(0, len(coords)):
        atom_indices.append(i)
    
    # Using itertools' combinations, every possible combination of 2 atoms is generated and added to a Numpy array.
    # For larger systems, a covalent radius cut-off could be introduced to limit computations.
    connect_list = list(combinations(atom_indices, 2))
    connect_array = np.array(connect_list)

    # For every combination defined in connect_array, the LJ potential is calculated and added to the total LJ potential.
    for (j,i) in enumerate(connect_array):
        r = math.sqrt((float(coords[i[0]][0]) - float(coords[i[1]][0]))**2 + (float(coords[i[0]][1]) - float(coords[i[1]][1]))**2 + (float(coords[i[0]][2]) - float(coords[i[1]][2]))**2)
        E = 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)
        E_LJ = E_LJ + E
           
    return E_LJ

def LJ_grad(coords, epsilon, sigma):
    """
    
    // Function which calculates the LJ gradient of a defined structure expressed in cartesian coordinates. //
    
    Parameters
    ----------
    coords : Numpy array
        Numpy array containing the xyz coordinates of the stucture.
    epsilon : float
        Lennard-Jones parameter.
    sigma : float
        Lennard-Jones parameter.
        
    Returns
    -------
    g_array : Numpy array
        Numpy array containing all the interations with their relevant gradient terms.

    """
    
    
    # The list of atom numbers is initialised so that all interations can be obtained.
    atom_indices = []
    for i in range(0, len(coords)):
        atom_indices.append(i)
    
    # Using itertools' combinations, every possible combination of 2 atoms is generated and added to a Numpy array.
    # For larger systems, a covalent radius cut-off could be introduced to limit computations.
    connect_list = list(combinations(atom_indices, 2))
    connect_array = np.array(connect_list)
    
    # The array of all the force contributions is initialised.
    # The first two columns show the atom indices which correspond to the gradient, and the third column gives the gradient term.
    g_array = connect_array.copy() + 1
    to_add = np.zeros((len(connect_array),1))
    g_array = np.hstack([g_array, to_add])

    # For every combination defined in connect_array, the LJ gradient is calculated.
    for (j,i) in enumerate(connect_array):
        r = math.sqrt((float(coords[i[0]][0]) - float(coords[i[1]][0]))**2 + (float(coords[i[0]][1]) - float(coords[i[1]][1]))**2 + (float(coords[i[0]][2]) - float(coords[i[1]][2]))**2)
        g = (-24 * epsilon) * (2 * (sigma**12 / r**13) - (sigma**6 / r**7))
        g_array[j][2] = g
            
    return g_array

def hess_eigen_calc(hess):
    """
    
    // Function which calculates the eigenvalues and eigenvectors of a hessian matrix. //
    
    Parameters
    ----------
    hess : Numpy array
        Numpy array containing the hessian matrix for the LJ structure.

    Returns
    -------
    evecs : array
        Numpy array containing the eigenvectors for the hessian matrix.
    evals: array
        Numpy array containing the eigenvalues for the hessian matrix.
    

    """
    
    # The eigenvalues and eigenvectors are unpacked from hess using Numpy's linear albegra routines.
    evals, evecs = la.eig(hess)
    
    # As the hessian is a symmetric matrix, the eigenvalues are always real, so the imaginary part is zero.
    # This criteria is checked in order to mitigate errors down the line.
    evals_real = evals.real
    for i in range(0, len(evals_real)):
        if evals[i] != evals_real[i]:
            raise RuntimeError("Eigenvalues are not real.")
    evals = evals.real
      
    # Again, as the hessian is symmetric, the eigenvectors should be orthogonal.
    # This criteria is checked in order to mitigate errors down the line.
    orthogonality = _is_orthog(evecs)
    if orthogonality is False:
        raise RuntimeError("Eigenvectors are not orthogonal.")
    
    return evecs, evals

def _is_orthog(vectors):
    """
    
    // Function which checks if a vector set is orthogonal to eachother. //
    
    Parameters
    ----------
    vectors : Numpy array
        Numpy array containing any series of vectors.

    Returns
    -------
    orthog : Boolean logic
        Logic which describes the vectors are orthogonal (True) or not (False).
    

    """
    
    # Variables are initialised.
    orthog = True
    vectors = np.atleast_2d(vectors)
    
    # Takes the dot product of each vector pair, and checks if the result is close to zero.
    for vector in vectors:
        for vector2 in vectors:
            # If the two vectors are equal, then there is no need to perform the dot product.
            if np.array_equal(vector, vector2):
                continue
            # The dot product always has some numerical precision remainder so 1e-5 is used.
            if abs(np.dot(vector, vector2)) > 1e-5:
                orthog = False
                
    return orthog

def proj(x, u):
    """
    
    // Function which projects one vector onto another. //
    
    Parameters
    ----------
    x : Numpy array
        Numpy array containing the vector to be projected.
    u : Numpy array
        Numpy array containing the vector which vector x is projected upon.

    Returns
    -------
    proj : Numpy array
        Numpy array containing the vector which consists of vector x projected onto u.
    
    """

    # The projection space is normalised.
    u = unit_vec(u)
    
    # The vector is projected by the usual formula.
    proj = np.inner(x, u) * u

    return proj

def unit_vec(x):
    """
    
    // Function which obtains the unit vector of an input vector. //
    
    Parameters
    ----------
    x : Numpy array
        Numpy array containing the vector to be transformed to a unit vector.

    Returns
    -------
    unit : Numpy array
        Numpy array containing the unit vector of x.
    
    """
    
    # The vector is transformed by the usual formula.
    unit = x / la.norm(x)
    
    return unit

def gramSchmidt(vectors):
    """
    
    // Function which orthonormalises a set of vectors by the Gram-Schmidt methodology. //
    
    Parameters
    ----------
    vectors : Numpy array
        Numpy array containing the vector subspace which is to be orthonormalised.

    Returns
    -------
    basis : Numpy array
        Numpy array containing the orthonormalised set of vectors with the last vector taken removed - it drops out due to negligibly small values.
    
    """
    
    # The input is ensured to be a 2d array, or if not it can be treated like one.
    vectors = np.atleast_2d(vectors)

    # The recursion end conditions are initialised.
    if len(vectors) == 0:
        return []
    if len(vectors) == 1:
        return unit_vec(vectors)
    
    # The second vector is taken as u.
    u = vectors[-1]

    # The rest of the vector space is orthonormalised.
    basis = gramSchmidt(vectors[0:-1])

    # The orthonormalised vector is calculated and is appended to basis if it meets the below criteria.
    w = np.atleast_2d(u - np.sum( proj(u,v)  for v in basis ))
    if (la.norm(w) > 1 * 10**-3) and (abs(w) > 1 * 10**-6).any():
        basis = np.append(basis, unit_vec(w), axis = 0)
    else:
        pass

    return basis

def svd_inv(array, hermitian = False):
    """
    
    // Function which obtains the generalised inverse an array by single value decomposition. //
    
    Parameters
    ----------
    array : Numpy array
        Numpy array containing the array which is to be inverted.
    hermitian : Boolean logic
        If True, array is assumed to be Hermitian (symmetric if real-valued), enabling a more efficient method for finding singular values.

    Returns
    -------
    array_inv : Numpy array
        Numpy array containing the inverted array/
    
    """
    
    # The threshold value for which eigenvalues which are taken as zero is defined.
    thresh = 1 * 10**-6
    
    # The array is unpacked using singular value decomposition.
    U, S, Vt = la.svd(array, hermitian = hermitian)
    
    # Vectors which have eigenvalues greater than the threshold are added to keep.
    keep = S > thresh
    
    # The array S_inv is calculated by using values in keep, and if a value is not in keep then it is zero in S_inv.
    S_inv = np.zeros_like(S)
    S_inv[keep] = 1 / S[keep]
    
    # The inverted array where only nonzero eigenvalues are inverted is obtained.
    array_inv = Vt.T.dot(np.diag(S_inv)).dot(U.T)
    
    return array_inv