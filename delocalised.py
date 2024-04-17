# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 13:32:24 2021

@author: Neil McFarlane
"""

"""
Delocalised internal coordinate (dlc) functions for the growing string method.
The functions within this code are called as dlc, i.e.,:
    
    import delocalised as dlc
    
"""

#################################################################################################################################################################################

# Standard library imports.
import math
import os

# Third party imports.
import numpy as np
from numpy import linalg as la
from itertools import combinations
import warnings

# Local imports.
import math_utils as m_utils
import utils as utils
import optimisers as opt

#################################################################################################################################################################################

def prim_coords(coords):
    """
    
    // Function which generates the primitive internal definition and an array of primitive values for a defined structure using cartesian coordinates. //
    // This function uses the total connectivity scheme to define the primitive internals. //
    
    Parameters
    ----------
    coords : Numpy array
        Numpy array containing the xyz coordinates of the stucture.

    Returns
    -------
    connect_array : Numpy array
        Numpy array containing the atom numbers which have been determined to have a primitive internal coordinate.
    prim_array : Numpy array
        Numpy array containing the primitive internal coordinates in the same order as the definition in connect_array.

    """
    
    # The list of atom numbers is initialised so that it can be used in the primitive array.
    atom_list = []
    for i in range(1, len(coords) + 1):
        atom_list.append(i)
    
    # Using itertools' combinations, every possible combination of 2 atoms is generated and added to a Numpy array.
    connect_list = list(combinations(atom_list, 2))
    connect_array = np.array(connect_list)
    
    # The actual values of the primitive coordinates are obtained using a series of for loops and added to prim_array.
    prims = []
    for item in connect_array:
        indices = []
        for atom in item:
            indices.append(atom-1)
        r = math.sqrt((float(coords[indices[0]][0]) - float(coords[indices[1]][0]))**2 + (float(coords[indices[0]][1]) - float(coords[indices[1]][1]))**2 + (float(coords[indices[0]][2]) - float(coords[indices[1]][2]))**2)
        prims.append(r)
    prim_array = np.array(prims)
    
    return connect_array, prim_array
    
def gen_B_matp(coords, connect_array):
    """
    
    // Function which generates the primitive B matrix from a set of cartesian and primitive internal definitions. //
    
    Parameters
    ----------
    coords : Numpy array
        Numpy array containing the xyz coordinates of the stucture.
    connect_array : Numpy array
        Numpy array containing the atom numbers which have been determined to have primitive internal coordinates.

    Returns
    -------
    B_matp : Numpy array
        Numpy array containing the primitive B matrix used to convert between cartesian and primitive internal coordinates.

    """
    
    # The Wilson B matrix is initialised.
    n_atoms = len(coords)
    n_prims = len(connect_array)
    B_matp = np.zeros([n_prims, n_atoms * 3])

    # Iterates over the Wilson B matrix and adds the radius gradient terms where appropriate.
    for i in range(0, n_prims):
        grad = m_utils.rad_grad(coords, int(connect_array[i][0]-1), int(connect_array[i][1]-1))
        k = 2
        for j in range(0, k):
            position = int(3*(connect_array[i][j] - 1))
            B_matp[i][position:position+3] = grad[j][:]
            
    return B_matp

def gen_G_mat(B_matp):
    """
    
    // Function which generates the G matrix from a primitive Wilson B matrix. //
    
    Parameters
    ----------
    B_matp : Numpy array
        Numpy array containing the primitive B matrix used to convert between cartesian and primitive internal coordinates.

    Returns
    -------
    G_mat : Numpy array
        Numpy array containing the G matrix used to obtain the non-redundant subspace.

    """
    
    # The dot product between the Wilson B matrix and its transpose gives the G matrix.
    # In future, if other primitive definitions are used, this could be weighted by and appropriately scaled unit matrix.
    G_mat = np.dot(B_matp, B_matp.T)
   
    # By definition, the G matrix is singular, so it should have a determinant of zero to numerical precision.
    # This criteria is checked in order to mitigate errors down the line.
    check = la.det(G_mat)
    if (round(check, 10) > 0):
        print('WARNING: The G Matrix is not singlular. This can either mean an entirely non-redundant primitive set, or some error.')
    
    return G_mat

def G_mat_diag(G_mat):  
    """
    
    // Function which diagonalises the G matrix and extracts the non-redundant and redundant subspaces. //
    
    Parameters
    ----------
    G_mat : Numpy array
        Numpy array containing the G matrix used to obtain the non-redundant subspace.

    Returns
    -------
    U_mat : Numpy array
        Numpy array containing the U matrix - the delocalised internal coordinate subspace.
    R_mat : Numpy array
        Numpy array containing the R matrix - the redundant subspace.

    """
    
    # The G matrix is inverted and the eigenvalues and eigenvectors are unpacked using Numpy's linear algebra routines.
    evals, evecs = la.eigh(G_mat)

    # The temporary storage of non-redundant (U) and redundant (R) subspaces are initialised.
    temp_U = []
    temp_R = []
    
    # All eigenvalues > 0 belong to the non-redundant subspace (U), and eigenvalues = 0 (to numerical precision) belong to the redundant (R) subspace.
    for i in range(0, len(evals)):
        check = la.norm(round(evals[i], 5))
        if check > 0.0:
            temp_U.append(evecs[i,:])
        else:
            temp_R.append(evecs[i,:])
           
    # The U matrix and R matrix are created from the eigenvectors stored in temp_U and temp_R, respectively.
    if len(temp_R) > 0:
        U_mat = np.stack(temp_U, axis = 0)
        R_mat = np.stack(temp_R, axis = 0)
    else:
        U_mat = np.stack(temp_U, axis = 0)
        R_mat = np.zeros(len(temp_U))

    return U_mat, R_mat

def update_B_mat(B_matp, U_mat):
    """
    
    // Function which updates the primitive B matrix into the dlc subspace. //
    
    Parameters
    ----------
    U_mat : Numpy array
        Numpy array containing the U matrix - the delocalised internal coordinate subspace.
    B_matp : Numpy array
        Numpy array containing the primitive B matrix used to convert between cartesian and primitive internal coordinates.
        
    Returns
    -------
    B_mat : Numpy array
        Numpy array containing the B matrix used to convert between cartesian and dlc.

    """
    
    # The B matrix is updated into the delocalised internal coordinate subspace.
    B_mat = np.dot(U_mat, B_matp)

    return B_mat   

def project_constraint(U_mat, constraint):
    """
    
    // Function which projects the constraint vector into dlc subspace and Gram-Schmidt orthonormalises it with the dlc to produce a subspace containing one constraint vector. //
    
    Parameters
    ----------
    U_mat : Numpy array
        Numpy array containing the U matrix - the delocalised internal coordinate subspace.
    constraint : Numpy array
        Numpy array containing the tangent vector.

    Returns
    -------
    V_k : Numpy array
        Numpy array containing the dlc subspace with a constraint vector.
        
    """  

    # The constraint vector is projected onto dlc subspace.
    proj_cons = np.sum( m_utils.proj(constraint,v) for v in U_mat )
    
    # The constraint vector is normalised.
    proj_cons = m_utils.unit_vec(proj_cons)

    # The first vector in V_mat is the constraint for the purposes of orthogonalisation.
    V_mat = np.vstack([[proj_cons], U_mat])
    
    # V_mat is Gram-Schmidt orthogonalised.
    V_k = m_utils.gramSchmidt(V_mat)

    # By definition, the first row in the V_k should be the projected constraint vector, and the vector set should be orthogonal.
    # These criteria are checked in order to minimise errors down the line.
    check = V_k[0] - proj_cons
    check = np.round(check, 10)
    if check.all() != 0.0:
        raise RuntimeError("The constraint vector is not found in the Gram-Schmidt orthogonalised vector space.")
    orthogonality = m_utils._is_orthog(V_k)
    if orthogonality is False:
        raise RuntimeError("The Gram-Schmidt orthogonalised set is not orthogonal - bit of an oxymoron...")

    return V_k
       
def gen_dlc(coords, constraint = None):
    """
    
    // Function which combines the coordinate conversion functions into one for ease of generation. //
    
    Parameters
    ----------
    coords : Numpy array
        Numpy array containing the xyz coordinates of the stucture.
    constraint : Numpy array
        Numpy array containing the primitive internal coordinate constraint which is to be projected onto dlc subspace and orthonormalised with the remaining subspace.
        
    Returns
    -------
    connect_array : Numpy array
        Numpy array containing the atom numbers which have been determined to have primitive internal coordinates.
    prim_array : Numpy array
        Numpy array containing the primitive internal coordinates in the same order as the definition in connect_array.
    B_matp : Numpy array
        Numpy array containing the primitive B matrix used to convert between cartesian and primitive internal coordinates.
    G_mat : Numpy array
        Numpy array containing the G matrix used to obtain the non-redundant subspace.
    U_mat : Numpy array
        Numpy array containing the U matrix - the delocalised internal coordinate subspace.
    B_mat : Numpy array
        Numpy array containing the B matrix used to convert between cartesian and internal coordinates which is updated as optimisation proceeds.
    V_k : Numpy array
        Numpy array containing the V_k matrix - the delocalised internal coordinate subspace when a constraint has been imposed.
    S : Numpy array
        Numpy array containing the actual dlc.

    """
    
    # The series of functions which creates the requirements for generating the dlc.
    connect_array, prim_array = prim_coords(coords)
    B_matp = gen_B_matp(coords, connect_array)
    G_mat = gen_G_mat(B_matp)
    U_mat, R_mat = G_mat_diag(G_mat)

    # If a constraint is used in generating the dlc, then it is projected and orthonormalised.
    # The Wilson B matrix is also updated into the constrained/unconstrained subspace.
    if constraint is not None:
        V_k = project_constraint(U_mat, constraint)
        B_mat = update_B_mat(B_matp, V_k)
        
        # The dlc are generated.   
        S = []
        for i in range(len(V_k)):
            temp_S = 0
            for j in range(len(prim_array)):
                temp_S += V_k[i][j] * prim_array[j]
            S.append(temp_S)
        S = np.array(S)
        
        return connect_array, prim_array, B_matp, G_mat, U_mat, B_mat, V_k, S
    else:
        B_mat = update_B_mat(B_matp, U_mat)

        # The dlc are generated.   
        S = []

        for i in range(len(U_mat)):
            temp_S = 0
            for j in range(len(prim_array)):
                temp_S += U_mat[i][j] * prim_array[j]
            S.append(temp_S)
        S = np.array(S)

        return connect_array, prim_array, B_matp, G_mat, U_mat, B_mat, S

def cartgrad_to_primgrad(g_array, B_matp):
    """
    
    // Function which updates the cartesian gradient to primitive internal coordinate subspace. //
    
    Parameters
    ----------
    g_array : Numpy array
        Numpy array containing all the interations with their relevant gradient terms.
    B_matp : Numpy array
        Numpy array containing the B matrix used to convert between cartesian and primitive internal coordinates.
        
    Returns
    -------
    g_array : Numpy array
        Numpy array containing only gradient terms in primitive internal coordinate subspace.

    """
    
    # Using single value decomposition, the Moore-Penrose inverse is constructed and used to convert the gradient array.
    BT_Ginv = np.dot(m_utils.svd_inv(np.dot(B_matp, B_matp.T)), B_matp)
    g_array = np.dot(g_array[:,2], BT_Ginv)
    
    return g_array

def cartgrad_to_dlcgrad(g_array, B_mat):
    """
    
    // Function which updates the cartesian gradient to dlc subspace. //
    
    Parameters
    ----------
    g_array : Numpy array
        Numpy array containing all the interations with their relevant gradient terms.
    B_mat : Numpy array
        Numpy array containing the B matrix used to convert between cartesian and dlc.
        
    Returns
    -------
    g_array : Numpy array
        Numpy array containing only gradient terms in dlc subspace.

    """
    
    # Using single value decomposition, the Moore-Penrose inverse is constructed and used to convert the gradient array.
    BT_Ginv = np.dot(B_mat.T, m_utils.svd_inv(np.dot(B_mat, B_mat.T)))
    g_array = np.dot(g_array[:,2], BT_Ginv)
    
    return g_array

def approx_hess_calc(U_mat_V_k, connect_array):
    """
    
    // Function which calculates the approximate hessian of a defined structure using primitive internal coordinates. //
    
    Parameters
    ----------
    U_mat_V_k : Numpy array
        Numpy array containing either the U matrix or V_k matrix depending on whether it is constrained or not - the delocalised internal coordinate subspace.
    connect_array : Numpy array
        Numpy array containing the atom numbers which have been determined to have a primitive internal coordinate.
        
    Returns
    -------
    hess : Numpy array
        Numpy array containing the dlc hessian matrix for the LJ structure.
    prim_hess : Numpy array
        Numpy array containing the primitive hessian matrix for the LJ structure.

    """   
    
    # The hessian is approximated by using a primitive coordinate hessian matrix.
    # Currently, all primitive internals are stretching terms, so a unit matrix with size of the number of primitive coordinates is an appropriate choice as the primitive coordinate hessian.
    prim_hess = np.identity(len(connect_array))
    
    # To generate the approximate hessian in delocalised subspace, we multiply this primitive coordinate hessian by the transpose and non-transpose of U.
    # The hessian is also rounded to mitigate errors down the line.
    hess = np.dot(U_mat_V_k, np.dot(prim_hess, U_mat_V_k.T))
    hess = np.round(hess, decimals = 4)
    
    # The hessian is a symmetric matrix by definition.
    # This criteria is checked in order to mitigate errors down the line.
    symmetry = (hess.transpose() == hess).all()
    if symmetry == False:
        raise RuntimeError("Hessian is not symmetric.")

    return hess, prim_hess

def primhess_to_dlchess(prim_hess, U_mat_V_k):
    """
    
    // Function which updates a primitive hessian into dlc subspace. //
    
    Parameters
    ----------
    U_mat_V_k : Numpy array
        Numpy array containing either the U matrix or V_k matrix depending on whether it is constrained or not - the delocalised internal coordinate subspace.
    prim_hess : Numpy array
        Numpy array containing the primitive hessian matrix for the LJ structure.
        
    Returns
    -------
    hess : Numpy array
        Numpy array containing the dlc hessian matrix for the LJ structure.

    """  

    # To update the primitive hessian to delocalised subspace, we multiply this primitive coordinate hessian by the transpose and non-transpose of V_k.
    # The hessian is also rounded to mitigate errors down the line.
    hess = np.dot(U_mat_V_k, np.dot(prim_hess, U_mat_V_k.T))
    hess = np.round(hess, decimals = 4)
    
    # The hessian is a symmetric matrix by definition.
    # This criteria is checked in order to mitigate errors down the line.
    symmetry = (hess.transpose() == hess).all()
    if symmetry == False:
        raise RuntimeError("Hessian is not symmetric.")

    return hess

def prim_to_cartesian(dq, prims_1, xyz_1, B_matp):
    """
    
    // Function which calculates a new set of cartesian coordinates from a displacement in primitive internal coordinates. //
    
    Parameters
    ----------
    dq : Numpy array
        Numpy array containing the change in primitive internal coordinates.
    prims_1 : Numpy array
        Numpy array containing the primitive internal coordinates of the starting point.
    xyz_1 : Numpy array
        Numpy array containing the xyz coordinates of the starting point.
    B_matp : Numpy array
        Numpy array containing the primitive B matrix used to convert between cartesian and primitive internal coordinates.

    Returns
    -------
    new_xyz : Numpy array
        Numpy array containing the new xyz coordinates.
        
    """  
    
    # Since cartesians are rectilinear and internal coordinates are curvilinear, a simple transformation cannot be used.
    # Instead, an iterative transformation procedure must be used.
    # The expression B(transpose) * G(inverse) is initialised as it is used to convert between coordinate systems.
    BT_Ginv = np.dot(m_utils.svd_inv(np.dot(B_matp, B_matp.T)), B_matp)
    
    # Initialising some values for convergence criteria.
    xyz_rms_1 = 0
    iter_counter = 0
    init_dq = dq.copy()
    target_prims = prims_1.copy() + dq.copy()

    while True:
        # The change in cartesian coordinates, its root-mean-square change is calculated, and the new cartesian geometry is found.
        dxyz = np.dot(BT_Ginv.T, dq)
        xyz_rms_2 = np.sqrt(np.mean(dxyz ** 2))
        xyz_2 = xyz_1 + dxyz.reshape(-1, 3)
        
        # The new primitive coordinates and the new B matrix for the next iteration is evaluated.
        _, prims_2, B_matp, _, _, _, _ = gen_dlc(xyz_2)
        
        # The transformation term for the next iteration is evaluated, but this is generally not necessary for typical situations.
        # Foramlly, this term is necessary, and accelerates convergence. So, for a simple system, it is worth evaluating.
        BT_Ginv = np.dot(m_utils.svd_inv(np.dot(B_matp, B_matp.T)), B_matp)
        
        # The change in primitive internals for the next iteration is evaluated, and any which do not change in the original change in primitive internals is set to zero.
        dq_n = prims_2 - target_prims
        for i in range(len(dq_n)):
            if init_dq[i] == 0.:
                dq_n[i] = 0

        # The first ending condition for this transformation is when the root-mean-square change in cartesians is less than 10^-6.
        # The second ending condition for this transformation is when the difference in root-mean-square change in cartesians between iteration i and i+1 is less than 10^-12.
        # The third ending condition for this transformation is when the difference between the target primitive internals and the calculated primitive internals is less than 10^-6.
        check = target_prims - prims_2
        if abs(xyz_rms_2) < 1 * 10**-6:
            break
        elif abs(xyz_rms_2 - xyz_rms_1) < 1 * 10**-12:
            break
        if (abs(check) < 1 * 10 **-12).any():
            break
            
        # Properties which are calculated from the iterative procedure are updated to be ith property for the next iteration.
        prims_1 = prims_2.copy()
        dq = dq_n.copy()
        xyz_1 = xyz_2.copy()
        xyz_rms_1 = xyz_rms_2
        
        # Exit condition for when cartesians cannot be solved.
        iter_counter += 1
        if iter_counter == 1000:
            raise RuntimeError("Cartesian coordinates could not be solved from the change in internal coordinates.")
    
    new_xyz = xyz_1.copy()
    return new_xyz

def dlc_to_cartesian(dS, S_1, xyz_1, B_mat, constraint = None):
    """
    
    // Function which calculates a new set of cartesian coordinates from a displacement in dlc. //
    
    Parameters
    ----------
    dS : Numpy array
        Numpy array containing the change in dlc.
    S_1 : Numpy array
        Numpy array containing the dlc of the starting point.
    xyz_1 : Numpy array
        Numpy array containing the xyz coordinates of the starting point.
    B_mat : Numpy array
        Numpy array containing the B matrix used to convert between cartesian and dlc.
    constraint : Numpy array
        Numpy array containing the constraint, if used. This is important in correct generation of the dlc.

    Returns
    -------
    new_xyz : Numpy array
        Numpy array containing the new xyz coordinates.
        
    """  
    
    # Since cartesians are rectilinear and internal coordinates are curvilinear, a simple transformation cannot be used.
    # Instead, an iterative transformation procedure must be used.
    # The expression B(transpose) * G(inverse) is initialised as it is used to convert between coordinate systems.
    BT_Ginv = np.dot(m_utils.svd_inv(np.dot(B_mat, B_mat.T)), B_mat)

    # Initialising some values for convergence criteria.
    xyz_rms_1 = 0
    iter_counter = 0
    init_dS = dS.copy()
    damp = 1.0
    ndqt = 100.0
    target_S = S_1.copy() + dS.copy()
    print(target_S)
    while True:
        # The change in cartesian coordinates, its root-mean-square change is calculated, and the new cartesian geometry is found.
        dxyz = np.dot(BT_Ginv.T, dS) * damp
        xyz_rms_2 = np.sqrt(np.mean(dxyz ** 2))
        xyz_2 = xyz_1 + dxyz.reshape(-1, 3)
        
        # The new dlc as well as the B matrix for the next iteration is evaluated.
        if constraint is not None:
            _, _, _, _, _, B_mat, _, S_2 = gen_dlc(xyz_2, constraint = constraint)
        else:
            _, _, _, _, _, B_mat, S_2 = gen_dlc(xyz_2)
        
        # The transformation term for the next iteration is evaluated, but this is generally not necessary for typical situations.
        # Foramlly, this term is necessary, and accelerates convergence. So, for a simple system, it is worth evaluating.
        BT_Ginv = np.dot(m_utils.svd_inv(np.dot(B_mat, B_mat.T)), B_mat)
        
        # The change in dlc for the next iteration is evaluated, and any which do not change in the original change in dlc is set to zero.
        dS_actual = (S_2 - S_1)
        #for i in range(len(dS_n)):
            #if init_dS[i] == 0.:
                #dS_n[i] = 0
        ndq = la.norm(dS_actual)
        if ndq > ndqt:
            damp /= 2
        print(S_2)
        # The first ending condition for this transformation is when the root-mean-square change in cartesians is less than 10^-6.
        # The second ending condition for this transformation is when the difference in root-mean-square change in cartesians between iteration i and i+1 is less than 10^-12.
        # The third ending condition for this transformation is when the difference between the target dlc and the calculated dlc is less than 10^-6.
        check = target_S - S_2
        if abs(xyz_rms_2) < 1 * 10**-6:
            break
        elif abs(xyz_rms_2 - xyz_rms_1) < 1 * 10**-12:
            break
        elif (abs(check) < 1 * 10 **-12).any():
            break
        
        # Properties which are calculated from the iterative procedure are updated to be ith property for the next iteration.
        S_1 = S_2.copy()
        dS = dS - dS_actual
        xyz_1 = xyz_2.copy()
        xyz_rms_1 = xyz_rms_2
        ndqt = ndq

        # Exit condition for when cartesians cannot be solved.
        iter_counter += 1 
        if iter_counter == 500:
            raise RuntimeError("Cartesian coordinates could not be solved from the change in internal coordinates.")

    new_xyz = xyz_1.copy()
    print(new_xyz)
    return new_xyz


     