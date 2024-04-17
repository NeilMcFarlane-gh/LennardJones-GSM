# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:04:02 2021

@author: Neil McFarlane
"""

"""
Optimiser functions for the growing string method.
The functions within this code are called as opt, i.e.,:
    
    import optimisers as opt
    
"""

#################################################################################################################################################################################

# Standard library imports.

# Third party imports.
import numpy as np
from numpy import linalg as la

# Local imports.
import delocalised as dlc
import math_utils as m_utils

#################################################################################################################################################################################

def quasi_newton(xyz_1, opt_steps, prim_tangent = None):
    """
    
    // Function which optimises a node using quasi newton methods where the hessian is updated in primitive internal coordiantes using the BFGS scheme. //
    
    Parameters
    ----------
    xyz_1 : Numpy array
        Numpy array containing the xyz coordinates of the starting point.
    opt_steps : integer
        The number of optimisation cycles taken for a given starting structure.
    prim_tangent : Numpy array
        Numpy array containing the primitive internal coordinate constraint constraint, if used.

    Returns
    -------
    new_xyz : Numpy array
        Numpy array containing the newly optimised xyz coordinates.
        
    """  
    
    # The dlc subspace is generated depending on whether the space is constrained or not.
    # The initial approximate hessian matrix is evaluated depending on whether the space is constrained or not.
    # Only the primitive hessian will be updated as optimisation proceeds, as this saves needless transformation to dlc at every iteration.
    if prim_tangent is None:
        connect_array_1, prims_1, B_matp_1, _, U_mat_1, B_mat_1, S_1 = dlc.gen_dlc(xyz_1)
        hess_1, prim_hess_1 = dlc.approx_hess_calc(U_mat_1, connect_array_1)
    elif prim_tangent is not None:
        connect_array_1, prims_1, B_matp_1, _, _, B_mat_1, V_k_1, S_1 = dlc.gen_dlc(xyz_1, prim_tangent)
        hess_1, prim_hess_1 = dlc.approx_hess_calc(V_k_1, connect_array_1)

    # The hessian is not updated on the first iteration.
    update_hess = False
    for i in range(opt_steps):
        # When update_hess is True, then the hessian is updated from primitive to dlc subspace.
        if update_hess == True:
            if prim_tangent is None:
                hess_1 = dlc.primhess_to_dlchess(prim_hess_1, U_mat_1)
            elif prim_tangent is not None:
                hess_1 = dlc.primhess_to_dlchess(prim_hess_1, V_k_1)
        
        # The gradient is obtained andupdated to primitive and dlc subspace.
        g_cart_1 = m_utils.LJ_grad(xyz_1, 1, 1)
        g_dlc_1 = dlc.cartgrad_to_dlcgrad(g_cart_1, B_mat_1)
        g_prim_1 = dlc.cartgrad_to_primgrad(g_cart_1, B_matp_1)

        # The change in dlc is evaluated using hessian and gradient information and then normalised.
        dS = -1 * np.dot(la.inv(hess_1), g_dlc_1)
        dS = m_utils.unit_vec(dS)
        
        # The next set of cartesian coordinates are evaluated using a backtrack linesearch method.
        xyz_2 = backtrack_linesearch(g_dlc_1, dS, S_1, xyz_1)
        
        # The new primitive internal coordinates, and hence the primitive Wilson B matrix are calculated.
        connect_array_2, prims_2 = dlc.prim_coords(xyz_2)
        B_matp_2 = dlc.gen_B_matp(xyz_2, connect_array_2)
        
        # The new cartesian gradient is calculated and converted to primitive internal coordinates.
        g_cart_2 = m_utils.LJ_grad(xyz_2, 1, 1)
        g_prim_2 = dlc.cartgrad_to_primgrad(g_cart_2, B_matp_2)
        
        # The primitive hessian is updated using the BFGS scheme.
        dg = g_prim_2 - g_prim_1
        dq = prims_2 - prims_1
        prim_hess_2 = update_bfgsp(prim_hess_1, dq, dg)
        update_hess = True

        # Properties which are calculated from the iterative procedure are updated to be ith property for the next iteration.
        prim_hess_1 = prim_hess_2.copy()
        xyz_1 = xyz_2.copy()
        if prim_tangent is None:
            connect_array_1, prims_1, B_matp_1, _, U_mat_1, B_mat_1, S_1 = dlc.gen_dlc(xyz_1)
        elif prim_tangent is not None:
            connect_array_1, prims_1, B_matp_1, _, _, B_mat_1, V_k_1, S_1 = dlc.gen_dlc(xyz_1, prim_tangent)
         
    new_xyz = xyz_1.copy()
    return new_xyz
        
def update_bfgsp(prim_hess, dq, dg):
    """
    
    // Function which updates a given primitive internal coordinate hessian matrix using the BFGS scheme. //
    
    Parameters
    ----------
    prim_hess : Numpy array
        Numpy array containing the primitive hessian matrix for the LJ structure for the previous step of optimsation.
    dq : Numpy array
        Numpy array containing the change in primitive internal coordinates between the present and previous step of optimisation.
    dg : Numpy array
        Numpy array containing the change in primitive internal coordinate gradient between the present and previous step of optimisation.

    Returns
    -------
    new_hess : Numpy array
        Numpy array containing the primitive hessian matrix for the LJ structure for the present step of optimisation.
        
    """  

    # All terms for the hessian update are initialised.
    Hdq = np.dot(prim_hess, dq)
    dqHdq = np.dot(dq.T, Hdq)
    dgdg = np.outer(dg, dg)
    dgtdq = np.dot(dg.T, dq)
    change = np.zeros_like(prim_hess)

    # Based on the magnitude of certain values, the hessian update is calculated differently such that the values are not vanishingly small.
    if dgtdq > 0.:
        if dgtdq < 0.001: dgtdq = 0.001
        change += dgdg / dgtdq
    if dqHdq > 0.:
        if dqHdq < 0.001: dqHdq = 0.001
        change -= np.outer(Hdq, Hdq) / dqHdq   
        
    # The new hessian is evaluated.
    new_hess = prim_hess + change

    return new_hess

def backtrack_linesearch(g, dS, S, xyz):
    """
    
    // Function which performs a backtrack linesearch algorithm for a given step of optimisation. //
    
    Parameters
    ----------
    g : Numpy array
        Numpy array containing the dlc gradient for the starting point.
    dS : Numpy array
        Numpy array containing the change in dlc as predicted by the optimisation algorithm chosen.
    S : Numpy array
        Numpy array containing the dlc of the starting point.
    xyz : Numpy array
        Numpy array containing the xyz coordinates of the starting point.
        
    Returns
    -------
    xyz : Numpy array
        Numpy array containing the xyz coordinates of the final structure found by the line search.
        
    """  
    
    # Constants for the Armijo and Wolfe conditions are initialised.
    dec = 0.5
    inc = 2.1
    wolfe = 0.9
    
    print(S)
    print(dS)
    import sys; sys.exit()
    
    # Some ending coniditons and initial, minimum, and maximum steps are initialised.
    count = 0
    ftol = 1 * 10**-4
    max_linesearch = 1000
    step = 0.1
    min_step = 0.0001
    max_step = 0.5

    # The initial gradient in the search direction is evaluated.
    dg_init = np.dot(g.T, dS)
    dg_test = ftol * dg_init

    # The initial energy is calculated as it is used in evaluation of the Armijo condition.
    E_init = m_utils.LJ_potential(xyz, 1, 1)

    # The Wilson B matrix is calculated, and cartesians are copied.
    _, _, _, _, _, B_mat, _ = dlc.gen_dlc(xyz)
    xyz_1 = xyz.copy()
    
    while True:
        # A step is taken along the range of dS and the new cartesian and dlc are calculated.
        S_n = S
        S_n = S_n + dS * step
        xyz = dlc.dlc_to_cartesian(S_n - S, S, xyz_1, B_mat, constraint = None)
        _, _, _, _, _, _, S = dlc.gen_dlc(xyz)
        
        # The energy and gradient at the new xyz geometry are evaluated.
        fx = m_utils.LJ_potential(xyz, 1, 1)
        gx = m_utils.LJ_grad(xyz, 1, 1)
        g = dlc.cartgrad_to_dlcgrad(gx, B_mat)
        
        # The count is updated and the width is reset.
        width = 1.
        count = count + 1

        # To ensure that the step is not too large, the Armijo condition is checked.
        if fx > E_init + (step * dg_test):
            width = dec
        else:
            # To ensure that the step is not too small, Wolfe conditions are checked.
            # First, check the normal Wolfe condition.
            dg = np.dot(g.T, dS)
            if dg < wolfe * dg_init:
                width = inc
            else:
                # Now, check the strong wolfe condition.
                if dg > -wolfe * dg_init:
                    width = dec
                else:
                    return xyz
        if max_linesearch <= count:
            return xyz
        if step <= min_step and width <= 1.:
            return xyz
        if step >= max_step and width >= 1.:
            return xyz

        # The step is updated and it is ensured that it is not too large.
        step = step * width
        if step > max_step:
            step = max_step