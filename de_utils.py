# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:37:24 2024

@author: Neil McFarlane
"""

"""
Functions for the double-ended variant of the growing string method.
The functions within this code are called as DE, i.e.,:
    
    import de_utils as DE
    
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
import delocalised as dlc

#################################################################################################################################################################################

def tangent(i_prims, j_prims):
    """
    
    // Function which calculates the coordinate tangent vector between nodes i and j for DE-GSM using primitive internal coordinates. //
    
    Parameters
    ----------
    i_prims : Numpy array
        Numpy array containing the primitive internal coordinates of node i.
    j_prims : Numpy array
        Numpy array containing the primitive internal coordinates of node j.

    Returns
    -------
    internal_tan : Numpy array
        Numpy array containing the tangent vector between nodes i and j.
        
    """  
        
    # To calculate the reaction tangent between nodes i and j, the difference between in primitive coordinates between the reactant and product is employed.
    prim_diff = np.zeros(len(j_prims))
    for i in range(0, len(j_prims)):
        prim_diff[i] = j_prims[i] - i_prims[i]
    
    return prim_diff

def add_nodes(i_coords, j_coords, nodes, current_nodes, new_nodes):
    """
    
    // Function which add either 1 or 2 nodes to the string depending on the evolution of the string. //
    
    Parameters
    ----------
    i_coords : Numpy array
        Numpy array containing the xyz coordinates of node i.
    j_coords : Numpy array
        Numpy array containing the xyz coordinates of node j.
    nodes : integer
        The total number of nodes which the user has selected for optimisation.
    current_nodes : integer
        Number of nodes currently on the string.
    new_nodes : integer
        Number of new nodes to be added to the string.

    Returns
    -------
    r_coords : Numpy array
        Numpy array containing the xyz coordinates of the new reactant side frontier node.
    p_coords : Numpy array
        Numpy array containing the xyz coordinates of the new product side frontier node (if calculated - new_nodes must equal 2).
        
    """ 

    # If new_nodes = 1, then only one node is added to the reactant side, otherwise 2 are added - one to reactant side, and one to product side.
    for i in range(0, new_nodes):
        if i == 0:
            # New node is added.
            r_coords = add_node(i_coords, j_coords, current_nodes, nodes, ID = 'R')
            
        elif i == 1:
            # New node is added.
            p_coords = add_node(i_coords, j_coords, current_nodes, nodes, ID = 'P')
    
    if new_nodes == 2:
        return r_coords, p_coords
    elif new_nodes == 1:
        return r_coords

def add_node(i_coords, j_coords, current_nodes, nodes, ID = None):
    """
    
    // Function which adds a new node to either the product or reactant side. //
    
    Parameters
    ----------
    i_coords : Numpy array
        Numpy array containing the xyz coordinates of node i.
    j_coords : Numpy array
        Numpy array containing the xyz coordinates of node j.
    current_nodes : integer
        Number of nodes currently on the string.
    nodes : integer
        The total number of nodes which the user has selected for optimisation.
    ID : string
        String which indicates whether a node is added to the reactant side ('R') or the product side ('P').

    Returns
    -------
    new_xyz : Numpy array
        Numpy array containing the new node's xyz coordinates.
        
    """ 
   
    # The stepsize of the coordinate change is defined based on the current number of nodes.
    if (nodes - current_nodes) > 1:
        stepsize = 1./float(nodes - current_nodes)
    else:
        stepsize = 0.5 
   
    # Using the kwarg ID, a new node is added to either the product or reactant side.
    if ID == 'R':
        # The coordinate systems are initialised.
        i_connects, i_prims, i_B_matp, i_G_mat, i_U_mat, i_B_mat, i_S = dlc.gen_dlc(i_coords)
        j_connects, j_prims, j_B_matp, j_G_mat, j_U_mat, j_B_mat, i_S = dlc.gen_dlc(j_coords)
        
        # The primitive internal coordinate tangent is generated, scaled by the stepsize, any which are small are set to zero, and are used to generate next node cartesians.
        ij_tan = tangent(i_prims, j_prims)
        dq = ij_tan * stepsize
        dq[abs(dq) < 1 * 10**-5] = 0
        new_xyz = dlc.prim_to_cartesian(dq, i_prims, i_coords, i_B_matp)
        
    elif ID == 'P':
        # The coordinate systems are initialised.
        i_connects, i_prims, i_B_matp, i_G_mat, i_U_mat, i_B_mat, i_S = dlc.gen_dlc(i_coords)
        j_connects, j_prims, j_B_matp, j_G_mat, j_U_mat, j_B_mat, j_S = dlc.gen_dlc(j_coords)
        
        # The primitive internal coordinate tangent is generated, scaled by the stepsize, any which are small are set to zero, and are used to generate next node cartesians.
        ji_tan = tangent(j_prims, i_prims)
        dq = ji_tan * stepsize
        dq[abs(dq) < 1 * 10**-5] = 0
        new_xyz = dlc.prim_to_cartesian(dq, j_prims, j_coords, j_B_matp)
       
    return new_xyz

def opt_frontier(i_coords, j_coords, num_nodes):
    """
    
    // Function which optimises the frontier nodes during the growth phase. //
    
    Parameters
    ----------
    i_coords : Numpy array
        Numpy array containing the xyz coordinates of node i.
    j_coords : Numpy array
        Numpy array containing the xyz coordinates of node j.
    num_nodes : integer
        The number of nodes which is to be optimised - 1 for just reactant side, and 2 for both sides.

    Returns
    -------
    i_coords_opt : Numpy array
        Numpy array containing the optimised xyz coordinates of node i.
    j_coords_opt : Numpy array
        Numpy array containing the optimised xyz coordinates of node j (if calculated - num_nodes must equal 2).
        
    """ 

    # Primitive internal coordinates for both frontier nodes are generated so that the tangents between them for both directions can be calculated.
    _, i_prims = dlc.prim_coords(i_coords)
    _, j_prims = dlc.prim_coords(j_coords)
    ij_tan = tangent(i_prims, j_prims)
    ji_tan = tangent(j_prims, i_prims)
    
    # If num_nodes is 2, then both nodes are optimised.
    # However, if num_nodes is 1, then only the reactant side frontier node is optimised. This represents the case of a central node.
    # Both/one of the node(s) are optimised using quasi-Newton methods.
    if num_nodes == 2:
        i_coords_opt = opt.quasi_newton(i_coords, 10)#, ij_tan)
        j_coords_opt = opt.quasi_newton(j_coords, 10)#, ji_tan)
        
        return i_coords_opt, j_coords_opt
    if num_nodes == 1:
        i_coords_opt = opt.quasi_newton(i_coords, 10)#, ij_tan)
        
        return i_coords_opt
    
def optimisation_phase(cwd, nodes):
    # For simplicity, the files saved from the growth phase are reopened and optimised.
    # Probably not very efficient, but works for now...
    for i in range(2, nodes - 1):
        # To define the tangent to be constrained, the paths for the previous and current node are initialised.
        prev_node_path = os.path.join(cwd, "Node_" + str(i - 1))
        cur_node_path = os.path.join(cwd, "Node_" + str(i))
        prev_node = os.path.join(cwd, "Node_" + str(i - 1) + '\geom_G.xyz')
        cur_node = os.path.join(cwd, "Node_" + str(i) + '\geom_G.xyz')
        prev_coords = utils.extract_coords2(prev_node)
        cur_coords = utils.extract_coords2(cur_node)
        
        # The tangent between the previous and current node is obtained and used in constrained optimisation of the current node.
        _, prev_prims = dlc.prim_coords(prev_coords)
        _, cur_prims = dlc.prim_coords(cur_coords)
        ij_tan = tangent(prev_prims, cur_prims)
        cur_coords_opt = opt.quasi_newton(cur_coords, 10)#, ij_tan)
        
        ###########################
        # Saves node to directory #
        ###########################
        os.chdir(cur_node_path)
        utils.xyz_format(cur_coords_opt, phase = 'O')
        potential = m_utils.LJ_potential(cur_coords_opt, 1, 1)
        g_array = m_utils.LJ_grad(cur_coords_opt, 1, 1)
        with open("Energy_opt", 'w') as f:
            f.write("Energy: " + str(potential))
        np.savetxt("Gradient_opt", g_array)
        os.chdir(cwd)
        
        return cur_coords_opt, cur_node_path