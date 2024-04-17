# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 13:32:23 2021

@author: Neil McFarlane
"""

"""
Cartesian functions for the growing string method.
The functions within this code are called as cart, i.e.,:
    
    import cartesian as cart
    
"""

#################################################################################################################################################################################

# Standard library imports.
import math
import re

# Third party imports.
import numpy as np
import sympy as sym

# Local imports.

#################################################################################################################################################################################
   
def exact_hess_LJ_equations(epsilon, sigma):
    """
    
    // Function which generates all the required equations for calculation of the exact hessian using the module Sympy. //
    
    Parameters
    ----------
    epsilon : float
        Lennard-Jones parameter.
    sigma : float
        Lennard-Jones parameter.

    Returns
    -------
    second_der_xA2, second_der_yA2, second_der_zA2, second_der_xA_xB, second_der_xA_yA, second_der_xA_zA, second_der_yA_xA, second_der_yA_yB, second_der_yA_zA, second_der_zA_xA, second_der_zA_yA, second_der_zA_zB : Sympy equations
        A series of second derivative expressions used to calculate the exact hessian.

    """
    
    # The symbols and expression for the LJ potential with expanded r is initialised.
    xA, yA, zA, xB, yB, zB = sym.symbols('xA yA zA xB yB zB')
    LJ_expr  = (4 * epsilon * ((sigma / sym.sqrt((xA - xB)**2 + (yA - yB)**2 + (zA - zB)**2))**12 - (sigma / sym.sqrt((xA - xB)**2 + (yA - yB)**2 + (zA - zB)**2))**6))

    # The first derivatives with respect to xA, yA and zA are expressed.
    first_der_xA = sym.diff(LJ_expr, 'xA')
    first_der_yA = sym.diff(LJ_expr, 'yA')
    first_der_zA = sym.diff(LJ_expr, 'zA')
    
    # The second derivatives with respect to xA, yA and zA are expressed.
    second_der_xA2 = sym.diff(first_der_xA, 'xA')
    second_der_yA2 = sym.diff(first_der_yA, 'yA')
    second_der_zA2 = sym.diff(first_der_zA, 'zA')
    
    # The second derivates with respect to xB, yA, yB, zA, and zB for xA are expressed.
    second_der_xA_xB = sym.diff(first_der_xA, 'xB')
    second_der_xA_yA = sym.diff(first_der_xA, 'yA')
    second_der_xA_zA = sym.diff(first_der_xA, 'zA')
    
    # The second derivates with respect to xA, yB and zA for yA are expressed.
    second_der_yA_xA = sym.diff(first_der_yA, 'xA')
    second_der_yA_yB = sym.diff(first_der_yA, 'yB')
    second_der_yA_zA = sym.diff(first_der_yA, 'zA')
    
    # The second derivates with respect to xA, yA and zB for zA are expressed.
    second_der_zA_xA = sym.diff(first_der_zA, 'xA')
    second_der_zA_yA = sym.diff(first_der_zA, 'yA')
    second_der_zA_zB = sym.diff(first_der_zA, 'zB')

    # All the second derivative equations are returned.
    return second_der_xA2, second_der_yA2, second_der_zA2, second_der_xA_xB, second_der_xA_yA, second_der_xA_zA, second_der_yA_xA, second_der_yA_yB, second_der_yA_zA, second_der_zA_xA, second_der_zA_yA, second_der_zA_zB
    
def exact_hess_calc(coords, epsilon, sigma):
    """
    
    // Function which calculates the exact hessian of a defined structure using cartesian coordinates. //
    
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
    hess : Numpy array
        Numpy array containing the hessian matrix for the LJ structure.

    """
  
    # The second derivative equations are initialised.
    second_der_xA2, second_der_yA2, second_der_zA2, second_der_xA_xB, second_der_xA_yA, second_der_xA_zA, second_der_yA_xA, second_der_yA_yB, second_der_yA_zA, second_der_zA_xA, second_der_zA_yA, second_der_zA_zB = exact_hess_LJ_equations(epsilon, sigma)    
  
    # The total number of atoms is taken from the length of the Numpy array coords.
    n_atoms = len(coords)
    
    # Using the total number of atoms, the shape of the Hessian is initialised in a Numpy array of dimensions 3N x 3N.
    hess = np.zeros([n_atoms * 3, n_atoms * 3], float)
    
    # The inidividual x, y, and z coordinates for every atom are initialised as labels in the list hess_labels.
    hess_labels = []
    for i in range(1, n_atoms + 1):
        x_label = 'x' + str(i)
        y_label = 'y' + str(i)
        z_label = 'z' + str(i)
        hess_labels.append(x_label)
        hess_labels.append(y_label)
        hess_labels.append(z_label)
  
    for i in range(0, n_atoms * 3):
        # In order to identify the atoms by number, the list variable atomi is created using hess_labels.
        atomi_lst = [int(num) for num in re.findall(r'\d+', hess_labels[i])]
        for x in atomi_lst:
            atomi = int(x)
   
        for j in range(0, n_atoms * 3):
            # Using the number of atoms as an upper bound, for each atom, the distance to every other atom is calculated.
            # If the distance is zero, then it is the same atom and is omitted from further calculation.
            # When the value is not zero, then r is added to the list r_list.
            interaction_list = []
            for l in range(1, n_atoms + 1):
                r = math.sqrt((float(coords[atomi-1][0]) - float(coords[l-1][0]))**2 + (float(coords[atomi-1][1]) - float(coords[l-1][1]))**2 + (float(coords[atomi-1][2]) - float(coords[l-1][2]))**2)
                if r > 0.00001:
                    if not r in interaction_list:
                        interaction_list.append(l)

            # In order to identify the atoms by number, the list variable atomj is created using hess_labels.
            atomj_lst = [int(num) for num in re.findall(r'\d+', hess_labels[j])]
            for x in atomj_lst:
                atomj = int(x)

            # The list lst is assigned based on the atom identifiers atomi and atomj
            lst = []
            if atomi == atomj:
                lst = interaction_list  
            elif atomi != atomj:
                lst.append(atomj)
   
            # For each term of the Hessian matrix, using lst, every term and its corresponding second derivative is calculated.
            # This is achieved by using a series of if and elif statements where the list hess_labels is used extensively.
            out_term = 0
            for k in lst: 
                
                if ("x" in hess_labels[i]):
                    if ("x" in hess_labels[i]) and ("x" in hess_labels[j]):
                        if hess_labels[i] == hess_labels[j]:
                            added_term = second_der_xA2.subs([('xA', coords[atomi-1][0]), ('yA', coords[atomi-1][1]), ('zA', coords[atomi-1][2]), ('xB', coords[k-1][0]), ('yB', coords[k-1][1]), ('zB', coords[k-1][2])])
                        else:
                            added_term = second_der_xA_xB.subs([('xA', coords[atomi-1][0]), ('yA', coords[atomi-1][1]), ('zA', coords[atomi-1][2]), ('xB', coords[k-1][0]), ('yB', coords[k-1][1]), ('zB', coords[k-1][2])])
                    elif ("x" in hess_labels[i]) and ("y" in hess_labels[j]):
                        added_term = second_der_xA_yA.subs([('xA', coords[atomi-1][0]), ('yA', coords[atomi-1][1]), ('zA', coords[atomi-1][2]), ('xB', coords[k-1][0]), ('yB', coords[k-1][1]), ('zB', coords[k-1][2])])
                    elif ("x" in hess_labels[i]) and ("z" in hess_labels[j]):
                        added_term = second_der_xA_zA.subs([('xA', coords[atomi-1][0]), ('yA', coords[atomi-1][1]), ('zA', coords[atomi-1][2]), ('xB', coords[k-1][0]), ('yB', coords[k-1][1]), ('zB', coords[k-1][2])])
                
                elif ("y" in hess_labels[i]):
                    if ("y" in hess_labels[i]) and ("y" in hess_labels[j]):
                        if hess_labels[i] == hess_labels[j]:
                            added_term = second_der_yA2.subs([('xA', coords[atomi-1][0]), ('yA', coords[atomi-1][1]), ('zA', coords[atomi-1][2]), ('xB', coords[k-1][0]), ('yB', coords[k-1][1]), ('zB', coords[k-1][2])])
                        else:
                            added_term = second_der_yA_yB.subs([('xA', coords[atomi-1][0]), ('yA', coords[atomi-1][1]), ('zA', coords[atomi-1][2]), ('xB', coords[k-1][0]), ('yB', coords[k-1][1]), ('zB', coords[k-1][2])])
                    elif ("y" in hess_labels[i]) and ("x" in hess_labels[j]):
                        added_term = second_der_yA_xA.subs([('xA', coords[atomi-1][0]), ('yA', coords[atomi-1][1]), ('zA', coords[atomi-1][2]), ('xB', coords[k-1][0]), ('yB', coords[k-1][1]), ('zB', coords[k-1][2])])
                    elif ("y" in hess_labels[i]) and ("z" in hess_labels[j]):
                        added_term = second_der_yA_zA.subs([('xA', coords[atomi-1][0]), ('yA', coords[atomi-1][1]), ('zA', coords[atomi-1][2]), ('xB', coords[k-1][0]), ('yB', coords[k-1][1]), ('zB', coords[k-1][2])])

                elif ("z" in hess_labels[i]):
                    if ("z" in hess_labels[i]) and ("z" in hess_labels[j]):
                        if hess_labels[i] == hess_labels[j]:
                            added_term = second_der_zA2.subs([('xA', coords[atomi-1][0]), ('yA', coords[atomi-1][1]), ('zA', coords[atomi-1][2]), ('xB', coords[k-1][0]), ('yB', coords[k-1][1]), ('zB', coords[k-1][2])])
                        else:
                            added_term = second_der_zA_zB.subs([('xA', coords[atomi-1][0]), ('yA', coords[atomi-1][1]), ('zA', coords[atomi-1][2]), ('xB', coords[k-1][0]), ('yB', coords[k-1][1]), ('zB', coords[k-1][2])])
                    elif ("z" in hess_labels[i]) and ("x" in hess_labels[j]):
                        added_term = second_der_zA_xA.subs([('xA', coords[atomi-1][0]), ('yA', coords[atomi-1][1]), ('zA', coords[atomi-1][2]), ('xB', coords[k-1][0]), ('yB', coords[k-1][1]), ('zB', coords[k-1][2])])
                    elif ("z" in hess_labels[i]) and ("y" in hess_labels[j]):
                        added_term = second_der_zA_yA.subs([('xA', coords[atomi-1][0]), ('yA', coords[atomi-1][1]), ('zA', coords[atomi-1][2]), ('xB', coords[k-1][0]), ('yB', coords[k-1][1]), ('zB', coords[k-1][2])])

                else:
                    raise RuntimeError("Missing coordinates, check your geometry files.")
                out_term = out_term + added_term
                
            # The term which is added to the hessian is rounded to mitigate errors down the line.
            hess[i,j] = round(out_term, 4)
            
    # The hessian is a symmetric matrix by definition.
    # This criteria is checked in order to mitigate errors down the line.
    symmetry = (hess.transpose() == hess).all()
    if symmetry == False:
        raise RuntimeError("Hessian is not symmetric.")  

    return hess

def DE_tangent(i_coords, j_coords):
    """
    
    // Function which calculates the coordinate tangent vector between frontier nodes i and j for DE-GSM using cartesian coordinates. //
    
    Parameters
    ----------
    i_coords : Numpy array
        Numpy array containing the cartesian coordinates of node i.
    j_coords : Numpy array
        Numpy array containing the cartesian coordinates of node j.

    Returns
    -------
    cart_tan : Numpy array
        Numpy array containing the tangent vector between nodes i and j.
    """    
    
    #########################################
    # TO-DO: THIS IS NOT STRICTLY A TANGENT #
    #########################################
    
    cart_tan = i_coords - j_coords
    
    return cart_tan

def new_coords(cart_tan, stepsize, coords):
    """
    
    // Function which calculates the cartesian coordinates of a new node using the tangent between the frontier nodes. //
    
    Parameters
    ----------
    cart_tan : Numpy array
        Numpy array containing the tangent vector between nodes i and j.
    stepsize : floating point
        An integer representing how big of a jump there should be between the frontier node and the new node.
    coords : Numpy array
        Numpy array containing the cartesian coordinates of the previous frontier node.

    Returns
    -------
    new_xyz : Numpy array
        Numpy array containing the cartesian coordinates of the new frontier node.
        
    """
    
    ############################################################
    # TO-DO: NOT SURE THIS IS THE CORRECT CALCULATION OF D_XYZ #
    ############################################################
    
    d_xyz = cart_tan * stepsize * -1
    new_xyz = coords + d_xyz
    
    return new_xyz
    

def DE_add_nodes(i_coords, j_coords, nodes, current_nodes, new_nodes):
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
        
    """ 

    # If new_nodes = 1, then only one node is added to the reactant side, otherwise 2 are added - one to reactant side, and one to product side.
    for i in range(0, new_nodes):
        if i == 0:
            new_r_coords = DE_add_reacnode(i_coords, j_coords, current_nodes, nodes)
        elif i == 1:
            new_p_coords = DE_add_prodnode(i_coords, j_coords, current_nodes, nodes)
    
    if 'new_p_coords' not in dir():
        return new_r_coords
    else:
        return new_r_coords, new_p_coords

def DE_add_node(i_coords, j_coords, stepsize, ID = None):
    
    #######
    # WIP #
    #######
    
    if ID == 'R':
        
        ij_tan = DE_tangent(i_coords, j_coords)
        new_xyz = new_coords(ij_tan, stepsize, i_coords)
        
        return new_xyz
    
    elif ID == 'P':
        
        ji_tan = DE_tangent(j_coords, i_coords)
        new_xyz = new_coords(ji_tan, stepsize, j_coords)
        
        return new_xyz
        

def DE_add_reacnode(i_coords, j_coords, current_nodes, nodes):
    
    #######
    # WIP #
    #######
    
    if (nodes - current_nodes) > 1:
        stepsize = 1./float(nodes - current_nodes)
    else:
        stepsize = 0.5

    new_node = DE_add_node(i_coords, j_coords, stepsize, ID = 'R')
    
    return new_node

    
def DE_add_prodnode(i_coords, j_coords, current_nodes, nodes):
        
    #######
    # WIP #
    #######
    
    if (nodes - current_nodes) > 1:
        stepsize = 1./float(nodes - current_nodes)
    else:
        stepsize = 0.5
        
    new_node = DE_add_node(i_coords, j_coords, stepsize, ID = 'P')
    
    return new_node