# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 11:09:56 2021

@author: Neil McFarlane
"""

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The program takes one of a series of optimised Lennard-Jones (LJ) cluster structures and generates a reaction pathway using the growing string method (GSM).
The program can operate with either use the double-ended GSM (DE-GSM) or single-ended GSM (SE-GSM) using delocalised internal coordinates (dlc).
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

#################################################################################################################################################################################

# Standard library imports.
import os
import math

# Third party imports.
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import shutil as shutil

# Local imports.
import utils as utils
import math_utils as m_utils
import delocalised as dlc
import de_utils as DE
import se_utils as SE

#################################################################################################################################################################################


def DE_GSM(r_dir, p_dir, LJ_structure, nodes):
    """
    
    // Function which starts the DE-GSM. //
    
    Parameters
    ----------
    r_dir : string
        String which contains the directory where the reactant structures are maintained.
    p_dir : string
        String which contains the directory where the product structures are maintained.
    LJ_structure : string
        String which identifies the LJ structure which is to be optimised.
    nodes : integer
        Integer which describes the number of nodes the user has selected to use for optimisation.

    Returns
    -------
    A series of directories representing the nodes which have been generated over the course of reaction path exploration.
        
    """ 
    
    # The reactant and product x, y and z coordinates are extracted.
    r_dir = r_dir + '/' + LJ_structure
    r_coords = utils.extract_coords(r_dir)
    p_dir = p_dir + '/' + LJ_structure
    p_coords = utils.extract_coords(p_dir)
    
    
    # The reactant and product node geometies, energies and gradients are saved to new directories for later use.
    r_path = os.path.join(cwd, "Node_" + str(1))
    p_path = os.path.join(cwd, "Node_" + str(nodes))
    if os.path.exists(r_path):
        shutil.rmtree(r_path)
    if os.path.exists(p_path):
        shutil.rmtree(p_path)
    os.mkdir(r_path)
    os.mkdir(p_path)
            
    # Reactant node data saved.
    os.chdir(r_path)
    utils.xyz_format(r_coords)
    r_potential = m_utils.LJ_potential(r_coords, Xe_epsilon, Xe_sigma)
    r_g_array = m_utils.LJ_grad(r_coords, Xe_epsilon, Xe_sigma)
    with open("Energy", 'w') as f:
        f.write("Energy: " + str(r_potential))
    np.savetxt("Gradient", r_g_array)
    os.chdir(cwd)
            
    # Product node data saved.
    os.chdir(p_path)
    utils.xyz_format(p_coords)  
    p_potential = m_utils.LJ_potential(p_coords, Xe_epsilon, Xe_sigma)
    p_g_array = m_utils.LJ_grad(r_coords, Xe_epsilon, Xe_sigma)
    with open("Energy", 'w') as f:
        f.write("Energy: " + str(p_potential))
    np.savetxt("Gradient", p_g_array)
    os.chdir(cwd)
                
    # Growth phase...
    utils.print_small("Growing the string...")
    
    # The current number of nodes is initially 2 because of the reactant and product structures.
    # Normal node addition scheme.
    current_nodes = 2
    for (i,j) in zip(range(2, nodes, 1), range(nodes - 1, math.ceil(nodes / 2), -1)):
        
        # Two new nodes are added, and the current number of nodes is updated.
        r_coords, p_coords = DE.add_nodes(r_coords, p_coords, nodes, current_nodes, new_nodes = 2)
        current_nodes = current_nodes + 2
        
        # The frontier nodes are optimised.
        r_coords, p_coords = DE.opt_frontier(r_coords, p_coords, num_nodes = 2)            
        
        # Initialising paths to save frontier nodes' data.
        r_path = os.path.join(cwd, "Node_" + str(i))
        p_path = os.path.join(cwd, "Node_" + str(j))
        os.mkdir(r_path)
        os.mkdir(p_path)
        
        # Reactant side frontier node data saved.
        os.chdir(r_path)
        utils.xyz_format(r_coords)
        r_potential = m_utils.LJ_potential(r_coords, Xe_epsilon, Xe_sigma)
        r_g_array = m_utils.LJ_grad(r_coords, Xe_epsilon, Xe_sigma)
        with open("Energy", 'w') as f:
            f.write("Energy: " + str(r_potential))
        np.savetxt("Gradient", r_g_array)
        os.chdir(cwd)
        
        # Product side frontier node data saved.
        os.chdir(p_path)
        utils.xyz_format(p_coords)  
        p_potential = m_utils.LJ_potential(p_coords, Xe_epsilon, Xe_sigma)
        p_g_array = m_utils.LJ_grad(r_coords, Xe_epsilon, Xe_sigma)
        with open("Energy", 'w') as f:
            f.write("Energy: " + str(p_potential))
        np.savetxt("Gradient", p_g_array)
        os.chdir(cwd)

        
    # For an odd number of nodes, one final node from the reactant side must be added and then the growth phase is terminated.
    if current_nodes < nodes:
        
        # One final node is added, and the current number of nodes is updated.
        r_coords = DE.add_nodes(r_coords, p_coords, nodes, current_nodes, new_nodes = 1)
        current_nodes = current_nodes + 1
        
        # Initialising path to save final frontier node data.
        r_path = os.path.join(cwd, "Node_" + str(math.ceil(nodes / 2)))
        os.mkdir(r_path)
        
        # Reactant side frontier node data saved.
        os.chdir(r_path)
        utils.xyz_format(r_coords)
        r_potential = m_utils.LJ_potential(r_coords, Xe_epsilon, Xe_sigma)
        r_g_array = m_utils.LJ_grad(r_coords, Xe_epsilon, Xe_sigma)
        with open("Energy", 'w') as f:
            f.write("Energy: " + str(r_potential))
        np.savetxt("Gradient", r_g_array)
        os.chdir(cwd)

    # Optimisation phase...
    utils.print_small("Optimising the string...")
    
    # Every node along the string is optimised with more optimisation iterations.
    #coords_opt, path = DE.optimisation_phase(cwd, nodes)

def SE_GSM(r_dir, LJ_structure):
    """
    
    // Function which starts the SE-GSM. //
    
    Parameters
    ----------
    r_dir : string
        String which contains the directory where the reactant structures are maintained.
    LJ_structure : string
        String which identifies the LJ structure which is to be optimised.

    Returns
    -------
    A series of directories representing the nodes which have been generated over the course of reaction path exploration.
        
    """ 
    
    # The reactant and product x, y and z coordinates are extracted.
    r_dir = r_dir + '/' + LJ_structure
    r_coords = utils.extract_coords(r_dir)
      
    # The reactant and product node geometies, energies and gradients are saved to new directories for later use.
    r_path = os.path.join(cwd, "Node_" + str(1))
    os.mkdir(r_path)
            
    # Reactant node data saved.
    os.chdir(r_path)
    utils.xyz_format(r_coords)
    r_potential = m_utils.LJ_potential(r_coords, Xe_epsilon, Xe_sigma)
    r_g_array = m_utils.LJ_grad(r_coords, Xe_epsilon, Xe_sigma)
    with open("Energy", 'w') as f:
        f.write("Energy: " + str(r_potential))
    np.savetxt("Gradient", r_g_array)
    os.chdir(cwd)
        
    # Growth phase...
    utils.print_small("Growing the string...")
        
    # The current number of nodes is initially 1 because of the reactant structure.
    # Normal node addition scheme.
    current_nodes = 1
    for i in range(2, nodes, 1):
        dlc.SE.add_node(r_coords, current_nodes, new_nodes = 1)
        current_nodes = current_nodes + 1 
  
#################################################################################################################################################################################

#######################################################################
### Making directories, assigning paths, and initialising constants ###
#######################################################################

# Using os.getcwd(), the necessary directories and paths are generated.
cwd = os.getcwd()
r_dir = cwd + '/reac_clusters'
p_dir = cwd + '/prod_clusters'

# The list of LJ cluster geometries in both product and reactant directories is obtained using os.listdir.
r_list = os.listdir(r_dir)
p_list = os.listdir(p_dir)

# LJ potential parameters taken from http://www.rsc.org/suppdata/c7/cp/c7cp07170a/c7cp07170a1.pdf.
Xe_sigma = 3.94 # Angstroms.
Xe_epsilon = 0.02 # eV.

#################################################################################################################################################################################

#########################
### Taking User Input ###
#########################

utils.print_big("Which of the following Lennard-Jones clusters would you like to optimise with the growing string method?")
for i in r_list:
    print(i + ', ', end = '')
print('\n')
#LJ_structure = str(input('> '))
LJ_structure = '7A'

if LJ_structure not in r_list:
    raise RuntimeError("The specified structure cannot be found.")
if LJ_structure in r_list:
    if LJ_structure not in p_list:
        utils.print_small("Note, no product structure could be found for this structure, so optimisation can only be done with SE-GSM.")
    
utils.print_big("Would you like to use the double- (1) or single-ended (2) growing string method?")
#gsm_choice = int(input('> '))
gsm_choice = 1

if gsm_choice != 1 and gsm_choice !=2:
    raise RuntimeError("Please only input either 1 for double-ended or 2 for single-ended.") 
if gsm_choice == 1 and LJ_structure not in p_list:
    raise RuntimeError("Only a reactant structure was found, so DE-GSM cannot be used with this structure.")  
if gsm_choice == 2:
    raise NotImplementedError("Sorry, you can't use the single-ended growing string method yet... ¯\_(ツ)_/¯")

if gsm_choice == 1:
    utils.print_big("How many nodes would you like to use (including reactant and product nodes)?")
    #nodes = int(input('> '))
    nodes = 4
    
    if isinstance(nodes, int) is False:
        raise RuntimeError("Please only input integers for the number of nodes.")
    elif nodes < 3:
        raise RuntimeError("The number of nodes must be greater than 2.")
    elif nodes == 0:
        raise RuntimeError("Zero nodes is not valid for optimisation.")

print("________________________________________________________________________________________________________" + '\n')

# Let's go!
if gsm_choice == 1:
    DE_GSM(r_dir, p_dir, LJ_structure, nodes)
if gsm_choice == 2:
    SE_GSM(r_dir, LJ_structure, nodes)