# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 13:32:22 2021

@author: Neil McFarlane
"""

"""
Utilities for the growing string method.
The functions within this code are called as utils, i.e.,:
    
    import utils as utils
    
"""

#################################################################################################################################################################################

# Standard library imports.

# Third party imports.
import numpy as np
import pandas as pd

# Local imports.

#################################################################################################################################################################################

def xyz_format(input_array, phase = 'G'):
    """
    
    // Function which takes a Numpy array containing only xyz coordinates and saves it to a formatted xyz file for viewing in VMD or similar. //
    
    Parameters
    ----------
    input_array : Numpy array
        A Numpy array containing only the xyz coordinates which is to be converted to a formatted xyz file.
    phase : string
        String representing whether it is the growth phase ('G') or optimisation phase ('O'). 'G' is the default.

    Returns
    -------
    output_file : file
        The formatted xyz is saved.
    
    """

    # The Pandas dataframe coord_df is created for ease of formatting.
    # The numbers are saved as floating point values, and all atoms are set to xenon.        
    coord_df = pd.DataFrame(input_array, columns = ['X', 'Y', 'Z'])
    coord_df['X'] = coord_df['X'].astype(float)
    coord_df['Y'] = coord_df['Y'].astype(float)
    coord_df['Z'] = coord_df['Z'].astype(float) 
    coord_df['ATOM'] = 'Xe'
    
    # Details are converted to variables for ease of working.
    tot_atoms = len(coord_df)
    title = 'LJ geometry.'
    coord_df = coord_df.to_string(columns = ['ATOM', 'X', 'Y', 'Z'], header = False, index = False)
        
    # The output xyz file is written.
    with open('geom_' + phase + '.xyz', 'w') as fo:
        fo.write(str(tot_atoms) + '\n')
        fo.write(title + '\n')
        fo.write(coord_df)
        
def extract_coords(input_geometry):
    """
    
    // Function which reads a geometry file containing only xyz coordinates and converts it to a Numpy array containing the coordinates. //
    
    Parameters
    ----------
    input_geometry : file
        A file containing only the xyz coordinates which is read and stored in a Numpy array.    

    Returns
    -------
    coords : Numpy array
        Numpy array containing the xyz coordinates of the initial stucture. This array will be updated as optimisation proceeds.

    """
    
    # The Numpy array, coords is initialised using the number of lines in the initial geometry file.
    num_atoms = 0
    with open(input_geometry, 'r') as f1:
        for line in f1:
            num_atoms += 1
        coords = np.zeros([num_atoms, 3])
            
    # The geometry selected by the user is opened and each line is read.
    # The x, y, and z coordinates are added to the Numpy array coords.
    with open(input_geometry, 'r') as f2:
        i = 0
        for line in f2:
            split_line = line.split()
            np_all = np.array(split_line)
            coords[i][0] = float(np_all[0])
            coords[i][1] = float(np_all[1])
            coords[i][2] = float(np_all[2])
            i += 1

    return coords

def extract_coords2(input_xyz):
    """
    
    // Function which reads a formatted xyz geometry file and converts it to a Numpy array containing the coordinates. //
    
    Parameters
    ----------
    input_xyz : file
        A formatted xyz file which is read and stored in a Numpy array.    

    Returns
    -------
    coords : Numpy array
        Numpy array containing the xyz coordinates of the initial stucture. This array will be updated as optimisation proceeds.

    """
    
    # The Numpy array, coords is initialised using the number of lines in the initial geometry file and subtracting 2 to account for formatting lines.
    num_atoms = 0
    with open(input_xyz, 'r') as f1:
        for line in f1:
            num_atoms += 1
        coords = np.zeros([num_atoms - 2, 3])
            
    # The geometry selected by the user is opened and each line is read.
    # The x, y, and z coordinates are added to the Numpy array coords.
    with open(input_xyz, 'r') as f2:
        i = 0
        # The first two lines are read to account for formatting
        f2.readline()
        f2.readline()
        for line in f2:
            split_line = line.split()
            np_all = np.array(split_line)
            coords[i][0] = float(np_all[1])
            coords[i][1] = float(np_all[2])
            coords[i][2] = float(np_all[3])
            i += 1

    return coords

def print_big(string):
    """
    
    // Function which takes a string, converts it to another string with stylised formatting, and prints it. //
    
    Parameters
    ----------
    string : string
        A string which is to be stylised.
        
    Returns
    -------
    string_big : printed string
        A string which is the same as the initial string, but stylised with a frame of hyphens.
    
    """
    
    text = '// ' + string + ' //'
    cap = len(text) * '-'
    
    string_big = cap + '\n' + text + '\n' + cap + '\n'
    
    print(string_big)
    
def print_small(string):
    """
    
    // Function which takes a string, converts it to another string with stylised formatting, and prints it. //
    
    Parameters
    ----------
    string : string
        A string which is to be stylised.
        
    Returns
    -------
    string_big : printed string
        A string which is the same as the initial string, but stylised with a border of forward-slashes.
    
    """
    
    text = '// ' + string + ' //' + '\n'

    print(text)    