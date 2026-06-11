#!/usr/bin/env python3
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
from shutil import rmtree
import glob
from PIL import Image
import rebound as rb
import reboundx as rbx

# using the argparse method
parser = argparse.ArgumentParser(description='This script allows to analyze the three body problem by the way of analyzing the orbits using rebound')

# arguments of the script
parser.add_argument('-Mbh', '--M_bh', type=float, default=20.0, help='Black hole mass, it must be in solar masses')
parser.add_argument('-Mstar', '--M_star', type=float, default=5.0, help='Star mass, it must be in solar masses.')
parser.add_argument('-e', '--ecc', type=float, default=0.0, help='The eccentricity of the binary orbit. It is a non-negative number that defines its shape. e=0: circular orbit; 0<e<1: Elliptic orbit.')
# Añadimos un valor por defecto o comportamiento para rho_p para evitar que falle si no se pasa
parser.add_argument('-rhop', '--rho_p', type=float, default=1.0, help='Density of the planet, it must be in terms of Earth density')

# Saving the arguments of the script
args = parser.parse_args()

# CORRECCIÓN: Los parámetros reciben nombres limpios (M_bh, M_star, ecc, rho_p)
def stability_function(M_bh, M_star, ecc, rho_p):
    '''
    This part calculates the distance separation between the black hole and star when mass transfer occurs. It is based on Egglenton's calculation
    '''
    q = M_star / M_bh
    R_star = M_star  # Assuming a main sequence star (R ~ M)
    
    # Using the Eggleton formula to determine the critical separation of the binary a_b
    numerator = 0.6 * q**(2/3) + np.log(1 + q**(1/3))
    denominator = 0.49 * q**(2/3)
    
    a_b_solar = R_star * (numerator / denominator)
    a_b_AU = 0.00465 * a_b_solar # Result in AU
    
    '''
    This part calculates the critical separation between the star and planet (Jupyter-like planet). It is based on Egglenton calculation
    '''
    u = M_star / (M_bh + M_star)
    # CORRECCIÓN: Se cambiaron las referencias internas de 'args.ecc' a 'ecc'
    ac = (0.464 - 0.380*u - 0.631*ecc + 0.586*ecc*u + 0.150*(ecc**2) - 0.198*u*(ecc**2)) * a_b_AU

    '''
    This part help us to calculate the black hole's disruption radius
    '''
    constant = 1.16
    density_term = rho_p**(-1/3)
    mass_term = (M_bh / 0.6)**(1/3)
    rt_solar = constant * density_term * mass_term
    rt_AU = 0.00465 * rt_solar # Result in AU

    return ac, rt_AU, a_b_AU

# Pasamos las variables del objeto 'args' como argumentos de la función
#re = stability_function(args.M_bh, args.M_star, args.ecc, args.rho_p)

#print('---------------------------------')
#print(f"The minimal semimajor axis for the binary is: {re[2]:.3f}")
#print('---------------------------------')
#print(f"The disruption radius is (lower limit for planet semimajor axis): {re[1]:.3f}")
#print('---------------------------------')
#print(f"The maximum semimajor axis for the planet is: {re[0]:.3f}")
#print('---------------------------------')
    


