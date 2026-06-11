#!/usr/bin/env python3
# coding: utf-8
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import sys
from shutil import rmtree
import glob
from PIL import Image
import rebound as rb
import reboundx as rbx
from tqdm import tqdm

# using the argparse method
parser = argparse.ArgumentParser(description='This script allows to analyze the three body problem by the way of analyzing the orbits using rebound')

# arguments of the script
parser.add_argument('-Mbh', '--M_bh', type=float, default=20.0, help='Black hole mass, it must be in solar masses')
parser.add_argument('-Mstar', '--M_star', type=float, default=5.0, help='Star mass, it must be in solar masses.')
parser.add_argument('-e', '--ecc', type=float, default=0.0, help='The eccentricity of the binary orbit. It is a non-negative number that defines its shape. e=0: circular orbit; 0<e<1: Elliptic orbit.')

parser.add_argument('-rhop', '--rho_p', type=float, default=1.0, help='Density of the planet, it must be in terms of Earth density')

# Saving the arguments of the script
#args = parser.parse_args()

# Useful for jupyter and console
if hasattr(sys, 'ps1') or 'ipykernel' in ''.join(sys.argv):
    args, unknown = parser.parse_known_args(args=[])
else:
    args = parser.parse_args()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Critical semimajor axis for the binary.
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def stability_function(M_bh, M_star, ecc):
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
    return  a_b_AU
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Now, with the previous result it is possible to consider the value for the planet semimajor axis
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def planet(M_bh, M_star, ecc, rho_p, a_b):
    '''
    This part calculates the critical separation between the star and planet (Jupyter-like planet). It is based on Egglenton calculation
    '''
    u = M_star / (M_bh + M_star)
    ac = (0.464 - 0.380*u - 0.631*ecc + 0.586*ecc*u + 0.150*(ecc**2) - 0.198*u*(ecc**2)) * a_b

    '''
    This part help us to calculate the black hole's disruption radius
    '''
    constant = 1.16
    density_term = rho_p**(-1/3)
    mass_term = (M_bh / 0.6)**(1/3)
    rt_solar = constant * density_term * mass_term
    rt_AU = 0.00465 * rt_solar # Result in AU
    return ac, rt_AU

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#function that defines the parameters of the simulation
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def planet_simulation(M_p, M_bh, M_star, R_p, R_star, a_p, a_b, ecc, ecc_p, step,use_gr=False, nout=100):
    # 1. Simulation set up
    sim = rb.Simulation()
    sim.units = ('yr', 'AU', 'Msun')# units: time in years, distance in astronomical units, Mass in solar masses
    
    # 2. Adding the elements of the system
    sim.add(m=M_bh) #black hole
    sim.add(m=M_star, a=a_b, e=ecc, r=R_star) #Star in solar masses, Radius of the star, 
    sim.add(m=M_p, r=R_p, a=a_p, e=ecc_p, primary=sim.particles[1]) # Planet, mass, radius, semimajor axis, eccentricity
    sim.move_to_com()
    sim.integrator = "ias15"
    
    # 3. General relativity configuration
    if use_gr:
        rebx = rbx.Extras(sim)
        gr = rebx.load_force("gr_full")
        rebx.add_force(gr)
        gr.params["c"] = 63239.7263
        
    # 4. Integration time (solution of the problem)
    a = sim.particles[1].a
    M_total = sim.particles[0].m + sim.particles[1].m
    period = 2 * np.pi * np.sqrt(a**3 / (sim.G * M_total))
    times = np.linspace(0, step * period, nout)
    
    # 5. Objects position
    x_s, y_s, z_s = np.zeros(nout), np.zeros(nout), np.zeros(nout)
    x_bh, y_bh, z_bh = np.zeros(nout), np.zeros(nout), np.zeros(nout)
    x_p, y_p, z_p = np.zeros(nout), np.zeros(nout), np.zeros(nout)
    orbits = np.zeros(nout)
    
    print(f'Period: {period} ~yrs')
    
    # 6. Integration cycle
    for i, t in enumerate(tqdm(times, desc="Simulation")):
        sim.integrate(t)
        
        # star position (particle 1)
        x_s[i], y_s[i], z_s[i] = sim.particles[1].x, sim.particles[1].y, sim.particles[1].z
        
        # black hole position (particle 0)
        x_bh[i], y_bh[i], z_bh[i] = sim.particles[0].x, sim.particles[0].y, sim.particles[0].z
        
        # planet position (particle 2)
        x_p[i], y_p[i], z_p[i] = sim.particles[2].x, sim.particles[2].y, sim.particles[2].z
        
        # eccentricity of the planet
        orbits[i] = sim.particles[2].orbit(primary=sim.particles[1]).e
        
    # 7. Distances
    r_pro = np.sqrt((y_p - y_s)**2 + (z_p - z_s)**2) #projected distance
    r_ps = np.sqrt((x_p - x_s)**2 + (y_p - y_s)**2) #distance between planet and star
    
    return times, x_s, y_s, x_bh, y_bh, x_p, y_p, r_ps, r_pro, orbits


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Function for plotting and creation of the simulation
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_orbit(tG,x_sG, x_bhG, x_pG,y_sG, y_bhG, y_pG, ab,gr=False):
    if os.path.isdir('./outputfolder'):
            print("Directory './outputfolder' already exists...")
    else:
        print("Directory './outputfolder' is being created...")
        os.mkdir('./outputfolder')
    if os.path.isdir('./outputfolder/imgs'):
            print("Directory './outputfolder/imgs' already exists...")
    else:
        print("Directory './outputfolder/imgs' is being created...")
        os.mkdir('./outputfolder/imgs')

    for t in range(len(tG)):
        plt.figure(figsize=(8, 8))
        # Final Positions of the objects (plotted as dots)
        plt.scatter(x_sG[t], y_sG[t], color="green", s=20)
        plt.scatter(x_bhG[t], y_bhG[t], color="orange", s=30)
        plt.scatter(x_pG[t], y_pG[t], color="blue", s=10)
        #plt.scatter(com.x,com.y, color='red',label='cm')
    
        plt.plot(x_sG[0:t+1], y_sG[0:t+1], linestyle = 'dotted',label="star", color="green", lw=3)
        plt.plot(x_bhG[0:t+1], y_bhG[0:t+1], linestyle = '--',label="Black Hole", color="orange")
        plt.plot(x_pG[0:t+1], y_pG[0:t+1], linestyle = '-',label="Planet", color="blue", alpha=0.6)
    
        #plt.axis('equal')
        plt.xlim(-ab-0.02,ab+0.5)
        plt.ylim(-ab-0.02,ab+0.02)
        plt.xlabel('Distance~[AU]')
        plt.ylabel('Distance~[AU]')
        if gr==True:
            plt.title(r"$Restricted ~system: Planet -> star-> black hole~GR$")
        else:
            plt.title(r"$Restricted ~system: Planet -> star-> black hole~Newton$")
        plt.legend()
        plt.savefig("./outputfolder/imgs/orbit{:04d}.png".format(t))
        plt.close()
    print('Frames created...')
    print('Creating movie...')
    while True:
        key = input("Introduce the name of your simulation (must be a valid file name): ").strip()
        if key: 
            break
        print("❌ Error: The file name cannot be empty.")

    #key=input("Introduce the name of your simulation(must be string): ")
    images_in = "./outputfolder/imgs/orbit******.png"
    gif_out = f"./outputfolder/{key}.gif"
    imgs = (Image.open(f) for f in sorted(glob.glob(images_in)))
    img = next(imgs)
    img.save(fp = gif_out, format='GIF', append_images=imgs, save_all=True, duration=0.01, loop=0)

    rmtree('./outputfolder/imgs')
    print("The movie was created and saved in './outputfolder'")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Transit time variations
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def transit_time(time, flux):
    # 1. Dataframe with the results
    df_lc = pd.DataFrame({'time': time, 'flux': flux})
    
    # 2. Identifying the points where transit happens.
    df_lc['in_transit'] = df_lc['flux'] < 1.0
    
    # 3. Detecting the specific points where transit happens (False-True)
    df_lc['start_transit'] = df_lc['in_transit'] & (~df_lc['in_transit'].shift(1, fill_value=False))
    
    # 4. Extracting the times
    times_transit = df_lc[df_lc['start_transit']]['time'].values
    
    # 5. Calculating the consecutive transit intervals
    intervals = np.diff(times_transit)
    average_period = np.mean(intervals)
    deviation_ttv = np.std(intervals)
    
    # 6. printing the results
    print("==================================================")
    print(f"Total number of transits detected: {len(times_transit)}")
    print(f"average period of transits: {average_period:.6f} años")
    print(f"Transit time variation magnitude (standard deviation): {deviation_ttv :.6f} años")
    print("==================================================")