#!/usr/bin/env python
# coding: utf-8

# # Python code for final project PHZ5305: Interaction matrix elements and Hartree-Fock (Nov. 29, 2024)
# 
# ## Introduction
# 
# This code is written for the final proejct of PHZ5305 (Nuclear Physics 1), which will provide the tool for the read the list of single particle states and me2b generated, and solves Hartree-Fock Equation.
# 
# Additionally, this code will optimize the me2b values into the value that simultaneously gives binding energies of the $^4 He$, $^{12} C$, $^{16} O$.

# ## Part 1: Import necessary packages.
# 
# `numpy` provides mathematical tools for calculation, such as `power`, `list`, and `array`.
# 
# `pandas` provides the reading csv file that contains the experimental informations that will provide the `me2b`.
# 
# `scipy` provides the optimization feature of this code.
# 
# `random` is imported for giving a random value assigned for the other `me2b` values.


import numpy as np
import pandas as pd
import scipy as scp
import scipy.sparse as sp
from scipy import optimize
from scipy.optimize import minimize
import random
import sys
import os
import curses

# ## Part 2: Class for Generate single-particle states, two-body matrix element calculation, and solve Hartree-Fock.
# 
# In this block, the Hartree-Fock equation will be solved via algorithm that intorudced at the class (Oct. 14, 2024) with single-particle states and two-body matrix elements. 
# 
# `SP_States` generates `dictionary` of SP states, $\pi 0s1/2$, $\nu 0s1/2$, $\pi 0p3/2$, $\nu 0p3/2$, $\pi 0p1/2$, $\nu 0p 1/2$ in `{index: [n, l, j, mj, tz]}` format.  
# 
# `me2b` gets 4 index values for $\alpha$, $\beta$, $\gamma$, and $\delta$.
# 
# `Solve_HF` solves Hartree-Fock equation with algorithm

class HF_Machina:
    # Initial definition of states
    def __init__(self):
        self.SP_States()
        self.TB_States()
        
    # Generate Dictionaries of Single-Particle States up to 16O
    def SP_States(self):
        
        # Initialize the important parameters which will be applied to loop #
        list_tz      = [-1/2, 1/2]
        n            = 0
        list_l       = [0, 1]
        dict_list_j  = {
            0: [1/2],
            1: [3/2, 1/2]
        }
        
        # Set the index and empty dictionary
        index        = 0 
        dict_SP      = {}
        
        # Loop for fullfill the dictionary
        for l in list_l:
            for j in dict_list_j[l]:     
                for tz in list_tz:
                    mj = -1*j
                    while mj < j+1:
                        dict_SP.update({index: [n,l,j,mj,tz]})
                        index += 1
                        mj += 1
        
        # Output
        self.dict_SP = dict_SP
        return dict_SP

    # Generate List of Two-body states in One-body basis (a,b) for Jpi = 0+
    def TB_States(self):
        
        #Initialize
        NSP = 16
        
        list_TB_ind = []
        
        for a in range(NSP):
            for b in range(NSP):
                # Imply possible conditions to be Jpi = 0+ 
                # ja = jb
                j_cond  = (self.dict_SP[a][2] == self.dict_SP[b][2])
                # mja = -mjb
                mj_cond = (self.dict_SP[a][3] == -1*self.dict_SP[b][3])
                # la = lb (since only possible l = 0 or 1)
                l_cond  = (self.dict_SP[a][1] == self.dict_SP[b][1])
                if j_cond and mj_cond and l_cond: list_TB_ind.append((a,b))
        
        list_TB_ind = list({frozenset(pair) for pair in list_TB_ind})
        list_TB_ind = [tuple(sorted(pair)) for pair in list_TB_ind]
        
        self.list_TB = list_TB_ind
        return list_TB_ind
        
    
    # Code for solving Hartree-Fock equation from core 12C 
    # A is the atomic number of nucleus 
    # Z is the charge of nucleus 
    # input_me2b is a me2b input with form of list [(a,b,c,d,v)]
    def Solve_HF(self, A, Z, input_me2b):
        
        # Prepare Initial Value
        self.A_c      = 12
        self.Z_c      = 6
        self.A        = A
        self.Z        = Z
        self.N        = A-Z
        NSP           = len(self.dict_SP)
        
        # Set list of protons
        list_proton   = []
        list_neutron  = []
        for key, value in self.dict_SP.items():
            if value[4] == -0.5 and len(list_proton) != self.Z: list_proton.append(key)
            elif value[4] == 0.5 and len(list_neutron) != self.N: list_neutron.append(key)
            if len(list_proton) == self.Z and len(list_neutron) == self.N : break
        
        list_A = list_proton+list_neutron
        
        # Prepare me2b
        self.dict_me2b = {(me2b[0], me2b[1], me2b[2], me2b[3]): me2b[4] for me2b in input_me2b}
        
        # Computing single-particle Hamiltonian (SPH)
        SPH = np.array([self.onebody(sp_ind) for sp_ind in range(len(self.dict_SP))])
        
        # Initialize the Coefficient Matrix Coe and Density Matrix Rho 
        Coe = np.eye(NSP)
        Rho = np.zeros((NSP, NSP), dtype=object)

        for i_gam in range(NSP):
            for i_del in range(NSP):
                Rho[i_gam, i_del] = sum(Coe[i_gam,i]*Coe[i_del,i] for i in list_A)
        
        # Now, we will do the algorithm that will be iterated until it converges at somewhere or its maximum iteration number
        # Set initial value for iteration
        maxiter, epsl = 100, 1e-3
        diff, i_count = 1.0, 0
        
        # Set Energies 
        oldE = np.zeros(NSP)
        newE = np.zeros(NSP)
        
        # Iterating alogrithm!
        while i_count < maxiter and diff > epsl:
            
            # Make Hartree-Fock Matrix
            MHF = np.zeros((NSP, NSP))
            
            sum_me2b = 0 
            
            processed_combinations = set()

            for i_alp in range(NSP):
                for i_bet in range(NSP):
                    if self.dict_SP[i_alp][1] != self.dict_SP[i_bet][1] : continue
                    sum_me2b = 0 
                    for i_gam in range(NSP):
                        for i_del in range(NSP):
                            # Check the conditions for matching SP values
                            if (self.dict_SP[i_alp][3]+self.dict_SP[i_gam][3] == self.dict_SP[i_bet][3]+self.dict_SP[i_del][3]) \
                               and (self.dict_SP[i_alp][4]+self.dict_SP[i_gam][4] == self.dict_SP[i_bet][4]+self.dict_SP[i_del][4]):
                                  
                                # ignore interaction between different js
                                if self.dict_SP[i_alp][2] != self.dict_SP[i_bet][2] : continue
                                # All possible permutations to check
                                permutations_to_check = [
                                    (i_alp, i_gam, i_bet, i_del),
                                    (i_alp, i_gam, i_del, i_bet),
                                    (i_gam, i_alp, i_del, i_bet),
                                    (i_gam, i_alp, i_bet, i_del)
                                ]
                                processed_combinations = set()
                                
                                # Find the first valid permutation in dict_me2b
                                for perm in permutations_to_check:
                                    if perm in processed_combinations:
                                        continue

                                    # Check if this permutation exists in dict_me2b
                                    value = self.dict_me2b.get(perm, None)
                                    processed_combinations.add(perm)
                                    
                                    if value is not None and value!=0:
                                        # Calculate the negative factor based on permutation number
                                        negative_factor = (-1) ** permutations_to_check.index(perm)

                                        # Add to sum with Rho and value
                                        sum_me2b += Rho[i_gam, i_del] * value * negative_factor
                                        
                                        # Mark this combination as processed
                                        #processed_combinations.add(perm)
                                        break
                    MHF[i_alp,i_bet] = sum_me2b + (SPH[i_alp] if i_alp == i_bet else 0)
            
            # Diagonalize and get the eigenstates of MHF
            Eeig, Coe = np.linalg.eigh(np.array(MHF))
            Rho = np.zeros((NSP, NSP))
            
            # Update Rho
            for i_gam in range(NSP):
                for i_del in range(NSP):
                    Rho[i_gam, i_del] = sum(Coe[i_gam,i]*Coe[i_del, i] for i in list_A)
            
            # Get new Energies and calculate convergence
            newE     = np.array([e for e in Eeig])
            diff     = np.sum(np.abs(newE-oldE)/NSP)
            oldE     = newE
            i_count += 1
            
            Coe_list_A = []
            for i in list_A:
                for row_idx, row in enumerate(Coe):
                    if row[i] != 0:
                        Coe_list_A.append(row_idx)
            
        # Return binding energies and energy level lists.
        return sum(oldE[i] for i in Coe_list_A), oldE
        
    # Single Particle Hamiltonian (SPH)
    def onebody(self, SP_ind):
        
        pm    = (1 if self.dict_SP[SP_ind][4]==-0.5 else -1)
        V_0   = -34*(1+pm*0.86*(self.N-self.Z)/(self.N+self.Z))
        
        # Base harmonic oscillator energy
        e_gap = 41.5*np.power(self.A_c, -0.333)
        n, l, j = self.dict_SP[SP_ind][0], self.dict_SP[SP_ind][1], self.dict_SP[SP_ind][2]
        E_ho =  e_gap*(2*n + l + 3/2)
        
        
        # Spin-Orbit term
        V_so = -32*np.power((self.A/(self.A_c+2)),0.333)
        E_so = V_so * (j*(j+1) - l*(l+1) - 3/4)
        
        
        return E_ho + E_so
    
    


# ## Part 3: Set the initial value of me2b for code
# 
# In this block, it will generate initial ME2B as a list of (a,b,c,d,v), where using the method $\langle j_1 j_2 |V| j_1 j_2\rangle = BE(CS\pm 2 , j_1 j_2, J^\pi = 0^+) - BE(CS) -  \epsilon^{HF}_{j_1} - \epsilon^{HF}_{j_2}$, as given as <Shell Model from a Practitioner’s Point of View> by Hubert Grawe. 
# 
# `_load_BE` will load the binding energy per nucleon from `mass_1.mas20.txt` file
# 
# `_get_BE` will get the binding energy for nuclei.
# 
# `generate_me2b` will generate the me2b values from the binding energies and assign random value centered at 0. 

class gen_me2b:
    
    def __init__(self):
        self.dict_BE = self._load_BE('mass_1.mas20.txt')
        
    def _load_BE(self, filename):
        dict_BE ={}
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith('#') or len(line.strip()) == 0: continue
                try:
                    N = int(line[4:9])
                    Z = int(line[9:14])
                    A = int(line[14:19])
                    BE_A = float(line[54:68])

                    total_BE = BE_A*A
                    dict_BE[(A,Z)] = total_BE
                except (ValueError, IndexError):
                    continue
        return dict_BE

    def _get_BE(self, A, Z):
        return -1*self.dict_BE.get((A,Z), None)

    def generate_me2b(self, dict_SP, list_TB):
        e_gap = 41.5 * np.power(12, -1/3) * 1000
        A_core = 12
        Z_core = 6

        A_gap = {
            1.5: -1,
            0.5: 1
        }

        Z_gap = {
            0.5:  0,
            -0.5: 1
        }

        Matrx_me2b = np.zeros((16, 16, 16, 16))
        list_me2b  = []


        for i in list_TB:
            for j in list_TB:
                if dict_SP[i[0]][3] + dict_SP[i[1]][3] != 0 or dict_SP[j[0]][3] + dict_SP[j[1]][3] != 0:
                    continue
                # tz conservation
                if dict_SP[i[0]][4] + dict_SP[i[1]][4] != dict_SP[j[0]][4] + dict_SP[j[1]][4]:
                    continue

                # Compute TBME with specific logic
                ji, jf = dict_SP[i[0]][2], dict_SP[j[0]][2]
                
                if i == j :
                    if dict_SP[i[0]][1] == 1 and dict_SP[j[0]][1] == 1:
                        try:
                            # Special calculation when i is the same as j
                            tz1, tz2 = dict_SP[i[0]][4], dict_SP[i[1]][4]

                            BECS     = self._get_BE(A_core, Z_core)
                            BECSpm2  = self._get_BE(A_core+2*A_gap.get(ji), Z_core+A_gap.get(ji)*(Z_gap.get(tz1)+Z_gap.get(tz2)))
                            BEj1     = self._get_BE(A_core+A_gap.get(ji), Z_core+A_gap.get(ji)*Z_gap.get(tz1))
                            BEj2     = self._get_BE(A_core+A_gap.get(ji), Z_core+A_gap.get(ji)*Z_gap.get(tz2))

                            me2b_value = BECS-BECSpm2+(BEj1-BECS)+(BEj2-BECS)

                            me2b_value = -1*me2b_value/1000.
                        except (IndexError, TypeError):
                            me2b_value = 0
                        

                    elif dict_SP[i[0]][1] == 0 and dict_SP[j[0]][1] == 0:
                        try:
                            tz1, tz2 = dict_SP[i[0]][4], dict_SP[i[1]][4]
                            BECS     = self._get_BE(4,2)
                            BECSm2   = self._get_BE(2,2-(Z_gap.get(tz1)+Z_gap.get(tz2)))
                            BEj1     = self._get_BE(3,2-Z_gap.get(tz1))
                            BEj2     = self._get_BE(3,2-Z_gap.get(tz2))
                            
                            me2b_value = BECS-BECSm2+(BEj1-BECS)+(BEj2-BECS)

                            me2b_value = -1*me2b_value/1000.
                        except (IndexError, TypeError):
                            me2b_value = 0
                            
                else:
                    # Random value centered at zero when i is not the same as j
                    if dict_SP[i[0]][1] == dict_SP[j[0]][1] and i!=j:                    
                        if ji == 1.5 and jf == 1.5: me2b_value = 0.34856382
                        elif ji == 1.5 and jf == 0.5: me2b_value = -0.2152312
                        elif ji == 0.5 and jf == 0.5: me2b_value = -0.6254905
                    else: me2b_value = 0

                # Ensure symmetry for (a,b,c,d) and (c,d,a,b)
                Matrx_me2b[i[0]][i[1]][j[0]][j[1]] = me2b_value
                Matrx_me2b[j[0]][j[1]][i[0]][i[1]] = me2b_value


        # Collect TBME values
        for i in list_TB:
            for j in list_TB:
                if Matrx_me2b[i[0]][i[1]][j[0]][j[1]] == 0 : continue
                list_me2b.append((i[0], i[1], j[0], j[1], Matrx_me2b[i[0]][i[1]][j[0]][j[1]]))

        return list_me2b


# ## Part 4: Set a function for optimization
# This part sets a function to be optimized by `scipy.optimize.minimize`. This function will return the error of $(E_C^{th}-E_C^{exp})^2-\sum_{i\neq j}(\Delta^{th}_{i,j}-\Delta^{exp}_{i,j})^2$ Where $\Delta_{i,j}$ is a energy difference between $i$ and $j$, where $i$ and $j$ are $^{4}\text{He}$, $^{12}\text{C}$, and $^{16}\text{O}$.

def minimize_this(x, me2b):
    exp_energies = {
        '4He':-28.30,
        '12C':-92.16,
        '16O':-127.62
    }
        
    HF = HF_Machina()
    
    list_tbme = me2b
    list_input_tbme = []
    
    for i, (a,b,c,d,_) in enumerate(list_tbme):
        list_input_tbme.append((a,b,c,d,x[i]))
    
    e_4He, _ = HF.Solve_HF(4,2,list_input_tbme)
    e_12C, _ = HF.Solve_HF(12,6,list_input_tbme)
    e_16O, _ = HF.Solve_HF(16,8,list_input_tbme)
    
    error = ((e_12C-e_4He)-(exp_energies['12C']-exp_energies['4He']))**2 + \
            ((e_16O-e_12C)-(exp_energies['16O']-exp_energies['12C']))**2 + \
            ((e_16O-e_4He)-(exp_energies['16O']-exp_energies['4He']))**2 + \
            (e_16O-exp_energies['16O'])**2
    
    error = float(error)

    sys.stdout.flush()
    sys.stdout.write("\r"" 4He energy : {0:15.3f}\n".format(e_4He))
    sys.stdout.write("\r"" 12C energy : {0:15.3f}\n".format(e_12C))
    sys.stdout.write("\r"" 16O energy : {0:15.3f}\n".format(e_16O))
    sys.stdout.write("\r"" error      : {0:15.3f}\n".format(error))
    sys.stdout.write("\r"" ----------------------------")
    sys.stdout.write("\x1b[1A"*4)

    
    return error


# ## Main

if __name__=='__main__':
    HF = HF_Machina()
    
    gen_me2b = gen_me2b()
    tbme0 = gen_me2b.generate_me2b(HF.dict_SP, HF.list_TB)
    
    x0 = [t[4] for t in tbme0]
    
    e_12O, e_list_12O = HF.Solve_HF(16,8,tbme0)
     
    result1 = minimize(minimize_this, x0, method='SLSQP', tol=1e-30, args=(tbme0))
    tbme_aftr_opt = []

    for i, (a,b,c,d,_) in enumerate(tbme0):
        tbme_aftr_opt.append((a,b,c,d,result1.x[i]))

    e_4He, e_list_4He = HF.Solve_HF(4,2,tbme_aftr_opt)
    e_12C, e_list_12C = HF.Solve_HF(12,6,tbme_aftr_opt)
    e_16O, e_list_16O = HF.Solve_HF(16,8,tbme_aftr_opt)
    
    sys.stdout.write(5*"\x1b[1B")
    print("\r"'***************************************\n')
    print("\r"'---------Minimization Finished---------\n')
    print("\r"'---------------------------------------\n')
    print("\r"'   Energy Levels after Fitting (MeV)  :\n')
    print("\r"'---------------------------------------\n')
    print("\r"'  4 He \t\t 12 C \t\t 16 O \n')
    print("\r"'---------------------------------------\n')
    for i in range(len(e_list_4He)):
        ie_4He = ("{0:5.2f}\t".format(e_list_4He[i]) if e_list_4He[i]<0 else "    \t")
        ie_12C = ("{0:5.2f}\t".format(e_list_12C[i]) if e_list_12C[i]<0 else "      \t")
        ie_16O = ("{0:5.2f}\t".format(e_list_16O[i]) if e_list_16O[i]<0 else "      \t")
        print("\r""  {0:s}\t{1:s}\t{2:s}\n".format(ie_4He, ie_12C, ie_16O))

