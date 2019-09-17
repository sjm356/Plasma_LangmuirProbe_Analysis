# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
'''
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
'''
'''
v1.0: transfering from Noh's code with MATLAB to PHYTHON
      In Savitzky-Golay filtering, Using Phython's signal module. it has different method with the way Noh's code, signal coefficients.

v2.0(expected): Changing the method to get the plasma potential. the cross point of two linear fitting is determined with Plasma Potential
      Due to unfiltered IV signal....
      - Changing the range of linear fitting for getting Te. Vf ~ Vf + (Vp-Vf) / 3
      - adding method to change NaN to zero number (np.nan_to_num) 
      
v2.1: Changing voltage range for linear fitting to get the Te
      Start Voltage = Vf + (Vp - Vf) / 2
      End Voltage = Vf + (Vp - Vf) * 3 / 2
v2.2: Adding floating potential data to analyzed data
      
v3.0(expected): Connecting EEPF data between SG data and BW data at similar data point. low energy -> SG, high energy -> BW
      
!!!!!!!!!!!!!! Before using this code, please check raw data(IV data) to confirm the expected range of the ion current !!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!! Check the sign of the current

'''
## import modules
from tkinter import *
from tkinter import filedialog
from scipy.optimize import curve_fit
from scipy import signal
from scipy.special import erf
import numpy as np
import matplotlib.pyplot as plt
import os
import re

## Physical constants
# Electron mass
from scipy.constants import m_e
# Electricity
from scipy.constants import e

np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.10f}".format(x)})

###############################################################################
############################ Langmuir probe information #######################
###############################################################################
# unit : m
L = 2.5e-3
R = 2.5e-4
Ap = L * 2 * np.pi * R + np.pi * R ** 2

# Loading IV data
# root = Tk()
file_path = filedialog.askdirectory(initialdir="C:\\Users\\jmsong\\Documents\\00_NFRI\\00_연구관리\\00_실험데이터\\00_LP", title="Select IV data Folder")
file_path_SG = filedialog.askopenfilename(initialdir="C:\\Users\\jmsong\\Documents\\00_NFRI\\00_연구관리\\00_실험데이터\\00_LP", title="Select SG parameter Folder")
root = Tk()

# File list in selected folder
file_list = os.listdir(file_path)
new_file_list = []
file_list.sort()

# Defining new directory for new output file
new_file_path = file_path + '/EEDF analysis'
# Making new folder in directory
if not os.path.exists(new_file_path):
    os.mkdir(new_file_path)

# Defining new variables
V = []
I = []
dV_new = 0.2
# IV[0] = Voltage
# IV[1] = Current
# IV[2] = Only electron current
# IV[3] = Optimized dI/dV, Savitzky-Golay filter
# IV[4] = Optimized d2I/dV2, Savitzky-Golay filter
# IV[5] = Optimized dI/dV, Blackman filter
# IV[6] = Optimized d2I/dV2, Blackman filter


new_file_name_Total = "V2_Plasma"
new_file_Total = open(new_file_path + "/" + new_file_name_Total + ".dat",'w')
new_file_Total.write(f'Filename \t Vf(IV) \t Vp(IV) \t Vp(SG) \t ne(IV) \t Te \n')

## Get V-I data from files
if len(file_list) == 0:
    print('!!No data files in this directory')
else:
    for iiii in file_list:
        if iiii == 'EEDF analysis':
            continue
        IV = np.loadtxt(file_path + '/' + iiii, delimiter = '\t')
#######!!!!!!! current + -
        IV[:,1] = IV[:,1]
        n = len(IV)
        h = IV[1,0] - IV[0,0]
        h = round(h, 10)
        
        ## Finding V_f
        for k, V in enumerate(IV):
            if V[1] == 0:
                Vf = float(V[0])
                #print(Vf)
                break
            elif IV[k+1][1] * IV[k][1] < 0:
                Vf = float((IV[k+1][0] + IV[k][0])/2)
                #print(Vf)
                break
        
        ## Fitting and Subratcting Ion current
            # Defining fitting range
        V_ion_1 = -1 * min(abs(IV[:,0] - (Vf - 50))) + (Vf - 50)    #adjusting range
        V_ion_2 = -1 * min(abs(IV[:,0] - (Vf - 20))) + (Vf - 20)    #adjusting range
        V_ion_1 = round(V_ion_1, 10)
        V_ion_2 = round(V_ion_2, 10)
        id_V_ion_1 = np.where(IV == V_ion_1)
        id_V_ion_2 = np.where(IV == V_ion_2)
            # Extracting IV range
        ion_V = IV[int(id_V_ion_1[0]):int(id_V_ion_2[0]),0]
        ion_I = IV[int(id_V_ion_1[0]):int(id_V_ion_2[0]),1]
            # Linear fitting tool
        def func(x, a, b):
            return a*x + b
        popt = curve_fit(func, ion_V, ion_I)
        a = popt[0][0]
        b = popt[0][1]
        I_i = a * IV[:,0] + b
            # Ion current
        I_noni = IV[:,1] - I_i
        
        I_noni = I_noni.reshape((len(I_noni),1))
            # IV curve Subtracted ion current
        IV = np.hstack([IV, I_noni])

        ## Getting lpe for 1st derivative & noise amplitude 
        M = 21
        w = np.blackman(M) / sum(np.blackman(M))
        dI_dV = np.gradient(IV[:,1],h)
        dI_dV = np.nan_to_num(dI_dV)
        mov_avg = signal.filtfilt(w, 1, dI_dV)
        id_Vp = np.nanargmax(mov_avg)
        Vp = float(IV[id_Vp,0])    # Plasma Potential
        Ie = float(IV[id_Vp,2])    # Electron saturation current
        Noise_amp_abs = float((max(I_noni[int(id_V_ion_1[0]):int(id_V_ion_2[0])]-min(I_noni[int(id_V_ion_1[0]):int(id_V_ion_2[0])]))) / 2)
        Noise_amp = Noise_amp_abs / float(I_noni[id_Vp])
        Noise = Noise_amp * (np.random.rand(n,1) - 0.5)
        
        ## Finding Electron Temperature, Te
        # Te is determined by floating potential and the difference with plasma potential
        ln_I = np.log(I_noni[:,0])
        low_lim = Vf + (Vp - Vf) / 2
        up_lim = Vf + (Vp - Vf) / 3 *2
        IV_low_lim = IV[:,0] - low_lim
        IV_up_lim = IV[:,0] - up_lim
        #id_low_lim = np.where(abs(IV[:,0] - low_lim) == (abs(IV[:,0] - low_lim)).min())
        id_low_lim = np.argmin(abs(IV[:,0] - low_lim))
        #id_up_lim = np.where(abs(IV[:,0] - up_lim)  == (abs(IV[:,0] - up_lim)).min())
        id_up_lim = np.argmin(abs(IV[:,0] - up_lim))
        Te_V = IV[int(id_low_lim):int(id_up_lim),0]
        Te_I = ln_I[int(id_low_lim):int(id_up_lim)]
        Te_I = np.nan_to_num(Te_I)
        popt_Te = curve_fit(func, Te_V, Te_I)
        # popt_Ie = curve_fit(func, 1)
        a_Te = popt_Te[0][0]
        b_Te = popt_Te[0][1]
        I_Te = a_Te * IV[:,0] + b_Te    # for plotting
        
            # Calculating Electron temperature(eV) & Electron density(cm-3), ne from IV curve
        Te = 1 / a_Te  
        ne = Ie / e / Ap / np.sqrt(Te * e / 2 / np.pi / m_e) / 1000000
        #print(Te)
        
        ## Ideal IV-curve
        noise_IV = np.zeros((n,2), dtype = float)
        idl_IV_d0 = np.zeros((n,2), dtype = float)
        idl_IV_d1 = np.zeros((n,2), dtype = float)
        idl_IV_d2 = np.zeros((n,2), dtype = float)
        idl_IV_d0[:,0] = IV[:,0]
        idl_IV_d1[:,0] = IV[:,0]
        idl_IV_d2[:,0] = IV[:,0]
        
        for i in range(1,n+1):
            if IV[i-1,0] <= Vp:
                idl_IV_d0[i-1,1] = np.exp((IV[i-1,0] - Vp) / Te)
                idl_IV_d1[i-1,1] = np.exp((IV[i-1,0] - Vp) / Te ) / Te
                idl_IV_d2[i-1,1] = np.exp((IV[i-1,0] - Vp) / Te ) / Te ** 2
            else:
                idl_IV_d0[i-1,1] = np.pi ** 0.5 * ((IV[i-1,0] - Vp) / Te) ** 0.5 * erf(((IV[i-1,0] - Vp) / Te) ** 0.5) + np.exp(-1 * ((IV[i-1,0] - Vp) / Te))
                idl_IV_d1[i-1,1] = np.pi ** 0.5 / 2 / ((IV[i-1,0] - Vp) / Te) ** 0.5 / Te * erf(((IV[i-1,0] - Vp) / Te) ** 0.5)
                idl_IV_d2[i-1,1] = 0.5 / ((IV[i-1,0] - Vp) / Te) * (-1 * np.pi ** 0.5 / 2 / ((IV[i-1,0] - Vp) / Te) ** 0.5 * erf (((IV[i-1,0] - Vp) / Te) ** 0.5) + np.exp(-1 * (IV[i-1,0] - Vp) / Te)) / Te ** 2
        
        noise_IV[:,0] = idl_IV_d0[:,0]
        noise_IV[:,1] = idl_IV_d0[:,1] + Noise[:,0]
        
        #######################################################################
        ######### Finding optiminzed parameters for S-G filtering #############
        #######################################################################
        SG_param = np.loadtxt(file_path_SG)
        n_SG_param = len(SG_param)
        SG_param_add = np.zeros(n_SG_param, dtype = float)
        k1 = SG_param[:,0]
        f1 = SG_param[:,2]
            # 1st derivative
        noise_dIdV = np.gradient(noise_IV[:,1] / dV_new)
        
        bb = 0

        for k,f in zip(k1, f1):
            k = int(k)
            f = int(f)
            g0 = signal.savgol_coeffs(f, k, deriv=0)
            g1 = signal.savgol_coeffs(f, k, deriv=1) * -1
            Half_win1 = ((f + 1) / 2) - 1
            Half_win1 = int(Half_win1)
            sg0 = np.zeros(int(n - Half_win1 - 1), dtype = float)
            sg1 = np.zeros(int(n - Half_win1 - 1), dtype = float)
            for ii in range(int((f + 1) / 2), int(n - (f + 1) / 2 + 1)):
                sg0[ii-1] = np.dot(g0, noise_dIdV[ii-1-Half_win1 : ii-1+Half_win1+1])
                sg1[ii-1] = np.dot(g1, noise_dIdV[ii-1-Half_win1 : ii-1+Half_win1+1])
                
            sg1 = sg1 / dV_new
            
            sg0_new = signal.savgol_filter(noise_dIdV, f, k, deriv = 0)
            sg1_new = signal.savgol_filter(noise_dIdV, f, k, deriv = 1)
            sg2_new = signal.savgol_filter(noise_dIdV, f, k, deriv = 2)
            sg1_new = sg1_new / dV_new
            
            SG_param_add[bb] = (sum((idl_IV_d2[0:len(sg1),1] - sg1) ** 2 / len(sg1)))
            bb = bb + 1
            
        SG_param_new = np.zeros((SG_param.shape[0],SG_param.shape[1]+1))
        SG_param_new[:,:-1] = SG_param
        SG_param_new[:,-1] = SG_param_add         
        
        id_min_SG_param = np.where(SG_param_new[:,4] == min(SG_param_new[:,4]))
        k1_min = SG_param_new[id_min_SG_param, 0]
        f1_min = SG_param_new[id_min_SG_param, 2]
        
        #######################################################################
        ########################### Blackman window ###########################
        #######################################################################
        M = list(range(1,31))
        n_M = len(M)
        BW_param1 = np.zeros((n_M,n_M), dtype = float)
        BW_param2 = np.zeros((n_M,n_M), dtype = float)
        mov_avg = np.zeros((n,2), dtype = float)
        for iii in M:
            for jjj in M:
                w1 = np.blackman(iii) / sum(np.blackman(iii))
                mov_avg[:,0] = signal.filtfilt(w1, 1, noise_dIdV, method="gust")
                noise_dIdV2 = np.gradient(mov_avg[:,0]) / dV_new
                w2 = np.blackman(jjj) / sum(np.blackman(jjj))
                mov_avg[:,1] = signal.filtfilt(w2, 1, noise_dIdV2, method="gust")
                param2 = ((np.log(abs(idl_IV_d2[:,1])) - np.log(abs(mov_avg[:,1]))) ** 2 / len(mov_avg[:,1])) * ((idl_IV_d2[:,1] - mov_avg[:,1]) **2 / len(mov_avg[:,1]))
                param3 = (np.log(abs(idl_IV_d2[:,1])) - np.log(abs(mov_avg[0:,1]))) ** 2 / len(mov_avg[:,1])
                aa2 = np.where(np.isinf(param2) == True)
                aa3 = np.where(np.isinf(param3) == True)
                param2 = np.delete(param2, list(aa2[0]))
                param3 = np.delete(param3, list(aa3[0]))
                bb2 = np.where(np.isnan(param2) == True)
                bb3 = np.where(np.isnan(param3) == True)
                param2 = np.delete(param2, list(bb2[0]))
                param3 = np.delete(param3, list(bb3[0]))
                BW_param1[iii-1,jjj-1] = sum(param2)
                BW_param2[iii-1,jjj-1] = sum(param3)
        
        id_BW_param1_zero = np.where(BW_param1 == 0)
        id_BW_param2_zero = np.where(BW_param2 == 0)
        for aaa2, bbb2 in zip(list(id_BW_param1_zero[0]), list(id_BW_param1_zero[1])):
            BW_param1[aaa2, bbb2] = 1000
        for aaa3, bbb3 in zip(list(id_BW_param2_zero[0]), list(id_BW_param2_zero[1])):
            BW_param2[aaa3, bbb3] = 1000
        id_min_BW_param1 = np.where(BW_param1 == BW_param1.min())
        id_x_min_BW_param1 = id_min_BW_param1[0][0]
        id_y_min_BW_param1 = id_min_BW_param1[1][0]
        id_min_BW_param2 = np.where(BW_param2 == BW_param2.min())
        id_x_min_BW_param2 = id_min_BW_param2[0][0]
        id_y_min_BW_param2 = id_min_BW_param2[1][0]
        
        opt_M1_1 = M[id_x_min_BW_param1]
        opt_w1_1 = np.blackman(opt_M1_1) / sum(np.blackman(opt_M1_1))
        opt_M2_1 = M[id_y_min_BW_param1]
        opt_w2_1 = np.blackman(opt_M2_1) / sum(np.blackman(opt_M2_1))
        opt_M1_2 = M[id_x_min_BW_param2]
        opt_w1_2 = np.blackman(opt_M1_2) / sum(np.blackman(opt_M1_2))
        opt_M2_2 = M[id_y_min_BW_param2]
        opt_w2_2 = np.blackman(opt_M2_2) / sum(np.blackman(opt_M2_2))

        #######################################################################
        ###################### IV data smoothing ##############################
        #######################################################################
        # Savitazky-golay method for low energy
        f1_min = int(f1_min)
        k1_min = int(k1_min)
        opt_g0 = signal.savgol_coeffs(f1_min, k1_min, deriv=0)
        opt_g1 = signal.savgol_coeffs(f1_min, k1_min, deriv=1)
        opt_g2 = signal.savgol_coeffs(f1_min, k1_min, deriv=2)
        opt_Half_win1 = ((f1_min + 1) / 2) - 1
        opt_Half_win1 = int(opt_Half_win1)
        opt_sg0 = np.zeros(int(n - opt_Half_win1 - 1), dtype = float)
        opt_sg1 = np.zeros(int(n - opt_Half_win1 - 1), dtype = float)
        opt_sg2 = np.zeros(int(n - opt_Half_win1 - 1), dtype = float)
        for t in range(int((f1_min + 1) / 2), int(n - (f1_min + 1) / 2 + 1)):
            # 0-th derivative (smoothing only)
            opt_sg0[t-1] = np.dot(opt_g0, dI_dV[t-1-opt_Half_win1 : t-1+opt_Half_win1+1])
            # 1-st differential
            opt_sg1[t-1] = np.dot(opt_g1, dI_dV[t-1-opt_Half_win1 : t-1+opt_Half_win1+1])
            # 2nd differential
            opt_sg2[t-1] = 2 * np.dot(opt_g2, dI_dV[t-1-opt_Half_win1 : t-1+opt_Half_win1+1])
        opt_sg1 = opt_sg1 / dV_new  # Turnning differential to derivative
        
        # direct filtering by using Python module
        opt_sg0_new = signal.savgol_filter(dI_dV, f1_min, k1_min, deriv = 0)
        opt_sg1_new = signal.savgol_filter(dI_dV, f1_min, k1_min, deriv = 1)
        opt_sg2_new = signal.savgol_filter(dI_dV, f1_min, k1_min, deriv = 2)
        opt_sg1_new = opt_sg1_new / dV_new

        opt_dI_dV_SG = opt_sg0_new
        opt_dI_dV2_SG = opt_sg1_new
        w_lp = np.blackman(7) / sum(np.blackman(7))
        opt_dI_dV2_SG = signal.filtfilt(w_lp, 1, opt_dI_dV2_SG)
        opt_dI_dV_SG_part = np.zeros((int(n-len(opt_dI_dV_SG)),1), dtype = float)
        opt_dI_dV2_SG_part = np.zeros((int(n-len(opt_dI_dV_SG)),1), dtype = float)
        opt_dI_dV_SG = opt_dI_dV_SG.reshape((len(opt_dI_dV_SG),1))
        opt_dI_dV2_SG = opt_dI_dV2_SG.reshape((len(opt_dI_dV2_SG),1))
        opt_dI_dV_SG = np.vstack((opt_dI_dV_SG, opt_dI_dV_SG_part))
        opt_dI_dV2_SG = np.vstack((opt_dI_dV2_SG, opt_dI_dV2_SG_part))

        # Blackman window method for high energy
        opt_dI_dV_BW = signal.filtfilt(opt_w1_2, 1, dI_dV)
        noise_dI_dV2_BW = np.gradient(opt_dI_dV_BW / dV_new)
        opt_dI_dV2_BW = signal.filtfilt(opt_w2_2, 1, noise_dI_dV2_BW)
        opt_dI_dV_BW = opt_dI_dV_BW.reshape((len(opt_dI_dV_BW),1))
        opt_dI_dV2_BW = opt_dI_dV2_BW.reshape((len(opt_dI_dV2_BW),1))
        
        IV = np.hstack([IV, opt_dI_dV_SG])      #IV[3] dI/dV SG
        IV = np.hstack([IV, opt_dI_dV2_SG])     #IV[4] d2I/dV2 SG
        IV = np.hstack([IV, opt_dI_dV_BW])      #IV[5] dI/dV BW
        IV = np.hstack([IV, opt_dI_dV2_BW])     #IV[6] d2I/dV2 BW
        
        ## Plasma Poetential from smoothed signal
        id_Vp_new = np.argmax(opt_dI_dV_SG)
        Vp_new = IV[id_Vp_new,0] - 6    # Plasma Potential from smoothed IV

        #######################################################################
        ########################### Calculating EEDF ##########################
        #######################################################################
        # d2I/dV2 of Savitzky-Golay filtering
        IV_eedf = np.zeros((n,2), dtype = float)
        IV_eedf[:id_Vp_new, 0] = 2 * m_e / (Ap * e * e) * np.sqrt(2 * e / m_e) * IV[:id_Vp_new, 4] / 100     # d2I/dV2, SG
        IV_eedf[id_Vp_new+1:, 0] = 0
        IV_eedf[:,0] = np.where(IV_eedf[:,0] < 0, 0, IV_eedf[:,0])
        #id_V_no = list(id_V_no[0])
        #id_V_noo = np.where(id_V_no > id_Vp_new - 10)
        
        # d2I/dV2 of Blackman filtering
        IV_eedf[:id_Vp_new, 1] = 2 * m_e / (Ap * e * e) * np.sqrt(2 * e / m_e) * IV[:id_Vp_new, 6] / 100     # d2I/dV2, BW
        IV_eedf[id_Vp_new+1:, 1] = 0
        IV_eedf[:,1] = np.where(IV_eedf[:,1] < 0, 0, IV_eedf[:,1])
        
        eedfdata = np.zeros((n,3), dtype = float)
        eepfdata = np.zeros((n,2), dtype = float)
        eedfdata[:,0] = Vp_new - IV[:,0]
        eedfdata[:,1] = IV_eedf[:,0]    # EEDF from SG
        eedfdata[:,2] = IV_eedf[:,1]    # EEDF from BW
        eepfdata[:,0] = eedfdata[:,1] / np.sqrt(eedfdata[:,0])  # EEPF from SG
        eepfdata[:,1] = eedfdata[:,2] / np.sqrt(eedfdata[:,0])  # EEPF from BW
        
        # Saving & Writing data
        new_file_name = "V2_EEDF_" + str(iiii)
        new_file_name_IV = "V2_IV_" + str(iiii)
        new_file = open(new_file_path + "/" + new_file_name,'w')
        new_file_IV = open(new_file_path + "/" + new_file_name_IV,'w')
        new_file.write(f'Electron_energy \t EEDF_SG \t EEDF_BW \t EEPF_SG \t EEPF_BW \n')
        new_file_IV.write(f'Voltage(V) \t Current(A) \t Electron_current(A) \t dI/dV(SG) \t dI/dV(BW) \t d2I/dV2(SG) \t d2I/dV2(BW) \n')

        for nnn in range(0, len(eedfdata)):
            energy = eedfdata[nnn][0]
            eedf_SG = eedfdata[nnn][1]
            eedf_BW = eedfdata[nnn][2]
            eepf_SG = eepfdata[nnn][0]
            eepf_BW = eepfdata[nnn][1]
            IV_V = IV[nnn][0]
            IV_I = IV[nnn][1]
            IV_eI = IV[nnn][2]
            IV_d1_SG = IV[nnn][3]
            IV_d2_SG = IV[nnn][4]
            IV_d1_BW = IV[nnn][5]
            IV_d2_BW = IV[nnn][6]
            new_file.write(f'{energy} \t {eedf_SG} \t {eedf_BW} \t {eepf_SG} \t {eepf_BW} \n')
            new_file_IV.write(f'{IV_V} \t {IV_I} \t {IV_eI} \t {IV_d1_SG} \t {IV_d1_BW} \t {IV_d2_SG} \t {IV_d2_BW} \n')
            
        new_file_Total.write(f'{iiii} \t {Vf} \t {Vp} \t {Vp_new} \t {ne} \t {Te} \n')
        new_file.close()
        new_file_IV.close()
        
        # Plotting
        #plt.plot(IV[:,0], dI_dV)
        #plt.plot(IV[:,0], IV[:,2])
        #plt.plot(IV[:,0], IV[:,3]) 
        plt.plot(eedfdata[:,0], eedfdata[:,1])
        plt.plot(eedfdata[:,0], eedfdata[:,2])
        #plt.plot(IV[:,0], dI_dV)
        #plt.plot(IV[:,0], Noise)
        #plt.plot(IV[:,0], ln_I)
        #plt.plot(IV[int(id_low_lim[0]):int(id_up_lim[0]),0], Te_I)
        #plt.plot(noise_IV[:,0], noise_IV[:,1])
        plt.yscale('log')
        plt.xlabel("Vlotage")
        plt.ylabel("Current")
        plt.show()
        
        
new_file_Total.close()
