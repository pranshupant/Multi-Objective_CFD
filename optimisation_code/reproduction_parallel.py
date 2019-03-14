from constants import Cmax, Cmin, nPop
from airfoil_Class import airfoil #baby_airfoil
import random
import copy
from multiprocessing import Pool
import numpy as np
import itertools

def p_run(progeny, j):

    progeny[j].cfd()
    #Airfoil.append(progeny[j])

def reproduction(Airfoil, gen, sigma, x, s):  
                     
    costs = []

    child = []
    
    for t in range(len(Airfoil)):
        costs.append(Airfoil[t].cost)

    BestCost = max(costs)

    WorstCost = min(costs)

    ratio = (Airfoil[x].cost - WorstCost)/(BestCost - WorstCost)
    C = int(Cmin + (Cmax - Cmin)*ratio)
    #print(Airfoil[x].cost)
    #print(C)

    if C > 0:

        progeny = []
        P = []
       
        for j in range(C):
            
            p = copy.deepcopy(Airfoil[x])
            p.copy_mutate(gen, s[0])
            progeny.append(p)

            progeny[j].new(sigma)
            progeny[j].bspline()
            progeny[j].write()
            progeny[j].savefig()
            progeny[j].show(gen, s[0])
            progeny[j].camber(gen, s[0])
            
            P.append(s[0])    
            s[0] += 1
            

        
        for i in range(C):
            child.append(i)

        y = Pool(4)
        #y.starmap(p_run, range(C))
        y.starmap(p_run, zip(itertools.repeat(progeny), child))

        
        y.close()
        
        y.join()        

        for j in range(len(P)):

            #progeny[j].xFoil()
            #progeny[j].cfd()
            #progeny[j].error('s1223.dat')
            #print(progeny[j].cost)
            progeny[j].cost = np.loadtxt('Results_CFD/Generation_%d/cost-%d'%(gen,P[j]))
            #progeny[j].max_Camber = np.loadtxt('Results_CFD/Generation_%d/camber-%d'%(gen,P[j]))            
            Airfoil.append(progeny[j])