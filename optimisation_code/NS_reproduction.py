from constants import Cmax, Cmin, nPop
from airfoil_Class import airfoil #baby_airfoil
import random
import copy
from NSGA2 import *
import numpy as np
from multiprocessing import Pool
import itertools


def p_run(progeny, j):

    progeny[j].cfd()

def Rank_Assign(Airfoil, Cost0, Cost1):

    #Cost=Airfoil[0].Lists(Airfoil,1)
    NDSa=fast_non_dominated_sort(Cost0,Cost1)
    #print(NDSa)
    CDv=[]
    for i in range(0,len(NDSa)):
    	CDv.append(crowding_distance(Cost0,Cost1,NDSa[i][:]))
    Rank_List=[]
    for i in range(0,len(NDSa)):
        NDSa2 = [index_of(NDSa[i][j],NDSa[i] ) for j in range(0,len(NDSa[i]))]
        front22 = sort_by_values(NDSa2[:], CDv[i][:])
        front = [NDSa[i][front22[j]] for j in range(0,len(NDSa[i]))]
        front.reverse()
        #print(front)
        for value in front:
    	    Rank_List.append(value)
    	    if(len(Rank_List)==len(Airfoil)):
    	    	break
        if (len(Rank_List) == len(Airfoil)):
    	    break
    #Sorted_Airfoil = []
    for i in range(len(Airfoil)):
    	Airfoil[Rank_List[i]].rank = i
        #Sorted_Airfoil.append(Airfoil[Rank_List[i]])

    return NDSa

def reproduction(Airfoil, gen, sigma, x, s): #(Airfoil, gen ,sigma, x, s, r, l): 
                     
    #costs = []
    child = []
    ranks = []
    
    for t in range(len(Airfoil)):
        #costs.append(Airfoil[t].cost)
        ranks.append(Airfoil[t].rank)

    WorstRank = max(ranks)
    ratio = (WorstRank - Airfoil[x].rank)/(WorstRank)
    C = int(Cmin + (Cmax - Cmin)*ratio)

    print(C)

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

        y = Pool(5)
        y.starmap(p_run, zip(itertools.repeat(progeny), child))

        
        y.close()
        
        y.join()       

        #for j in range(C):
        #    progeny[j].cfd()       


        for j in range(len(P)):

            #progeny[j].xFoil()
            #progeny[j].camber(gen, s[0])
            #progeny[j].mean_camber(gen, s[0])
            #print(progeny[j].cost)
            #progeny[j].cfd()
            progeny[j].cost = np.loadtxt('Results_CFD/Generation_%d/cost-%d'%(gen,P[j]))
            Airfoil.append(progeny[j])