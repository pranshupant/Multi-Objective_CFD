from constants import maxIt, Gen0, sigma_final, sigma_initial, exponent, nPop, gen
from airfoil_Class import airfoil#, baby_airfoil
import subprocess as sp
import os
import numpy as np
#from reproduction import reproduction
from NS_reproduction import *
from multiprocessing import Pool
import pickle
import matplotlib.pyplot as plt
import itertools

Airfoil = []
#A = []
st = [0]
s = np.array(st)
Af = []
it = []

for i in range(Gen0):
    it.append(i)
#gen = 0

if gen == 0:
    print('Generation-%d STARTED'%gen)
    for i in range(Gen0):

        Airfoil.append(airfoil(0,i))
        Airfoil[i].ctrlPoints()
        Airfoil[i].bspline()
        Airfoil[i].write()
        Airfoil[i].savefig()
        Airfoil[i].show(gen, i)
        Airfoil[i].camber(gen, i)

    def run(i):
        Airfoil[i].cfd()
        #print(Airfoil[i].cost)
        #Airfoil[i].camber()
        #np.savetxt('../../cost-%d'%i, Airfoil[i].cost)

    y = Pool(5)
    #y.starmap(run, zip(itertools.repeat(Airfoil), it))
    y.map(run, range(Gen0))
    
    y.close()
    
    y.join()

    for i in range(Gen0):
        Airfoil[i].cost = np.loadtxt('Results_CFD/Generation_0/cost-%d'%i)


pickle_out = open("pickle/gen-%d.pickle"%gen, "wb")
pickle.dump(Airfoil, pickle_out)
pickle_out.close()

if gen == 0:
    gen = 1
    
if __name__ == "__main__":

    while gen < maxIt:

        G = np.linspace(0, gen, num=gen+1)

        print('Generation-%d STARTED'%gen)

        ng = gen - 1

        pickle_in = open("pickle/gen-%d.pickle"%ng, "rb")
        Airfoil = pickle.load(pickle_in)

        Cost0 = []
        Cost1 = []
        Cost2 = []

        for j in range(len(Airfoil)):
            Cost0.append(Airfoil[j].cost)
            Cost1.append(-Airfoil[j].max_Camber)
            Cost2.append(Airfoil[j].max_Camber)
        print('******')
            

        plt.scatter(Cost2,Cost0,s=5,c='black', label = 'Total Population')
	    #plt.scatter(Cost[0],Cost[1],s=3.5,c='blue')
        plt.ylim(-15, 125)
        plt.xlim(-5, 15)
        plt.ylabel('L/D')
        plt.xlabel('Max Camber')
        plt.savefig('Pics/%i.svg'%(gen))

        r = []
        NDSa = []
        total = 0
        NDSa = Rank_Assign(Airfoil, Cost1, Cost0)#Cost0, Cost1 -original
        l = len(NDSa)
        for i in range(len(NDSa)):
            count = len(NDSa[i])
            total += count
            r.append(total)

        sigma = (((maxIt - float(gen-1))/maxIt)**exponent)*(sigma_initial - sigma_final) + sigma_final
        
        Airfoil.sort(key = lambda Airfoil: Airfoil.rank)
        
        #if(gen == 1):

            #A.append(Airfoil[0].cost)

        f = open('Results_CFD/Generation_%d/Ranks.txt'%ng, 'w')

        f.write('GENERATION-%d'%gen)
        f.write('\n')

        for i in range(len(Airfoil)):
            f.write('-----------\n')
            f.write(str(Airfoil[i].rank))
            f.write('\n')
            f.write('-----------\n')
            f.write(str(Airfoil[i].cost))
            f.write('\n')
            f.write(str(Airfoil[i].max_Camber))
            f.write('\n')
            f.write('\n')   

        f.close()   

        for i in range(len(Airfoil)):
            print('-----------')
            print(Airfoil[i].rank)
            print('-----------')
            print(Airfoil[i].cost)
            print(Airfoil[i].max_Camber)
            print('')           

        del Airfoil[nPop:]
               
        S_Cost0 = []
        S_Cost1 = []

        for j in range(len(Airfoil)):
            S_Cost0.append(Airfoil[j].cost)
            S_Cost1.append(Airfoil[j].max_Camber)

        plt.scatter(S_Cost1,S_Cost0,s=5,c='red', label = 'Selected Population')
	    #plt.scatter(Cost[0],Cost[1],s=3.5,c='blue')
        plt.ylim(-15, 125)
        plt.xlim(-5, 15)
        plt.legend(loc = 'best')
        plt.savefig('Pics/%i.svg'%(gen))
        plt.close()             
    
        for k in range(nPop):
            Airfoil[k].copy(gen, s[0])
            Airfoil[k].copy_Results(gen, s[0])
            Airfoil[k].show(gen, s[0])
            Airfoil[k].camber(gen, s[0])
            #print(Airfoil[k].cost)
            s[0] += 1 

        '''ranks = []
        k = []
        total = nPop - 1

        x = []
        for i in range(nPop):
            x.append(i)

        for t in range(len(Airfoil)):
            ranks.append(Airfoil[t].rank)

        for i in range(len(Airfoil)):
            
            WorstRank = max(ranks)
            ratio = (WorstRank - Airfoil[i].rank)/(WorstRank)
            C = int(Cmin + (Cmax - Cmin)*ratio) 
            k.append(total) 
            total += C

        y = Pool(2)
        y.starmap(reproduction, zip(itertools.repeat(Airfoil), itertools.repeat(sigma), itertools.repeat(gen), x, k))

        y.close()
        
        y.join()'''              
        for x in range(len(Airfoil)):
            reproduction(Airfoil, gen, sigma, x, s) #(Airfoil, gen, sigma, x, s, r, l)    
        
        #A.append(Airfoil[0].cost)

        pickle_out = open("pickle/gen-%d.pickle"%gen, "wb")
        pickle.dump(Airfoil, pickle_out)
        pickle_out.close()
                
        gen += 1 
        s[0] = 0 

        '''T = len(G) - len(A)

        if T > 0:   
            A.reverse()
            for i in range(T):
                A.append(0)
            A.reverse()

        plt.plot(G, A,'k',linewidth=1.5,label='Max Cost')
        plt.xlabel('Generation')
        plt.ylabel('Cost')
        plt.legend(loc='best')
        plt.axis([0, 50, 0, 200])
        plt.title('IWO Convergence')
        plt.savefig('IWO_convergence_%d.svg'%gen, bbox_inches = 'tight')
        plt.close()'''
       
    print("OPTIMISATION COMPLETE")