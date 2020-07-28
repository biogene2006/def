# ------------------------------------------------------------
# ------------------------------------------------------------
#                      ESP THORBOT
#                     décembre 2012
# ------------------------------------------------------------
# ------------------------------------------------------------
# auteur : benjamin Chouvion - CAPSULE
# contact : bchouvion@capsule-ea.fr
#           04 84 25 07 00
# ------------------------------------------------------------
# ------------------------------------------------------------

# analyse de la courbe de réponse coté balle après impact de la raquette


## ========================================================================
# Recupère les données enregistrées lors du calibrage
# =========================================================================
"""
if ~exist('rendement_balle.mat') # un calibrage doit avoir été fait
    error('faire calibrage balle d''abord');
end
if ~exist('mLI_balle.mat')
    error('faire calibrage balle d''abord');
end
load('rendement_balle.mat');
load('mLI_balle');
"""
import os.path
import numpy as np

import math
import matplotlib.pyplot as plt

if os.path.isfile('mLI_balle.py'):# recupere précédentes valeurs
    from mLI_balle import mLI_b 
else :
    print ('faire calibrage balle d abord ou place le fichier dans le repertoire courant')
    exit()
if os.path.isfile('rendement_balle.py'):# recupere précédentes valeurs
    from rendement_balle import rendement_b 
else :
    print ('faire calibrage balle dabord')
    exit()
    
import scipy.io
# open file
from pathlib import Path
import numpy as np
a = np.array([])
t = np.array([])
data_folder = Path("source_data/text_files/")
p = Path('.')
print (p.absolute())
file_to_open = p.absolute() / "impact_nd.txt"

## ========================================================================
# cherche le nombre de tests et les intervales d'intérêt
# =========================================================================
global temps
global raquette
global balle
global accelero
a = balle
t=temps/1000      

t= t-t[0]
#ang = a; # angle en radian
ang=np.radians(-a); # angle en radian

# fait la moyenne des n premieres valeurs pour trouver celle au repos
ang0 = (np.mean(ang[0:1000]))
# soustrait à toutes les autres pour mettre le repos sur angle=0
ang = np.subtract(ang, ang0)
angMin = np.radians(30); # angle en radian
angMax = np.radians(150); # angle en radian
#print("max admis", angMax)
#print ("max = ",max(ang))
#print("min admis", angMin)
#print ("min = ",min(ang))

a = 0
#if max(ang)< angMax:
#    print('changer les valeurs de angMin et angMax car je vais calculer l''energie sur des points après le premier pic (non recommandé)')

ind=[]

compt = 1 
ind.append(0)

for i  in ang : 
    if (i>angMin and i<angMax) : 
        ind.append(compt)
#        print (ind)
    compt+=1
# temps entre 2 mesures
indic = np.asarray(ind)
    
indSub = indic[1:] - indic[:-1]

#for i in indSub:
#    if i>1 :
#        print ('igrand',i)
#print ("ind " , indSub)    

## correspond à un debut d'impact (nouveau test) si le nombre entre 2 mesures est grand

tpsEntreMesures = 3 
indTestDebutB = []
compt  = 0
for i in range (len(indic[:-1])) :
    if indSub[compt] > (tpsEntreMesures/(t[2]-t[1])) :
        indTestDebutB.append(indic[i+1])
    compt+=1
   
nbrTests = len (indTestDebutB)

vitesseBallekmh=[]
indTestFin= np.arange(nbrTests)
for ii in range (nbrTests) :
    indTestFin[ii] = indTestDebutB[ii]
    compteurLoc=0
    while (compteurLoc<100) and (ang[indTestFin[ii]]<angMax): # max 1000 points
        indTestFin[ii]+=1;
        compteurLoc=compteurLoc+1; 

Energie_balle=np.zeros(nbrTests) # initialisation
Energie_balle_cplt=np.zeros([nbrTests*len(mLI_b)]) # initialisation
for n in range (nbrTests):
    print("n=,",n)
#    print("indTestFin=,",indTestFin[n])
#    print("indTestDebutB=,", indTestDebutB[n])
    t_test=t[int(indTestDebutB[n]):int(indTestFin[n])];
    ang_test=ang[int(indTestDebutB[n]):int(indTestFin[n])]; 
#    # recale de telle manière que le premier temps soit=0
    t_test=t_test-t_test[0];
    
#    #----- interpolation lineaire de la mesure sur ce petit intervalle
    p=np.polyfit(t_test,ang_test,1)
    mesRegress = p[0]*t_test+p[1]

#    plot(t_test,rad2deg(mesRegress),'-c'); # trace l'interpolation linéaire
#    
#    # la vitesse est egale au coefficient directeur de la droite (provenant de l'interpolation linéaire)
    omega =p[0]# vitesse en rad/s
    vitesse = omega*0.78# en lineaire (m/s) pour information
    vitesseBallekmh.append(vitesse*3.6)
    theta = np.mean(ang_test)

#    print ("Test numero =", n , " / vitesse balle a = " , theta ," /degres apres impact=", vitesse*3.6 , "km/h = " )
#    #----- énergie après impact    
    for ii in range (len(mLI_b)):
        g = 9.81
        mL=mLI_b[ii][0]
        I=mLI_b[ii][1]
#        print ("ml = ", mL , " I = ", I)
        Energie_balle_cplt[n*(len(mLI_b)) + (ii) ] = (1/2*I*math.pow(omega,2)+mL*g*(1-math.cos(theta)))
#Energie_balle
#
        
#Energie_balle=np.mean(Energie_balle_cplt);
#print('*******************************************************')
#print('Energie (J) transmise dans la balle (sans perte) est :')
#print('eme colonne pour test d''impact numero j')
#print(Energie_balle)        
#plt.figure(1)
#plt.figure( figsize=(18, 16))
##plt.gcf().subplots_adjust(wspace = 0, hspace = 4)
##    plot(t(indTestDebutB(ii):indTestFin(ii)),rad2deg(ang(indTestDebutB(ii):indTestFin(ii))),'.r');
#
#plt.subplot(311)
#plt.ylabel('teta')
#plt.plot(t , ang,'k' )
#for ii in range(nbrTests):
#    plt.plot(t[int(indTestDebutB[ii]):indTestFin[ii]],ang[int(indTestDebutB[ii]):indTestFin[ii]],'.r');
#plt.plot(t[indTestDebutB],ang[indTestDebutB],'.g')
#    
#plt.subplot(312)
#plt.plot(t_test,mesRegress,'-c');
#plt.subplot(313)
#plt.plot(t[indic] , ang[indic],'r' )
#
#plt.show()