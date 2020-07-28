# -*- coding: utf-8 -*-
# coding: utf-8

print ("start")

# ------------------------------------------------------------
# ------------------------------------------------------------
#                      ESP THORBOT
#                     decembre 2012
# ------------------------------------------------------------
# ------------------------------------------------------------
# auteur : benjamin Chouvion - CAPSULE
# contact : bchouvioncapsule-ea.fr
#           04 84 25 07 00
# ------------------------------------------------------------
# ------------------------------------------------------------

# analyse de la courbe de reponse cote raquette apres impact


## ========================================================================
# Recupere les donnees enregistrees lors du calibrage
# =========================================================================
print ("start ENERGIE RAQUETTE")
"""
if exist('rendement_raquette.mat','file')
    disp('*******************************************************')
    error('faire calibrage raquette d''abord');
end
if exist('mLI_raquette.mat','file')
    error('faire calibrage raquette d''abord');
end
load('mLI_raquette');
load('rendement_raquette');
load('donneesGeomRaquette');
""" 
import os.path
import numpy as np

import math
# sphinx_gallery_thumbnail_number = 3
import matplotlib.pyplot as plt

if os.path.isfile('mLI_raquette.py'):# recupere precedentes valeurs
    from mLI_raquette import mLI_r 
else :
    print ('faire calibrage Raquette dabord')
#    exit()
if os.path.isfile('rendement_raquette.py'):# recupere precedentes valeurs
    from rendement_raquette import rendement_r 
else :
    print ('faire calibrage Raquette dabord')
#    exit()
    
## ========================================================================
# cherche le nombre de tests et les intervales d'interet
# =========================================================================
a = np.array([])
t = np.array([])

global temps
global raquette
global balle
global accelero

a = raquette
t = temps/1000
ang=np.radians(-a); # angle en radian

# fait la moyenne des n premieres valeurs pour trouver celle au repos
ang0 = (np.mean(ang[0:1000]))

# soustrait a toutes les autres pour mettre le repos sur angle=0
ang = np.subtract(ang, ang0)
angMin = np.radians(-30); # angle en radian
angMax = np.radians(-20); # angle en radian

a = 0
if max(ang)< angMax:
    print('changer les valeurs de angMin et angMax car je vais calculer lenergie sur des points apres le premier pic (non recommande)')

# ------ pour chercher les pics d'impact
# selectionne que les pics reels (et non les rebonds)
from detect_peaks import detect_peaks
indexes = detect_peaks(ang, mph=0.3 , mpd=1000)

ind=[]
from detect_peaks import detect_peaks
indPic = detect_peaks(ang, mph=0.087 , mpd=1000)
compt = 1
ind = indPic 
ind = np.insert(ind , 0 ,0 )

#% cherche ceux pour lesquels il n'y a pas de pics juste avant
    
indSub = ind[1:] - ind[:-1]
# pas de pics si le nb de points entre 2 max consecutifs est grand
periode = indSub*(t[2]-t[1])
tpsEntreMesures = 2 
ind = np.delete(ind,0)
indPic = ind[np.where((periode)>tpsEntreMesures)]
nbTestRaq=len(indPic);

#plt.ion()
##plt.figure(1)
#plt.figure( figsize=(8, 6))
##plt.gcf().subplots_adjust(wspace = 0, hspace = 4)
##    plot(t(indTestDebutB(ii):indTestFin(ii)),rad2deg(ang(indTestDebutB(ii):indTestFin(ii))),'.r');
#
#plt.ylabel('teta')
#plt.plot(t , ang,'k' )
#plt.plot(t[indPic] , ang[indPic],'ro')
#plt.draw()

indTestDebut = []
indTestFin = []
compt  = 0
for i in range (nbTestRaq) :
#    print ("pour le test n= ",i)
    indTestFin.append(indPic[i])
    while ang[indTestFin[i]] > angMax : 
        indTestFin[i] = indTestFin[i] - 1 
    indTestDebut.append(indTestFin[i])
    while ang[indTestDebut[i]] > angMin :
        indTestDebut[i] = indTestDebut[i] -1 
        

#plt.figure( figsize=(8, 6))
#plt.plot(t , ang,'k' )

#for ii in range(nbTestRaq):
#    plt.plot(t[int(indTestDebut[ii]):indTestFin[ii]],ang[int(indTestDebut[ii]):indTestFin[ii]],'.r');
#plt.show()
#plt.close(1)
# part de ces pics et recule dans les increments pour recuperer les indTestDebut et indTestFin juste avant

## ========================================================================
#  etudie chaque test independamment
# =========================================================================

Energie_raquette=np.zeros(nbTestRaq) # initialisation
Energie_raquette_cplt=np.zeros([nbTestRaq*len(mLI_b)]) # initialisation

vitesseRaquettekmh=[]
global indAEnlever
indAEnlever=[]
for n in range (nbTestRaq):
    t_test=t[int(indTestDebut[n]):int(indTestFin[n])];
    ang_test=ang[int(indTestDebut[n]):int(indTestFin[n])]; 
    # recale de telle maniere que le premier temps soit=0
    t_test=t_test-t_test[0]
    plt.plot(t_test,ang_test,'.')
    plt.show()
#    #----- interpolation lineaire de la mesure sur ce petit intervalle
#    p=polyfit(t_test,ang_test,1)
    p=np.polyfit(t_test,ang_test,1)
    mesRegress = p[0]*t_test+p[1]
    plt.plot(t_test,mesRegress,'-c')
    plt.show() 
    save_essai = input("voulez-vous sauvegarder cette essai")
 
    omega =p[0]
    vitesse = omega*0.78
    vitesseRaquettekmh.append(vitesse*3.6)
    theta = np.mean(ang_test)
#    print ("Test numero =", n , " / vitesse raquette a = " , theta ," degres avant impact=", vitesse*3.6 , " km/h = " )
#    #----- energie apres impact    
    for ii in range (len(mLI_r)):
        if save_essai != '1':
            indAEnlever.append(n*(len(mLI_r)) + ii )
        g = 9.81
        mL=mLI_r[ii][0]
        I=mLI_r[ii][1]
        print ("ml = ", mL , " I = ", I)
        Energie_raquette_cplt[n*(len(mLI_r)) + ii] = (1/2*I*math.pow(omega,2)+mL*g*(1-math.cos(theta)))
        print (Energie_raquette_cplt[n*(len(mLI_r)) + (ii)])
#Energie_balle
#
#Energie_raquette=np.mean(Energie_raquette_cplt);
#print('*******************************************************')
#print('Energie (J) transmise dans la raquete (sans perte) est :')
#print('eme colonne pour test d''impact numero j')
#print(Energie_raquette)        
