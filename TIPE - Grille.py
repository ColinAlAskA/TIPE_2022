##########  TIPE - GRILLE ##########

from math import *
import time
import random as r
import copy as c
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

cmax = 100
deplacement_diag=True

def agrandir1():
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(250,50,1400,1000)
    plt.tight_layout(pad=0.5)

def agrandir2():
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(200,50,1500,1000)
    #f.set_tight_layout(True)
    plt.tight_layout(pad=1.5)





## Fonctions utiles ##

def vecteur(case1,case2):
    return([case2[0]-case1[0],case2[1]-case1[1]])

def mur_ext(M):
    p,q=np.shape(M)[:2]
    new_M=np.array([[[0,0,0] for _ in range(q+2)] for _ in range(p+2)])
    new_M[1:p+1,1:q+1]=M[:,:]
    return(new_M)

def convertir(M):
    p,q=np.shape(M)
    new_M=np.zeros((p,q,3),dtype=int)
    for i in range(p):
        for j in range(q):
            new_M[i,j]=[255*M[i,j] for _ in range(3)]
    return(new_M)

def cout_chemin(M,chemin):
    n=len(chemin)
    res=0
    for k in range(n-1):
        (i,j)=chemin[k]
        (i2,j2)=chemin[k+1]
        if (i2,j2) in [(i-1,j+1),(i+1,j+1),(i+1,j-1),(i-1,j-1)]:
            res += cout(M,(i2,j2))*sqrt(2)
        else:
            res += cout(M,(i2,j2))
    return(np.round(res,2))

def distance_2points(case1,case2):
    #return(abs(case2[0]-case1[0])+abs(case2[1]-case1[1]))
    return(np.round(sqrt((case2[0]-case1[0])**2 + (case2[1]-case1[1])**2),3))

def voisins(M,case,bool): # Un booléen pour ajouter ou non les cases diagonales dans les voisins
    # La fonction renvoie la liste vf telle que vf=[(v,d),...] où v est une case voisine et d est un booléen qui vaut True si v est un voisin diagonal, False sinon.

    i,j=case
    v=[(i-1,j),(i,j+1),(i+1,j),(i,j-1)]         # haut, droite, bas, gauche
    l_cout=[cout(M,v[k]) for k in range(4)]
    vf=[]
    for k in range(4):
        if l_cout[k]!=cmax:
            vf.append((v[k],False))
    if bool:
        vd=[(i-1,j+1),(i+1,j+1),(i+1,j-1),(i-1,j-1)] # On commence en haut à droite...
        for k in range(4):
            if l_cout[k]!=cmax and l_cout[(k+1)%4]!=cmax and cout(M,vd[k])!=cmax:
                vf.append((vd[k],True))
    return(vf)





## Premier test ##

a = np.array([[1,1,0,1,1,1],
              [0,1,1,1,0,1],
              [1,0,1,0,0,0],
              [1,0,1,0,0,1],
              [1,1,1,1,0,1],
              [1,0,1,1,1,1]])

def cout(M,case):
    moyenne=np.mean(M[case])
    if moyenne<25:
        return(cmax)
    else:       # à préciser pour des cartes plus complexes
        return(1)

f = plt.figure()

M=mur_ext(convertir(a))

plt.imshow(M)
plt.show()


##### Parcours en largeur (ou Breadth First Search) #####


def parcours_en_largeur(M,depart,arrivee):  # depart et arrivee sous la forme d'un tuple
    p,q=np.shape(M)[:2]
    file=[depart]
    C=np.array([["blanc" for _ in range(q)] for _ in range(p)])
    P=np.array([[None for _ in range(q)] for _ in range(p)])
    P[depart]=(-1,-1)

    k=1
    while file!=[]:
        if k%10000==0 : print(k)
        pivot=file.pop(0)
        C[pivot]="noir"
        if pivot==arrivee:
            break
        for (v,_) in voisins(M,pivot,deplacement_diag):
            if C[v]== "blanc":
                file.append(v)
                P[v]=pivot
                C[v]="gris"
        k+=1
    return(P)

def PCC_BFS(M,depart,arrivee):  #depart et arrivee sous la forme de tuples
    P=parcours_en_largeur(M,depart,arrivee)
    x=arrivee
    chemin=[x]
    while x != depart:
        x=P[x]
        chemin = [x] + chemin
    return(chemin)


## Tracé graphique ##

a = np.array([[1,1,0,1,1,1],
              [0,1,1,1,0,1],
              [1,0,1,0,0,0],
              [1,0,1,0,0,1],
              [1,1,1,1,0,1],
              [1,0,1,1,1,1]])

M=mur_ext(convertir(a))
depart=(1,1)
arrivee=(4,6)

f = plt.figure()
f.add_subplot(1,2,1)

M[depart]=[40, 180, 99]
M[arrivee]=[235, 0, 0]
plt.imshow(M)


f.add_subplot(1,2,2)

chemin=PCC_BFS(M,depart,arrivee)
print('Coût du chemin :',cout_chemin(M,chemin))

plt.plot(np.array(chemin)[:,1],np.array(chemin)[:,0],color='blue',linewidth=5)

agrandir2()
plt.imshow(M)
plt.show()



##### GBFS #####

a = np.array([[1,1,0,1,1,1],
              [0,1,1,1,0,1],
              [1,0,1,1,0,0],
              [1,0,1,0,1,1],
              [1,1,1,1,0,1],
              [1,0,1,1,1,1]])
depart=(1,1)
arrivee=(3,4)

M=mur_ext(convertir(a))

# def ajouter_file(file,case,distance):
#     n=len(file)
#     bool=True
#     i=0
#     while i<n :
#         if bool and distance<file[i][1]:
#             bool=False
#             file=file[:i]+[(case,distance)]+file[i:]
#             i+=1
#             n+=1
#         elif case==file[i][0]:
#             if bool and distance>file[i][1]:
#                 bool=False
#             else:
#                 file.pop(i)
#                 i-=1
#                 n-=1
#         i+=1
#     if bool: return(file+[(case,distance)])
#     else: return(file)



def ajouter_file(file,case,distance):
    b=len(file)-1
    a=0
    while a<=b:
        m=(a+b)//2
        if case==file[m][0]:
            if distance<file[m][1]:
                file.pop(m)
                b=m-2
            else:
                return(file)
        elif distance<file[m][1]:
            b=m-1
        else:
            a=m+1
    file=file[:a]+[(case,distance)]+file[a:]
    return(file)


def GBFS(M,depart,arrivee):  #depart et arrivee sous la forme d'un tuple
    p,q=np.shape(M)[:2]
    file=[(depart,-1)]
    C=np.array([["blanc" for _ in range(q)] for _ in range(p)])
    P=np.array([[None for _ in range(q)] for _ in range(p)])
    P[depart]=(-1,-1)

    k=1
    while file!=[]:
        if k%10000 ==0 : print(k)
        pivot=file.pop(0)[0]
        #print(pivot)
        C[pivot]="noir"
        if pivot==arrivee:
            break
        for (v,_) in voisins(M,pivot,deplacement_diag):
            if C[v]== "blanc":
                P[v]=pivot
                d=distance_2points(v,arrivee)
                #print('On ajoute :',(v,d))
                file=ajouter_file(file,v,d)
                C[v]="gris"
        k+=1
        #print(file,'\n')
    return(P)

P=GBFS(M,depart,arrivee)
print(P)

def PCC_GBFS(M,depart,arrivee):  #depart et arrivee sous la forme de tuples
    P=GBFS(M,depart,arrivee)
    x=arrivee
    chemin=[x]
    while x != depart:
        x=P[x]
        chemin = [x] + chemin
    return(chemin)

chemin = PCC_GBFS(M,depart,arrivee)
print(chemin)

## Tracé graphique ##


a = np.ones((15,15))
a[2,2:13]=0
a[3:12,12]=0
a[12,2:13]=0

M=mur_ext(convertir(a))

depart=(13,1)
arrivee=(7,14)


f = plt.figure()

f.add_subplot(1,2,1)

M[depart]=[40, 180, 99]
M[arrivee]=[235, 0, 0]

chemin=PCC_BFS(M,depart,arrivee)
print('Coût du chemin :',cout_chemin(M,chemin))

plt.plot(np.array(chemin)[:,1],np.array(chemin)[:,0],color='blue',linewidth=5)
plt.title("Parcours en largeur",fontsize=25)

plt.imshow(M)

f.add_subplot(1,2,2)

M[depart]=[40, 180, 99]
M[arrivee]=[235, 0, 0]

chemin_GBFS=PCC_GBFS(M,depart,arrivee)
print('Coût du chemin :',cout_chemin(M,chemin_GBFS))

plt.plot(np.array(chemin_GBFS)[:,1],np.array(chemin_GBFS)[:,0],color='blue',linewidth=5)
plt.title("Parcours en largeur glouton",fontsize=25)


agrandir2()
plt.imshow(M)
plt.show()





##### Dijkstra #####


a = np.array([[0,0,1,1,1,1,1,1,0,0],
              [0,0,1,0.5,1,1,1,1,1,1],
              [0,1,1,0.5,0,0.5,0.5,0.5,1,1],
              [1,1,1,0.5,0.5,0.5,0.5,0.5,0.5,1],
              [1,1,1,0.5,0.5,0.5,0.5,0,1,1],
              [1,1,1,0.5,0.5,0,0.5,0.5,1,1],
              [1,1,1,0.5,0.5,0.5,0.5,0.5,1,1],
              [0,0,1,1,0.5,1,1,1,1,0],
              [0,1,1,1,1,1,1,1,0,0],
              [0,0,0,1,1,0,1,1,1,0]])
depart=(4,1)
arrivee=(7,10)

M=mur_ext(convertir(a))

def cout(M,case):
    moyenne=np.mean(M[case])
    if moyenne==0:
        return(cmax)
    elif moyenne==127:
        return(4)
    else:
        return(1)

def Dijkstra(M,depart,arrivee):
    # ici, depart et arrivee sont sous la forme d'un tuple
    p,q=np.shape(M)[:2]
    file=[(depart,0)]

    C=np.array([["blanc" for _ in range(q)] for _ in range(p)])
    P=np.array([[None for _ in range(q)] for _ in range(p)])
    D=np.array([[np.infty for _ in range(q)] for _ in range(p)])

    P[depart]=(-1,-1)
    D[depart]=0

    k=1
    while file!=[]:
        if k%10000 ==0 : print(k)
        pivot=file.pop(0)[0]
        C[pivot]="noir"
        if pivot==arrivee:
            break
        for (v,d) in voisins(M,pivot,deplacement_diag):
            if d:
                newD=np.round(D[pivot]+cout(M,v)*sqrt(2),2)
            else:
                newD=np.round(D[pivot]+cout(M,v),2)
            if C[v]=="blanc" and newD < D[v]:
                D[v]=newD
                P[v]=pivot
                file=ajouter_file(file,v,newD)
        k+=1
    return(P)


def PCC_Dijkstra(M,depart,arrivee): #depart et arrivee sous la forme de tuples
    P=Dijkstra(M,depart,arrivee)
    x=arrivee
    chemin=[x]
    while x != depart:
        x=P[x]
        chemin = [x] + chemin
    return(chemin)


print(PCC_Dijkstra(M,depart,arrivee))


## Tracé graphique ##

a = np.array([[0,0,1,0.5,0.5,1,1,1,0,0],
              [0,0,1,0.5,0.5,0.5,0.5,1,1,1],
              [0,1,1,0.5,0,0.5,0.5,0.5,1,1],
              [1,1,1,0.5,0.5,0.5,0.5,0.5,0.5,1],
              [1,1,1,0.5,0.5,0.5,0.5,0,1,1],
              [1,1,1,0.5,0.5,0,0.5,0.5,1,1],
              [1,1,1,0.5,0.5,0.5,0.5,0.5,1,1],
              [0,0,1,1,0.5,1,1,1,1,0],
              [0,1,1,1,1,1,1,1,0,0],
              [0,0,0,1,1,0,1,1,1,0]])
depart=(4,1)
arrivee=(2,10)
M=mur_ext(convertir(a))


f = plt.figure()

f.add_subplot(1,2,1)

M[depart]=[40, 180, 99]
M[arrivee]=[235, 0, 0]
plt.imshow(M)


f.add_subplot(1,2,2)

chemin=PCC_Dijkstra(M,depart,arrivee)
print('Coût du chemin :',cout_chemin(M,chemin))

plt.plot(np.array(chemin)[:,1],np.array(chemin)[:,0],color='blue',linewidth=5)


agrandir2()
plt.imshow(M)
plt.show()









#####  A*  #####

a = np.array([[0,0,1,0.5,0.5,1,1,1,0,0],
              [0,0,1,0.5,0.5,0.5,0.5,1,1,1],
              [0,1,1,0.5,0,0.5,0.5,0.5,1,1],
              [1,1,1,0.5,0.5,0.5,0.5,0.5,0.5,1],
              [1,1,1,0.5,0.5,0.5,0.5,0,1,1],
              [1,1,1,0.5,0.5,0,0.5,0.5,1,1],
              [1,1,1,0.5,0.5,0.5,0.5,0.5,1,1],
              [0,0,1,1,0.5,1,1,1,1,0],
              [0,1,1,1,1,1,1,1,0,0],
              [0,0,0,1,1,0,1,1,1,0]])

def Astar(M,depart,arrivee):
    # ici, depart et arrivée sont sous la forme d'un tuple
    p,q=np.shape(M)[:2]
    file=[(depart,0)]

    C=np.array([["blanc" for _ in range(q)] for _ in range(p)])
    P=np.array([[None for _ in range(q)] for _ in range(p)])
    D=np.array([[np.infty for _ in range(q)] for _ in range(p)])

    P[depart]=(-1,-1)
    D[depart]=0

    k=1
    while file!=[]:
        if k%10000 ==0 : print(k)
        pivot=file.pop(0)[0]
        C[pivot]="noir"
        #print(pivot)
        if pivot==arrivee:
            break
        for (v,d) in voisins(M,pivot,deplacement_diag):
            if d:
                newD=np.round(D[pivot]+cout(M,v)*sqrt(2),2)
            else:
                newD=np.round(D[pivot]+cout(M,v),2)
            if C[v]=="blanc" and newD < D[v]:
                D[v]=newD
                P[v]=pivot
                d=np.round(newD + distance_2points(v,arrivee),2)
                file=ajouter_file(file,v,d)
        #print(file,'\n')
        k+=1
    return(P)


def PCC_Astar(M,depart,arrivee): #depart et arrivee sous la forme de tuples
    P=Astar(M,depart,arrivee)
    x=arrivee
    chemin=[x]
    while x != depart:
        x=P[x]
        chemin = [x] + chemin
    return(chemin)

print(PCC_Astar(M,depart,arrivee))



## Tracé graphique ##

a = np.array([[0,0,1,0.5,0.5,1,1,1,0,0],
              [0,0,1,0.5,0.5,0.5,0.5,1,1,1],
              [0,1,1,0.5,0,0.5,0.5,0.5,1,1],
              [1,1,1,0.5,0.5,0.5,0.5,0.5,0.5,1],
              [1,1,1,0.5,0.5,0.5,0.5,0,1,1],
              [1,1,1,0.5,0.5,0,0.5,0.5,1,1],
              [1,1,1,0.5,0.5,0.5,0.5,0.5,1,1],
              [0,0,1,1,0.5,1,1,1,1,0],
              [0,1,1,1,1,1,1,1,0,0],
              [0,0,0,1,1,0,1,1,1,0]])
depart=(4,1)
arrivee=(2,10)
M=mur_ext(convertir(a))


f = plt.figure()

f.add_subplot(1,2,1)

M[depart]=[40, 180, 99]
M[arrivee]=[235, 0, 0]
plt.imshow(M)


f.add_subplot(1,2,2)

chemin=PCC_Astar(M,depart,arrivee)
print('Coût du chemin :',cout_chemin(M,chemin))

plt.plot(np.array(chemin)[:,1],np.array(chemin)[:,0],color='blue',linewidth=5)


agrandir2()
plt.imshow(M)
plt.show()


## Affichage des deux ##

a = np.array([[0,0,1,0.5,0.5,1,1,1,0,0],
              [0,0,1,0.5,0.5,0.5,0.5,1,1,1],
              [0,1,1,0.5,0,0.5,0.5,0.5,1,1],
              [1,1,1,0.5,0.5,0.5,0.5,0.5,0.5,1],
              [1,1,1,0.5,0.5,0.5,0.5,0,1,1],
              [1,1,1,0.5,0.5,0,0.5,0.5,1,1],
              [1,1,1,0.5,0.5,0.5,0.5,0.5,1,1],
              [0,0,1,1,0.5,1,1,1,1,0],
              [0,1,1,1,1,1,1,1,0,0],
              [0,0,0,1,1,0,1,1,1,0]])
depart=(4,1)
arrivee=(2,10)
M=mur_ext(convertir(a))
M[depart]=[40, 180, 99]
M[arrivee]=[235, 0, 0]


f = plt.figure()

f.add_subplot(1,2,1)

chemin_Dijkstra=PCC_Dijkstra(M,depart,arrivee)
print('Coût du chemin :',cout_chemin(M,chemin_Dijkstra))

plt.plot(np.array(chemin_Dijkstra)[:,1],np.array(chemin_Dijkstra)[:,0],color='blue',linewidth=5)
plt.title("Dijkstra",fontsize=25)

plt.imshow(M)


f.add_subplot(1,2,2)

chemin_Astar=PCC_Astar(M,depart,arrivee)
print('Coût du chemin :',cout_chemin(M,chemin_Astar))

plt.plot(np.array(chemin_Astar)[:,1],np.array(chemin_Astar)[:,0],color='blue',linewidth=5)
plt.title("A*",fontsize=25)

agrandir2()
plt.imshow(M)
plt.show()
















#########################
##### Rues de Paris #####
#########################

## Image de travail ##

link = 'C:/Colin/Cours/3ème année/TIPE/Images/'
image = 'Paris'

img0 = mpimg.imread (link + image + '.jpg')
img = mur_ext(np.array(img0,dtype=int))
print(np.shape(img))

depart=(117,623)
arrivee=(419,206)

plt.imshow(img)
agrandir1()
plt.show()

##

def cout(M,case):
    moyenne=np.mean(M[case])
    if moyenne<220:
        return(cmax)
    else:       # à préciser pour des cartes plus complexes
        return(1)

def carte_cout(M):
    p,q=np.shape(M)[:2]
    C=np.zeros((p,q))
    for i in range(p):
        for j in range(q):
            c_ij=cout(M,(i,j))
            if c_ij==cmax:
                C[i,j]=0
            else:
                C[i,j]=1
    return(C)

A=c.deepcopy(img)
Ac=carte_cout(A)



##### Tracé graphique BFS #####

# ATTENTION : temps d'attente plutôt élevé !

chemin=PCC_BFS(A,depart,arrivee)
print('\nChemin trouvé')
print('Coût du chemin :',cout_chemin(A,chemin))

##### Tracé graphique GBFS #####

# ATTENTION : temps d'attente plutôt élevé !

chemin=PCC_GBFS(A,depart,arrivee)
print('\nChemin trouvé')
print('Coût du chemin :',cout_chemin(A,chemin))

##### Tracé graphique Dijkstra #####

# ATTENTION : temps d'attente plutôt élevé !

chemin=PCC_Dijkstra(A,depart,arrivee)
print('\nChemin trouvé')
print('Coût du chemin :',cout_chemin(A,chemin))

##### Tracé graphique A* #####

# ATTENTION : temps d'attente plutôt élevé !

chemin=PCC_Astar(A,depart,arrivee)
print('\nChemin trouvé')
print('Coût du chemin :',cout_chemin(A,chemin))

## Sur la carte des coûts

plt.plot(np.array(chemin)[:,1],np.array(chemin)[:,0],color='cyan',linewidth=2)
plt.plot(depart[1],depart[0],color="g",marker='o',markersize=4)
plt.plot(arrivee[1],arrivee[0],color='r',marker='o',markersize=4)

plt.imshow(Ac,cmap='gray')
agrandir1()
plt.show()


## Sur la carte classique

plt.plot(np.array(chemin)[:,1],np.array(chemin)[:,0],color='cyan',linewidth=2)
plt.plot(depart[1],depart[0],color="g",marker='o',markersize=4)
plt.plot(arrivee[1],arrivee[0],color='r',marker='o',markersize=4)

plt.imshow(A)
agrandir1()
plt.show()









##########################
##### Paris amélioré #####
##########################

## Fonctions utiles ##

# ATTENTION, cette fonction est différente de la fonction "voisins" initiale

def voisins_directs(case,bool):
    i,j=case
    l=[(i-1,j),(i,j+1),(i+1,j),(i,j-1)]
    if bool: return(l+[(i-1,j+1),(i+1,j-1),(i+1,j+1),(i-1,j-1)])
    else: return(l)

def inter_cercle(M,centre,rayon,couleur):
    cx,cy=centre
    tab_case=[[(i,j) for i in range(round(cx-rayon),round(cx+rayon)+1)] for j in range(round(cy-rayon),round(cy+rayon)+1)]
    for l_case in tab_case:
        for case in l_case:
            if distance_2points(case,centre)<=rayon and np.mean(M[case])>=220:
                M[case]=couleur


def inter_rectangle(M,coin,theta,largeur,longueur,couleur):
    cx,cy=coin
    tab_case=[]
    for i in range(cx-longueur,cx):
        for j in range(cy,cy+largeur):
            vec=vecteur((i,j),coin)
            norme=np.linalg.norm(vec)
            v1 = vec / norme
            v2 = [-1,0]
            produit = np.dot(v1,v2)
            angle = abs(np.arccos(produit))
            (i2,j2)=(round(coin[0]+norme*cos(theta+angle)),round(coin[1]+norme*sin(theta+angle)))
            tab_case.append((i2,j2))
            tab_case += voisins_directs((i2,j2),False)

    for case in tab_case:
        if np.mean(M[case])>=220:
            M[case]=couleur


c1=np.array([255,0,0])          # Rouge
c2=np.array([124,50,0])         # Marron
c3=np.array([252,161,11])       # Orange
c4=np.array([241,235,54])       # Jaune

def cout(M,case):
    moyenne=np.mean(M[case])
    if np.all(M[case]==c1):
        return(8)
    # elif np.all(M[case]==c2):
    #     return(4)
    elif np.all(M[case]==c3):
        return(6)
    elif np.all(M[case]==c4):
        return(2)
    else:
        if moyenne<220:
            return(cmax)
        else:
            return(1)



## Image de travail ##

link = 'C:/Colin/Cours/3ème année/TIPE/Images/'
image = 'Paris'

img0 = mpimg.imread (link + image + '.jpg')
img = mur_ext(np.array(img0,dtype=int))
print(np.shape(img)[:2])


# circle1 = plt.Circle((465,345), 30,ec='r',fc='None')
# fig, ax = plt.subplots()
# ax.add_patch(circle1)

##

depart=(219,699)
arrivee=(445,208)

##

img = mur_ext(np.array(img0,dtype=int))

t1=time.time()

inter_cercle(img,(345,465),40,c1)
inter_cercle(img,(225,254),40,c1)
inter_cercle(img,(378,651),40,c1)

# inter_cercle(img,(150,429),15,c2)
# inter_cercle(img,(239,609),15,c2)
# inter_cercle(img,(247,645),15,c2)
# inter_cercle(img,(188,594),15,c2)
# inter_cercle(img,(203,574),10,c2)
# inter_cercle(img,(532,424),15,c2)
# inter_cercle(img,(328,202),10,c2)


inter_rectangle(img,(381,280),pi/4,52,106,c3)
inter_rectangle(img,(317,427),1.12,20,23,c1)
inter_rectangle(img,(317,401),1.12,33,65,c3)
inter_rectangle(img,(278,348),1.12,16,80,c3)

for _ in range(20):
    a,b=r.randint(100,550),r.randint(100,700)
    inter_cercle(img,(a,b),50,c4)


t2=time.time()

print("Temps de calcul des opérations annexes sur l'image :",np.round(t2-t1,4))

plt.imshow(img)
agrandir1()
plt.show()


A=c.deepcopy(img)

##### Tracé graphique BFS #####

# ATTENTION : temps d'attente plutôt élevé !

chemin_BFS=PCC_BFS(A,depart,arrivee)
print('\nChemin trouvé')
print('Coût du chemin :',cout_chemin(A,chemin_BFS))

##### Tracé graphique GBFS #####

# ATTENTION : temps d'attente plutôt élevé !

chemin_GBFS=PCC_GBFS(A,depart,arrivee)
print('\nChemin trouvé')
print('Coût du chemin :',cout_chemin(A,chemin_GBFS))

##### Tracé graphique Dijkstra #####

# ATTENTION : temps d'attente plutôt élevé !

chemin_Dijkstra=PCC_Dijkstra(A,depart,arrivee)
print('\nChemin trouvé')
print('Coût du chemin :',cout_chemin(A,chemin_Dijkstra))

##### Tracé graphique A* #####

# ATTENTION : temps d'attente plutôt élevé !

chemin_Astar=PCC_Astar(A,depart,arrivee)
print('\nChemin trouvé')
print('Coût du chemin :',cout_chemin(A,chemin_Astar))


## Sur la carte classique

plt.plot(np.array(chemin_BFS)[:,1],np.array(chemin_BFS)[:,0],color='blue',linewidth=3)
plt.plot(np.array(chemin_GBFS)[:,1],np.array(chemin_GBFS)[:,0],color='cyan',linewidth=3)
plt.plot(np.array(chemin_Dijkstra)[:,1],np.array(chemin_Dijkstra)[:,0],color='magenta',linewidth=3)
plt.plot(np.array(chemin_Astar)[:,1],np.array(chemin_Astar)[:,0],color='green',linewidth=3)

plt.plot(depart[1],depart[0],color="lime",marker='o',markersize=10)
plt.plot(arrivee[1],arrivee[0],color='r',marker='o',markersize=10)

plt.imshow(A)
agrandir1()
plt.show()


##### Temps des calculs #####

## BFS ##

N=1

l=[]
for _ in range(N):
    t0=time.time()
    chemin_BFS=PCC_BFS(A,depart,arrivee)
    t1=time.time()
    chemin_BFS=[]
    l.append(t1-t0)

print(np.round(np.mean(l),5))


## GBFS ##

N=1

l=[]
for _ in range(N):
    t0=time.time()
    chemin_GBFS=PCC_GBFS(A,depart,arrivee)
    t1=time.time()
    chemin_GBFS=[]
    l.append(t1-t0)

print(np.round(np.mean(l),5))


## Dijkstra ##

N=1

l=[]
for _ in range(N):
    t0=time.time()
    chemin_Dijkstra=PCC_Dijkstra(A,depart,arrivee)
    t1=time.time()
    chemin_Dijkstra=[]
    l.append(t1-t0)

print(np.round(np.mean(l),5))


## A* ##

N=1

l=[]
for _ in range(N):
    t0=time.time()
    chemin_Astar=PCC_Astar(A,depart,arrivee)
    t1=time.time()
    chemin_Astar=[]
    l.append(t1-t0)

print(np.round(np.mean(l),5))











##############################
##### Polygone aléatoire #####
##############################


## Image de travail ##

link = 'C:/Colin/Cours/3ème année/TIPE/Images/'
image = 'Polygone_cercle'

img0 = mpimg.imread (link + image + '.png')
img = np.array(img0[35:,40:960,:3]*255,dtype=int)
img[202,627]=[0,0,0]
img[82,642]=[0,0,0]
img[85,643]=[0,0,0]

print(np.shape(img))

depart=(105,237)
arrivee=(465,458)

plt.plot(depart[1],depart[0],color="lime",marker='o',markersize=10)
plt.plot(arrivee[1],arrivee[0],color='r',marker='o',markersize=10)

plt.imshow(img)
agrandir1()
plt.show()

##


def cout(M,case):
    moyenne=np.mean(M[case])
    if moyenne<20:
        return(1)
    else:       # à préciser pour des cartes plus complexes
        return(cmax)

def carte_cout(M):
    p,q=np.shape(M)[:2]
    C=np.zeros((p,q))
    for i in range(p):
        for j in range(q):
            c_ij=cout(M,(i,j))
            if c_ij==cmax:
                C[i,j]=0
            else:
                C[i,j]=1
    return(C)

A=c.deepcopy(img)
Ac=carte_cout(A)


##### Tracé graphique BFS #####

# ATTENTION : temps d'attente plutôt élevé !

chemin_BFS=PCC_BFS(A,depart,arrivee)
print('\nChemin trouvé')
print('Coût du chemin :',cout_chemin(A,chemin_BFS))

##### Tracé graphique GBFS #####

# ATTENTION : temps d'attente plutôt élevé !

chemin_GBFS=PCC_GBFS(A,depart,arrivee)
print('\nChemin trouvé')
print('Coût du chemin :',cout_chemin(A,chemin_GBFS))

##### Tracé graphique Dijkstra #####

# ATTENTION : temps d'attente plutôt élevé !

chemin_Dijkstra=PCC_Dijkstra(A,depart,arrivee)
print('\nChemin trouvé')
print('Coût du chemin :',cout_chemin(A,chemin_Dijkstra))

##### Tracé graphique A* #####

# ATTENTION : temps d'attente plutôt élevé !

chemin_Astar=PCC_Astar(A,depart,arrivee)
print('\nChemin trouvé')
print('Coût du chemin :',cout_chemin(A,chemin_Astar))

## Sur la carte des coûts

plt.plot(np.array(chemin_BFS)[:,1],np.array(chemin_BFS)[:,0],color='red',linewidth=3,label="BFS")
plt.plot(np.array(chemin_GBFS)[:,1],np.array(chemin_GBFS)[:,0],color='cyan',linewidth=3,label="GBFS")
plt.plot(np.array(chemin_Dijkstra)[:,1],np.array(chemin_Dijkstra)[:,0],color='blue',ls='--',linewidth=3,label="Dijkstra")
plt.plot(np.array(chemin_Astar)[:,1],np.array(chemin_Astar)[:,0],color='magenta',linewidth=3,label="A*")

plt.plot(depart[1],depart[0],color="lime",marker='o',markersize=10)
plt.plot(arrivee[1],arrivee[0],color='r',marker='o',markersize=10)

plt.legend(loc='lower right',fontsize=18)

plt.imshow(Ac,cmap='gray')
agrandir1()
plt.show()


## Sur la carte classique

plt.plot(np.array(chemin_BFS)[:,1],np.array(chemin_BFS)[:,0],color='red',linewidth=3,label="BFS")
plt.plot(np.array(chemin_GBFS)[:,1],np.array(chemin_GBFS)[:,0],color='cyan',linewidth=3,label="GBFS")
plt.plot(np.array(chemin_Dijkstra)[:,1],np.array(chemin_Dijkstra)[:,0],color='blue',ls='--',linewidth=3,label="Dijkstra")
plt.plot(np.array(chemin_Astar)[:,1],np.array(chemin_Astar)[:,0],color='magenta',linewidth=3,label="A*")

plt.plot(depart[1],depart[0],color="lime",marker='o',markersize=10)
plt.plot(arrivee[1],arrivee[0],color='r',marker='o',markersize=10)

plt.legend(loc='lower right',fontsize=18)

plt.imshow(A)
agrandir1()
plt.show()






##### Temps des calculs #####

## BFS ##

N=1

l=[]
for _ in range(N):
    t0=time.time()
    chemin_BFS=PCC_BFS(A,depart,arrivee)
    t1=time.time()
    chemin_BFS=[]
    l.append(t1-t0)

print(np.round(np.mean(l),5))


## GBFS ##

N=1

l=[]
for _ in range(N):
    t0=time.time()
    chemin_GBFS=PCC_GBFS(A,depart,arrivee)
    t1=time.time()
    chemin_GBFS=[]
    l.append(t1-t0)

print(np.round(np.mean(l),5))


## Dijkstra ##

N=1

l=[]
for _ in range(N):
    t0=time.time()
    chemin_Dijkstra=PCC_Dijkstra(A,depart,arrivee)
    t1=time.time()
    chemin_Dijkstra=[]
    l.append(t1-t0)

print(np.round(np.mean(l),5))


## A* ##

N=1

l=[]
for _ in range(N):
    t0=time.time()
    chemin_Astar=PCC_Astar(A,depart,arrivee)
    t1=time.time()
    chemin_Astar=[]
    l.append(t1-t0)

print(np.round(np.mean(l),5))








#
#
# #############################
# ##### Horizon Zero Dawn #####
# #############################
#
#
# ## Image de travail ##
#
# link = 'C:/Colin/Cours/3ème année/TIPE/Images/'
# image = 'HZD_Map'
#
# img0 = mpimg.imread (link + image + '.png')
# img1 = np.array(img0[:,:,:3]*255,dtype=int)
# rajout1 = np.zeros((15,1000,3),dtype=int)
# rajout2 = np.zeros((14,1000,3),dtype=int)
# img = np.concatenate([rajout1,img1,rajout2],axis=0)
#
#
# print(np.shape(img))
# plt.imshow(img)
# agrandir1()
# plt.show()
#
#









###########################
##### Carte aléatoire #####
###########################

# Dimensions de la carte :
p,q=100,100

img=mur_ext(255*np.ones((p,q,3),dtype=int))

depart=(33,23)
arrivee=(p,q-18)

c1=[224,205,169]        # Sable
c2=[53, 194, 53]        # Herbe
c3=[37, 77, 213]        # Eau
blanc=[255,255,255]
noir=[0,0,0]


##

link = 'C:/Colin/Cours/3ème année/TIPE/Images/'
image = 'Maison'

img0 = mpimg.imread (link + image + '.png')
maison = np.array(255*img0[:,:,:3],dtype=int)

n1,n2=np.shape(maison)[:2]
for i in range(n1):
    for j in range(n2):
        if np.mean(maison[i,j])>222:
            maison[i,j]=c2

##

link = 'C:/Colin/Cours/3ème année/TIPE/Images/'
image = 'coffre'

img0 = mpimg.imread (link + image + '.png')
coffre = np.array(255*img0[:,:,:3],dtype=int)

n1,n2=np.shape(coffre)[:2]
for i in range(n1):
    for j in range(n2):
        if np.mean(coffre[i,j])>200:
            coffre[i,j]=c1


##

def mur_ext_False(M):
    new_M=np.array([[False for _ in range(q+2)] for _ in range(p+2)])
    new_M[1:p+1,1:q+1]=M[:,:]
    return(new_M)

def voisins_directs(case,bool):
    i,j=case
    l=[(i-1,j),(i,j+1),(i+1,j),(i,j-1)]
    if bool: return(l+[(i-1,j+1),(i+1,j-1),(i+1,j+1),(i-1,j-1)])
    else: return(l)

# def voisins_directs_2(case):
#     i,j=case
#     return([(i-1,j),(i,j+1),(i+1,j),(i,j-1),(i-1,j+1),(i+1,j-1),(i+1,j+1),(i-1,j-1),(i-2,j-2),(i-1,j-2),(i,j-2),(i+1,j-2),(i+2,j-2),(i+2,j-1),(i+2,j),(i+2,j+1),(i+2,j+2),(i+1,j+2),(i,j+2),(i-1,j+2),(i-2,j+2),(i-2,j+1),(i-2,j),(i-2,j-1)])

def voisins_mur(case,bool):
        i,j=case
        if bool: return([(i-1,j),(i+1,j)])
        else: return([(i,j-1),(i,j+1)])

def applique_carre(M,B,case,couleur,k):
    if k==0:
        return(M)
    else:
        M[case]=couleur
        l=voisins_directs(case,True)
        for (ii,jj) in l:
            if B[ii,jj]:
                M=applique_carre(M,B,(ii,jj),couleur,k-1)
        return(M)

def applique_trait(M,B,bool,case,couleur,k):
    if k==0:
        return(M)
    else:
        M[case]=couleur
        l=voisins_mur(case,bool)
        for (ii,jj) in l:
            if B[ii,jj]:
                M=applique_trait(M,B,bool,(ii,jj),couleur,k-1)
        return(M)

def cout(M,case):
    moyenne=np.mean(M[case])
    if np.all(M[case]==c1):         # Sable
        return(2)
    elif np.all(M[case]==c2):       # Herbe
        return(1)
    elif np.all(M[case]==c3):       # Eau
        return(5)
    elif np.all(M[case]==blanc):    # Piège
        return(10)
    else:
        return(cmax)


##

B=mur_ext_False(np.array([[True for _ in range(q)] for _ in range(p)]))

def initialise(M,P,Pherbe,Peau):
# Peau + Herbe + Ppiege = 1 et P = la proba que la case soit un obstacle*
    M[1:p+1,1:q+1]=c1
    M[1:37,1:35]=c2
    M[p-16:p+1,q-18:q+1]=c1
    M[1:33,1:32]=maison
    M[p-15:p+1,q-17:q+1]=coffre
    B=mur_ext_False(np.array([[True for _ in range(q)] for _ in range(p)]))
    B[:36,:34]=False
    B[p-16:,q-18:]=False

    for i in range(1,p+1):
        for j in range(1,q+1):
            if B[i,j]:
                X=r.random()
                if X<=P:
                    P1=Pherbe
                    P2=P1+Peau
                    Y=r.random()
                    if Y<=P1:                                    # Herbe
                        M=applique_carre(M,B,(i,j),c2,4)
                    elif (P1<Y) and (Y<=P2):                     # Eau
                        M=applique_carre(M,B,(i,j),c3,5)
                    else:                                        # Piege
                        M=applique_carre(M,B,(i,j),blanc,1)

    M[1:round((2/3)*p),round((1/3)*q)+1]=noir
    M[round((1/3)*p):p+1,round((2/3)*q)+2]=noir
    pas=7
    M[33,69-pas:69+pas+1]=noir
    M[66,34-pas:34+pas+1]=noir
    return(M)

A=c.deepcopy(img)
A=initialise(A,0.2,0.07,0.05)

plt.imshow(A)
agrandir1()
plt.show()





##### Tracé graphique BFS #####

# ATTENTION : temps d'attente plutôt élevé !

chemin_BFS=PCC_BFS(A,depart,arrivee)
print('\nChemin trouvé')
print('Coût du chemin :',cout_chemin(A,chemin_BFS))

##### Tracé graphique GBFS #####

# ATTENTION : temps d'attente plutôt élevé !

chemin_GBFS=PCC_GBFS(A,depart,arrivee)
print('\nChemin trouvé')
print('Coût du chemin :',cout_chemin(A,chemin_GBFS))

##### Tracé graphique Dijkstra #####

# ATTENTION : temps d'attente plutôt élevé !

chemin_Dijkstra=PCC_Dijkstra(A,depart,arrivee)
print('\nChemin trouvé')
print('Coût du chemin :',cout_chemin(A,chemin_Dijkstra))

##### Tracé graphique A* #####

# ATTENTION : temps d'attente plutôt élevé !

chemin_Astar=PCC_Astar(A,depart,arrivee)
print('\nChemin trouvé')
print('Coût du chemin :',cout_chemin(A,chemin_Astar))


## Sur la carte classique

plt.plot(np.array(chemin_BFS)[:,1],np.array(chemin_BFS)[:,0],color='magenta',linewidth=4)
plt.plot(np.array(chemin_GBFS)[:,1],np.array(chemin_GBFS)[:,0],color='orange',linewidth=4)
plt.plot(np.array(chemin_Dijkstra)[:,1],np.array(chemin_Dijkstra)[:,0],color='yellow',linewidth=4)
plt.plot(np.array(chemin_Astar)[:,1],np.array(chemin_Astar)[:,0],color='red',linewidth=4)

plt.plot(depart[1],depart[0],color="g",marker='o',markersize=4)
plt.plot(arrivee[1],arrivee[0],color='r',marker='o',markersize=4)

plt.imshow(A)
agrandir1()
plt.show()




##### Temps des calculs #####

## BFS ##

N=10

l=[]
for _ in range(N):
    A=initialise(A,0.2,0.07,0.05)
    t0=time.time()
    chemin_BFS=PCC_BFS(A,depart,arrivee)
    t1=time.time()
    chemin_BFS=[]
    l.append(t1-t0)

print(np.round(np.mean(l),5))


## GBFS ##

N=10

l=[]
for _ in range(N):
    A=initialise(A,0.2,0.07,0.05)
    t0=time.time()
    chemin_GBFS=PCC_GBFS(A,depart,arrivee)
    t1=time.time()
    chemin_GBFS=[]
    l.append(t1-t0)

print(np.round(np.mean(l),5))


## Dijkstra ##

N=10

l=[]
for _ in range(N):
    A=initialise(A,0.2,0.07,0.05)
    t0=time.time()
    chemin_Dijkstra=PCC_Dijkstra(A,depart,arrivee)
    t1=time.time()
    chemin_Dijkstra=[]
    l.append(t1-t0)

print(np.round(np.mean(l),5))


## A* ##

N=10

l=[]
for _ in range(N):
    A=initialise(A,0.2,0.07,0.05)
    t0=time.time()
    chemin_Astar=PCC_Astar(A,depart,arrivee)
    t1=time.time()
    chemin_Astar=[]
    l.append(t1-t0)

print(np.round(np.mean(l),5))








##########################
##### WA* pour Paris #####
##########################


def WAstar(M,w,depart,arrivee):
    # ici, depart et arrivée sont sous la forme d'un tuple
    p,q=np.shape(M)[:2]
    file=[(depart,0)]

    C=np.array([["blanc" for _ in range(q)] for _ in range(p)])
    P=np.array([[None for _ in range(q)] for _ in range(p)])
    D=np.array([[np.infty for _ in range(q)] for _ in range(p)])

    P[depart]=(-1,-1)
    D[depart]=0

    k=1
    while file!=[]:
        if k%10000 ==0 : print(k)
        pivot=file.pop(0)[0]
        C[pivot]="noir"
        if pivot==arrivee:
            break
        for (v,d) in voisins(M,pivot,deplacement_diag):
            if d:
                newD=np.round(D[pivot]+cout(M,v)*sqrt(2),2)
            else:
                newD=np.round(D[pivot]+cout(M,v),2)
            if C[v]=="blanc" and newD < D[v]:
                D[v]=newD
                P[v]=pivot
                d=np.round(newD + w*distance_2points(v,arrivee),2)
                file=ajouter_file(file,v,d)
        #print(file,'\n')
        k+=1
    return(P)


def PCC_WAstar(M,w,depart,arrivee): #depart et arrivee sous la forme de tuples
    P=WAstar(M,w,depart,arrivee)
    x=arrivee
    chemin=[x]
    while x != depart:
        x=P[x]
        chemin = [x] + chemin
    return(chemin)

## Image de travail ##

link = 'C:/Colin/Cours/3ème année/TIPE/Images/'
image = 'Paris'

img0 = mpimg.imread (link + image + '.jpg')
img = mur_ext(np.array(img0,dtype=int))
print(np.shape(img)[:2])


# circle1 = plt.Circle((465,345), 30,ec='r',fc='None')
# fig, ax = plt.subplots()
# ax.add_patch(circle1)

##

depart=(116,465)
arrivee=(366,252)




##

img = mur_ext(np.array(img0,dtype=int))

t1=time.time()

inter_cercle(img,(345,465),40,c1)
inter_cercle(img,(225,254),40,c1)
inter_cercle(img,(378,651),40,c1)

# inter_cercle(img,(150,429),15,c2)
# inter_cercle(img,(239,609),15,c2)
# inter_cercle(img,(247,645),15,c2)
# inter_cercle(img,(188,594),15,c2)
# inter_cercle(img,(203,574),10,c2)
# inter_cercle(img,(532,424),15,c2)
# inter_cercle(img,(328,202),10,c2)


inter_rectangle(img,(381,280),pi/4,52,106,c3)
inter_rectangle(img,(317,427),1.12,20,23,c1)
inter_rectangle(img,(317,401),1.12,33,65,c3)
inter_rectangle(img,(278,348),1.12,16,80,c3)

for _ in range(20):
    a,b=r.randint(100,550),r.randint(100,700)
    inter_cercle(img,(a,b),50,c4)


t2=time.time()

print("Temps de calcul des opérations annexes sur l'image :",np.round(t2-t1,4))

plt.imshow(img)
agrandir1()
plt.show()


A=c.deepcopy(img)



##### Tracé graphique 1 #####

# ATTENTION : temps d'attente plutôt élevé !

chemin1=PCC_WAstar(A,0,depart,arrivee)
print('\nChemin trouvé')
print('Coût du chemin :',cout_chemin(A,chemin1))

##### Tracé graphique 2 #####

# ATTENTION : temps d'attente plutôt élevé !

chemin2=PCC_WAstar(A,1,depart,arrivee)
print('\nChemin trouvé')
print('Coût du chemin :',cout_chemin(A,chemin2))

##### Tracé graphique 3 #####

# ATTENTION : temps d'attente plutôt élevé !

chemin3=PCC_WAstar(A,2,depart,arrivee)
print('\nChemin trouvé')
print('Coût du chemin :',cout_chemin(A,chemin3))

##### Tracé graphique 4 #####

# ATTENTION : temps d'attente plutôt élevé !

chemin4=PCC_WAstar(A,5,depart,arrivee)
print('\nChemin trouvé')
print('Coût du chemin :',cout_chemin(A,chemin4))

##### Tracé graphique 5 #####

# ATTENTION : temps d'attente plutôt élevé !

chemin5=PCC_WAstar(A,10,depart,arrivee)
print('\nChemin trouvé')
print('Coût du chemin :',cout_chemin(A,chemin5))

##### Tracé graphique 6 #####

# ATTENTION : temps d'attente plutôt élevé !

chemin6=PCC_WAstar(A,50,depart,arrivee)
print('\nChemin trouvé')
print('Coût du chemin :',cout_chemin(A,chemin6))


## Sur la carte classique

plt.plot(np.array(chemin6)[:,1],np.array(chemin6)[:,0],color=('#2DF497'),linewidth=3,label="w=50")
plt.plot(np.array(chemin5)[:,1],np.array(chemin5)[:,0],color=('#00FFF7'),linewidth=3,label="w=10")
plt.plot(np.array(chemin4)[:,1],np.array(chemin4)[:,0],color=('#00AEFF'),linewidth=3,label="w=5")
plt.plot(np.array(chemin3)[:,1],np.array(chemin3)[:,0],color=('#0065FF'),linewidth=3,label="w=2")
plt.plot(np.array(chemin2)[:,1],np.array(chemin2)[:,0],color=('#3100CD'),linewidth=3,label="w=1")
plt.plot(np.array(chemin1)[:,1],np.array(chemin1)[:,0],color=('#1A245B'),linewidth=3,label="w=0")


plt.plot(depart[1],depart[0],color="lime",marker='o',markersize=10)
plt.plot(arrivee[1],arrivee[0],color='r',marker='o',markersize=10)
plt.legend(loc='lower right',fontsize=25,framealpha=1)

plt.imshow(A)
agrandir1()
plt.show()




##### Temps des calculs #####

## WA* ##

N=1
w=0

l=[]
for _ in range(N):
    t0=time.time()
    chemin=PCC_WAstar(A,w,depart,arrivee)
    t1=time.time()
    l.append(t1-t0)

print(np.round(np.mean(l),5))






















