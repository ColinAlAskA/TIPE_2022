

a = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
              [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
              [1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
              [1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
              [1,1,1,1,1,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1],
              [1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1],
              [1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1],
              [1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1],
              [1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1],
              [1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1],
              [1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1],
              [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1],
              [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1],
              [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1],
              [1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
              [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])



def agrandir2():
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(200,50,1500,1000)
    f.set_tight_layout(True)
    plt.tight_layout(pad=0.5)


depart=(7,2)
arrivee=(8,20)


##### GBFS #####



def GBFS2(M,depart,arrivee):  #depart et arrivee sous la forme d'un tuple
    p,q=np.shape(M)[:2]
    file=[(depart,-1)]
    C=np.array([["blanc" for _ in range(q)] for _ in range(p)])
    P=np.array([[None for _ in range(q)] for _ in range(p)])
    P[depart]=(-1,-1)

    k=1
    while file!=[] and k<250:
        if k%10000 ==0 : print(k)
        pivot=file.pop(0)[0]
        #print(pivot)
        C[pivot]="noir"
        M[pivot]=[249, 173, 41]
        if pivot==arrivee:
            break
        for (v,_) in voisins(M,pivot,deplacement_diag):
            if C[v]== "blanc":
                P[v]=pivot
                d=distance_2points(v,arrivee)
                #print('On ajoute :',(v,d))
                file=ajouter_file(file,v,d)
                C[v]="gris"
                M[v]=[246, 253, 27 ]
        k+=1
        #print(file,'\n')
    return(P)

def PCC_GBFS2(M,depart,arrivee):  #depart et arrivee sous la forme de tuples
    P=GBFS2(M,depart,arrivee)
    x=arrivee
    chemin=[x]
    while x != depart:
        x=P[x]
        chemin = [x] + chemin
    return(chemin)


M=mur_ext(convertir(a))

M[depart]=[40, 180, 99]
M[arrivee]=[255, 0, 0]


##


f = plt.figure()
f.add_subplot(1,2,1)

plt.imshow(MM)

f.add_subplot(1,2,2)

chemin=PCC_GBFS2(M,depart,arrivee)
print('Coût du chemin :',cout_chemin(M,chemin))

plt.plot(np.array(chemin)[:,1],np.array(chemin)[:,0],color='blue',linewidth=5)

M[depart]=[40, 180, 99]
M[arrivee]=[235, 0, 0]

agrandir2()
plt.imshow(M)
plt.show()



##### Dijkstra #####


def Dijkstra2(M,depart,arrivee):
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
        M[pivot]=[249, 173, 41]
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
                M[v]=[246, 253, 27 ]
                file=ajouter_file(file,v,newD)
        k+=1
    return(P)


def PCC_Dijkstra2(M,depart,arrivee): #depart et arrivee sous la forme de tuples
    P=Dijkstra2(M,depart,arrivee)
    x=arrivee
    chemin=[x]
    while x != depart:
        x=P[x]
        chemin = [x] + chemin
    return(chemin)


M=mur_ext(convertir(a))
Dijkstra2(M,depart,arrivee)

M[depart]=[40, 180, 99]
M[arrivee]=[255, 0, 0]

MM=M
agrandir1()
plt.imshow(M)
plt.show()

##


f = plt.figure()
f.add_subplot(1,2,1)

plt.imshow(MM)

f.add_subplot(1,2,2)

chemin=PCC_Dijkstra2(M,depart,arrivee)
print('Coût du chemin :',cout_chemin(M,chemin))

plt.plot(np.array(chemin)[:,1],np.array(chemin)[:,0],color='blue',linewidth=5)

M[depart]=[40, 180, 99]
M[arrivee]=[235, 0, 0]

agrandir1()
plt.imshow(M)
plt.show()



##### A* #####


def Astar2(M,depart,arrivee):
    # ici, depart et arrivée sont sous la forme d'un tuple
    p,q=np.shape(M)[:2]
    file=[(depart,0)]

    C=np.array([["blanc" for _ in range(q)] for _ in range(p)])
    P=np.array([[None for _ in range(q)] for _ in range(p)])
    D=np.array([[np.infty for _ in range(q)] for _ in range(p)])

    P[depart]=(-1,-1)
    D[depart]=0

    k=1
    while file!=[] and k<192:
        if k%10000 ==0 : print(k)
        pivot=file.pop(0)[0]
        C[pivot]="noir"
        M[pivot]=[249, 173, 41]
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
                M[v]=[246, 253, 27 ]
                d=np.round(newD + distance_2points(v,arrivee),2)
                file=ajouter_file(file,v,d)
        #print(file,'\n')
        k+=1
    return(P)


def PCC_Astar2(M,depart,arrivee): #depart et arrivee sous la forme de tuples
    P=Astar2(M,depart,arrivee)
    x=arrivee
    chemin=[x]
    while x != depart:
        x=P[x]
        chemin = [x] + chemin
    return(chemin)


M=mur_ext(convertir(a))
Astar2(M,depart,arrivee)

M[depart]=[40, 180, 99]
M[arrivee]=[255, 0, 0]

M6=M
agrandir1()
plt.imshow(M)
plt.show()


##


f = plt.figure()
f.add_subplot(2,3,1)


plt.imshow(M1)
plt.title("k = 30 ",fontsize=25)

f.add_subplot(2,3,2)

plt.imshow(M2)
plt.title("k = 70 ",fontsize=25)

f.add_subplot(2,3,3)

plt.imshow(M3)
plt.title("k = 110 ",fontsize=25)

f.add_subplot(2,3,4)

plt.imshow(M4)
plt.title("k = 150 ",fontsize=25)

f.add_subplot(2,3,5)

plt.imshow(M5)
plt.title("k = 170 ",fontsize=25)



f.add_subplot(2,3,6)

chemin=PCC_Astar2(M,depart,arrivee)
print('Coût du chemin :',cout_chemin(M,chemin))

plt.plot(np.array(chemin)[:,1],np.array(chemin)[:,0],color='blue',linewidth=5)

M[depart]=[40, 180, 99]
M[arrivee]=[235, 0, 0]

agrandir2()
plt.imshow(M)
plt.title("k = 192 ",fontsize=25)


plt.show()






##### WA* #####

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
        M[pivot]=[249, 173, 41]
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
                M[v]=[246, 253, 27 ]
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



## WA* pour w=0 et Dijkstra

M=mur_ext(convertir(a))


f = plt.figure()

f.add_subplot(1,2,1)

M1=c.deepcopy(M)

chemin=PCC_Dijkstra2(M1,depart,arrivee)
print('Coût du chemin :',cout_chemin(M1,chemin))

plt.plot(np.array(chemin)[:,1],np.array(chemin)[:,0],color='blue',linewidth=5)
plt.title("Dijkstra",fontsize=30)


M1[depart]=[40, 180, 99]
M1[arrivee]=[235, 0, 0]

plt.imshow(M1)



f.add_subplot(1,2,2)

M2=c.deepcopy(M)

chemin=PCC_WAstar(M2,0,depart,arrivee)
print('Coût du chemin :',cout_chemin(M2,chemin))

plt.plot(np.array(chemin)[:,1],np.array(chemin)[:,0],color='blue',linewidth=5)
plt.title("WA* pour w=0 ",fontsize=30)


M2[depart]=[40, 180, 99]
M2[arrivee]=[235, 0, 0]


agrandir2()
plt.imshow(M2)
plt.show()





## WA* pour w=1 et Astar

M=mur_ext(convertir(a))


f = plt.figure()

f.add_subplot(1,2,1)

M1=c.deepcopy(M)

chemin=PCC_Astar2(M1,depart,arrivee)
print('Coût du chemin :',cout_chemin(M1,chemin))

plt.plot(np.array(chemin)[:,1],np.array(chemin)[:,0],color='blue',linewidth=5)
plt.title("A*",fontsize=30)


M1[depart]=[40, 180, 99]
M1[arrivee]=[235, 0, 0]

plt.imshow(M1)



f.add_subplot(1,2,2)

M2=c.deepcopy(M)

chemin=PCC_WAstar(M2,1,depart,arrivee)
print('Coût du chemin :',cout_chemin(M2,chemin))

plt.plot(np.array(chemin)[:,1],np.array(chemin)[:,0],color='blue',linewidth=5)
plt.title("WA* pour w=1 ",fontsize=30)


M2[depart]=[40, 180, 99]
M2[arrivee]=[235, 0, 0]



agrandir2()
plt.imshow(M2)
plt.show()





## WA* pour w>>1 et GBFS

M=mur_ext(convertir(a))


f = plt.figure()

f.add_subplot(2,3,2)

M1=c.deepcopy(M)

chemin=PCC_GBFS2(M1,depart,arrivee)
print('Coût du chemin :',cout_chemin(M1,chemin))

plt.plot(np.array(chemin)[:,1],np.array(chemin)[:,0],color='blue',linewidth=5)
plt.title("GBFS",fontsize=30)


M1[depart]=[40, 180, 99]
M1[arrivee]=[235, 0, 0]


plt.imshow(M1)



f.add_subplot(2,3,4)

M2=c.deepcopy(M)

chemin=PCC_WAstar(M2,5,depart,arrivee)
print('Coût du chemin :',cout_chemin(M2,chemin))

plt.plot(np.array(chemin)[:,1],np.array(chemin)[:,0],color='blue',linewidth=5)
plt.title("WA* pour w=5 ",fontsize=30)


M2[depart]=[40, 180, 99]
M2[arrivee]=[235, 0, 0]

plt.imshow(M2)


f.add_subplot(2,3,5)

M3=c.deepcopy(M)

chemin=PCC_WAstar(M3,10,depart,arrivee)
print('Coût du chemin :',cout_chemin(M3,chemin))

plt.plot(np.array(chemin)[:,1],np.array(chemin)[:,0],color='blue',linewidth=5)
plt.title("WA* pour w=10 ",fontsize=30)


M3[depart]=[40, 180, 99]
M3[arrivee]=[235, 0, 0]

plt.imshow(M3)


f.add_subplot(2,3,6)

M4=c.deepcopy(M)

chemin=PCC_WAstar(M4,50,depart,arrivee)
print('Coût du chemin :',cout_chemin(M4,chemin))

plt.plot(np.array(chemin)[:,1],np.array(chemin)[:,0],color='blue',linewidth=5)
plt.title("WA* pour w=50 ",fontsize=30)


M4[depart]=[40, 180, 99]
M4[arrivee]=[235, 0, 0]


agrandir2()
plt.imshow(M4)



plt.show()























