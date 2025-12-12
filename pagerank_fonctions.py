import numpy as np

#sources :
#https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html : utilisation pour résoudre les équations linéaires
#https://courspython.com/tableaux.html : utilisation de numpy
#https://numpy.org/doc/stable/reference/generated/numpy.sum.html : utilisation des sommes de numpy en préservant les dimensions initiales de la matrice
#https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html : redimensionnement de numpy

#formules utilisées :
#x=αPx+(1−α)v
#(I - alpha * P) * x = (1 - alpha) * v

def pageRankLinear(A: np.matrix, alpha: float, v: np.array) -> np.array:

    #normalisation de la matrice A en matrice de transitions de probabilités
    n = A.shape[0] #taille de A
    row_sum = A.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    P = A / row_sum #matrice de transition de probabilités
    
    I = np.eye(n) #matrice d'identité
    b = (1-alpha)*v
    x = np.linalg.solve(I-alpha * P.T, b) #faire la transposée de P car on calcule par la gauche et non par la droite
    x = x/np.sum(x) #normalisation de x
    return x

def pageRankPower(A: np.matrix, alpha: float, v: np.array) -> np.array:
    # Implémentation de la power method
    n = A.shape[0]

    #normalisation de la matrice A en matrice de transitions de probabilités
    row_sum = A.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    P = A / row_sum


    I = np.eye(n) #matrice diagonale remplie de 1 = matrice identité
    v = v/np.sum(v)  #normalisation du vecteur v de personnalisation
    G = alpha * P + (1 - alpha) * np.ones((n,1)) @ v[np.newaxis, :] #
    x = np.ones(n)/n #initilialisation du vecteur de probabilités uniformes 
    while True :
        new = x @ G #multiplication par la gauche ( et non droite
        new = new/np.sum(new) #normalisation de new (x)
        if (np.linalg.norm(new - x) < 0.0000000001):
            break
        x = new
    return new
    
def randomWalk(A: np.matrix, alpha: float, v: np.array) -> np.array:
    # Simulation de la marche aléatoire (10 000 pas)
    # Retourne le vecteur x des scores PageRank approximés
    
    n = A.shape[0]

    # normalisation de matrice A en matrice de transitions de probabilité
    row_sum = A.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    P = A / row_sum

    v = v/np.sum(v)  # normalisation du vecteur v de personnalisation

    steps = 10000
    current = 0 # noeud de départ(A)
    count = np.zeros(n) # compteur de visites pour chaque noeud
    old = np.zeros(n)
    tol = 0.0000000001  # condition d'arrêt demandée

    for s in range(steps):
        # suivre un lien sortant selon P
        if np.random.rand() < alpha: # génère un nombre aléatoire entre 0 et 1
            pro = P[current] 
            current = np.random.choice(n, p=pro) # choisi le prochain noeud en fonction des probabilités 
        else:
            current = np.random.choice(n, p=v) # si pas lien sortant, téléportation selon v
        count[current] += 1 # incrémenter le noeud visité

        d = count / (s+1)

        if np.linalg.norm(d - old) < tol:
            break

        old = d
    count = count/count.sum() # nombre de visites /nombre total de pas
    return count