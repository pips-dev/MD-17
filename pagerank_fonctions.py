import numpy as np

#sources :
#https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html : utilisation pour résoudre les équations linéaires
#https://courspython.com/tableaux.html : utilisation de numpy

#formules utilisées :
#x=αPx+(1−α)v
#(I - alpha * P) * x = (1 - alpha) * v

def pageRankLinear(A: np.matrix, alpha: float, v: np.array) -> np.array:
    n = A.shape[0] #taille de A
    rows = A.sum(axis=1) #la somme des lignes de A
    P = np.zeros((n,n)) #matrice n*n remplie de zéros
    for i in range(n):
        for j in range(n):
            if (rows[i] > 0):
                P[i][j] = A[i][j]/rows[i]
            else :
                P[i][j]= 1/n
    I = np.eye(n)
    print(I)
    v = v/np.sum(v)
    b = (1-alpha)*v
    x = np.linalg.solve(I-alpha * P, b)
    x = x/np.sum(x)
    return x

def pageRankPower(A: np.matrix, alpha: float, v: np.array) -> np.array:
    # Implémentation de la power method
    # Affiche les matrices A, P, G, les 3 premières itérations, et le résultat final
    n = A.shape[0] #taille de A
    print(n)
    rows = A.sum(axis=1) #la somme des lignes de A
    print(rows)
    P = np.zeros((n,n)) #matrice n*n remplie de zéros
    print(P)
    for i in range(n):
        if rows[i] > 0:
            P[i, :] = A[i, :] / rows[i]
        else:
            P[i, :] = 1/n

    v = v/np.sum(v)
    b = (1-alpha)*np.outer(np.ones(n), v) #application de la formule de la power méthode
    G = alpha * P + b
    x = np.ones(n)/n
    for m in range (10000):
        new = G @ x
        if (np.linalg.norm(new - x, 1) < 1e-8):
            return new / np.sum(new)
        x = new

def randomWalk(A: np.matrix, alpha: float, v: np.array) -> np.array:
    # Simulation de la marche aléatoire (10 000 pas)
    # Retourne le vecteur x des scores PageRank approximés
    pass
