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
    b = (1-alpha)*v
    x = np.linalg.solve(I-alpha * P, b)
    x = x/np.sum(x)
    return x

def pageRankPower(A: np.matrix, alpha: float, v: np.array) -> np.array:
    # Implémentation de la power method
    n = A.shape[0]

    #normalisation de la matrice A en matrice de transistions de probabilités
    row_sum = A.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    P = A / row_sum

    I = np.eye(n) #matrice diagonale remplie de 1 = matrice identité
    v = v/np.sum(v)  #normalisation du vecteur v
    v = np.transpose(v)
    x_{k+1} = α P x_k + (1−α) v
    G = alpha * P + (1
    x = np.ones(n)/n
    while True :
        new = G @ x
        if (np.linalg.norm(new - x) < 0.0000000001):
            return new
        x = new
    
def randomWalk(A: np.matrix, alpha: float, v: np.array) -> np.array:
    # Simulation de la marche aléatoire (10 000 pas)
    # Retourne le vecteur x des scores PageRank approximés
    pass