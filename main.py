import numpy as np
from pagerank_fonctions import pageRankLinear, pageRankPower, randomWalk

def main():

    #lecture des fichiers avec la matrice d'adjacence et le vecteur de personnalisation
    A = np.genfromtxt('MatriceAdjacence.csv', delimiter=',', skip_header=1)
    v = np.genfromtxt('VecteurPersonnalisation_Groupe17.csv', delimiter= ',')
    print(v.shape)

    #paramètre de téléportation
    alpha = 0.9

    #appel des fonctions afin de calculer pagerank
    linear = pageRankLinear(A, alpha, v)
    power = pageRankPower(A, alpha, v)
    #random = randomWalk(A, alpha, v)

    #afficher les résultats obtenus
    print('this is for linear')
    print(linear)
    print('\n')
    print('this is for power')
    print(power)
    #print(random)

if (__name__ == '__main__'):
    main()
