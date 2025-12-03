import numpy as np
from pagerank_fonctions import pageRankLinear, pageRankPower, randomWalk

def main():

    #lecture des fichiers avec la matrice d'adjacence et le vecteur de personnalisation
    A = np.loadtxt('MatriceAdjacence.csv', delimiter=',')
    v = np.loadtxt('VecteurPersonnalisation.csv', delimiter= ',')

    #paramètre de téléportation
    alpha = 0.9

    #appel des fonctions afin de calculer pagerank
    linear = pageRankLinear(A, alpha, v)
    power = pageRankPower(A, alpha, v)
    #random = randomWalk(A, alpha, v)

    #afficher les résultats obtenus
    print(linear)
    print(power)
    #print(random)

def main2(): #Tests
    A = np.loadtxt('MatriceAdjacence.csv', delimiter=',')
    v = np.loadtxt('VecteurPersonnalisation.csv', delimiter= ',')


if (__name__ == '__main__'):
    main()