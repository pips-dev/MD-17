import numpy as np
import matplotlib.pyplot as plt
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
    random, graph = randomWalk(A, alpha, v)

    #afficher les résultats obtenus
    print(linear)
    print(power)
    print(random)
    
    plt.plot(graph)
    plt.title("Simulation de la marche aléatoire", fontsize=14, fontweight='semibold')
    plt.xlabel("Nombre de pas au temps k", fontsize=12, fontweight='medium')
    plt.ylabel("Erreur moyenne(ϵ)", fontsize=12, fontweight='medium')
    plt.show()
    
if (__name__ == '__main__'):
    main()