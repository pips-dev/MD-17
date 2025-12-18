import numpy as np
from pagerank_fonctions import pageRankLinear, pageRankPower, randomWalk
import matplotlib.pyplot as plt

def main():

    #lecture des fichiers avec la matrice d'adjacence et le vecteur de personnalisation
    A = np.genfromtxt('MatriceAdjacence.csv', delimiter=',', skip_header=1)
    v = np.genfromtxt('VecteurPersonnalisation_Groupe17.csv', delimiter= ',')

    #paramètre de téléportation
    alpha = 0.9

    #appel des fonctions afin de calculer pagerank
    linear = pageRankLinear(A, alpha, v)
    power, P, G, iterations = pageRankPower(A, alpha, v)
    random, error = randomWalk(A, alpha, v)

    #afficher les résultats obtenus
    print("\n Matrice d'Adjacence:")
    print(A)
    print("_____________________________________________")
    print("\n Matrice de transition de probabilités P:")
    print(P)
    print("_______________________________________________")
    print("\n Matrice Google G:")
    print(G)
    print("_______________________________________________")

    print('\n Itérations de la Power Method:')
    for i, vec in enumerate(iterations):
        print("\n Itération {}:".format(i))
        print(vec)
    print("_______________________________________________")
    
    print('\n This is Linear Method')
    print(linear)
    print
    print('\n This is Power Method')
    print(power)
    print
    print('\n This is RandomWalk Method')
    print(random)

    plt.plot(error)
    plt.title("Graphique des Erreurs par RandomWalk")
    plt.xlabel("Nombre d'Itérations de la Marche Aléatoire")
    plt.ylabel("Erreur Moyenne : ϵ")
    plt.show()

    x = range(10)
    plt.plot(x, power, color="green", label="Power")
    plt.plot(x, linear, color="orange", linestyle="--", label="Linear")
    plt.plot(x, random, color="black", label="Random Walk")
    plt.title('Comparaisons des Résolutions de "PageRank"')
    plt.xlabel("Noeud N")
    plt.ylabel("Score Pagerank")
    plt.legend()
    plt.show()

if (__name__ == '__main__'):
    main()