# Wiki personnel pour revenir facilement aux notions abordées.


## **Définition d'un réseau de neurones**

Un réseau de neurones est un type de modèle informatique inspiré de la façon dont fonctionne le cerveau humain. Il est composé de plusieurs couches de "neurones" connectés entre eux qui travaillent ensemble pour effectuer des tâches telles que la reconnaissance d'images ou de la génération de texte.

Chacun des neurones dans un réseau de neurones est relié à plusieurs autres neurones, et chacun de ces liens associé à un poids qui détermine la force de l'influence d'un neurone sur un autre.

Les réseaux de neurones peuvent être "entraînés" en utilisant des données d'entraînement et des algorithmes de rétropropagation pour ajuster les poids de manière à améliorer les performances du modèle.


## **Fonction d'activation**

Une fonction d'activation est une fonction mathématique utilisée dans les réseaux de neurones pour décider si un neurone doit être activé ou non. Elle prend en entrée la somme pondérée des entrées d'un neurone et produit une sortie qui peut être utilisée comme entrée pour les neurones suivants dans la réseau.

Il existe différentes fonctions d'activation comme la fonction sigmoïde, la fonction ReLU et la fonction tangente hyperbolique.

### **Fonction d'activation ReLU**

Elle permet de ne garder que les valeurs positives en sortie des neurones. Elle améliore la convergence du réseau et évite le problème du gradient vanishing.

$$\large
f(x) = max(0, x) = \begin{cases}
0 & \text{si } x \leq 0 \\
x & \text{si } x > 0 \\
\end{cases}
$$

### **Fonction d'activation Softmax**

La fonction d'activation softmax est souvent utilisée dans les réseaux de neurones pour les tâches de classification multiclasse. Elle prends en entrée un vecteur de valeurs réelles et produit en sortie un vecteur de probabilités, c'est-à-dire une distribution de probabilité sur les classes possibles. Chaque élément de la sortie est compris entre 0 et 1 et la somme des éléments vaut 1.

On écrit sa fonction comme suit:
$$\large
    f_i(x) = \frac{e^{x_i}}{\sum_{j=1}^{k}e^{x_j}}
$$

On l'utilise généralement pour un problème de classification.


## **Fonction de perte et rétropropagation**


### **Fonction de perte Categorical cross-entropy**

La fonction de perte categorical cross entropy est une fonction de coût couramment utilisée pour les tâches de classification multiclasse. Elle mesure la différence entre la distribution de probabilité prédite par le modèle et la distribution de probabilité réelle pour chaque classe.

On écrit sa fonction comme suit:

$$\large
    L_i = -\sum_j y_{i,j} log(\hat{y}_{i,j})
$$

On l'utilise souvent en conjonction avec la fonction d'activation softmax.


## Optimisation d'algorithme

### Softmax et Categorical cross-entropy.

La fonction d'activation softmaw et la fonction de perte categorical cross-entropy peut être simplifier et on peut les calculer plus vite. Voici les équations.

<!-- TODO: Expliqué comment on fait et pourquoi on le fait. -->

### Comment interpréter les courbes de Loss ?

![Learning rate courbe](img/Learning_rate.png)