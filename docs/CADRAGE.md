# Cadrage de la tâche

## Question principale

Peut-on identifier, au début d'un accident, les incidents classés aux niveaux
3 ou 4 d'impact sur le trafic à partir de leur contexte, et les performances
se maintiennent-elles l'année suivante ?

L'objectif est un **modèle prédictif complet à 33 variables admissibles**.
Les modèles simples sont des références ; les ablations permettent de comprendre
l'information utilisée. La recherche du modèle minimal n'est pas l'objectif central.

## Cible confirmée

Severity va de 1 à 4. On prédit `y = (Severity >= 3)` :
3 et 4 sont positifs, 1 et 2 sont négatifs. « Fort impact » désigne ce regroupement,
pas une frontière officielle en minutes. On ne prédit ni les quatre classes
séparément, ni les blessures, ni la durée, ni la survenue d'un accident.

La [documentation de la cible](DOCUMENTATION_CIBLE.md) donne
la source officielle et les proportions mesurées. Dans Source2 entre 2016 et
2021, environ 33,89 % des lignes sont positives, presque toutes de niveau 3.
Les scores binaires ne démontrent pas à eux seuls une bonne détection du niveau 4.

## Instant de prédiction et population

`Start_Time` représente le début dans cette étude rétrospective. Les variables
futures sont exclues. La météo n'est retenue que si elle est observée dans les
deux heures précédentes. L'heure réelle de publication et l'instant d'attribution
de Severity sont inconnus : le fichier ne prouve pas la faisabilité au premier
signalement en temps réel.

Source2 est retenue pour limiter l'hétérogénéité entre fournisseurs.
Apprentissage 2016-2019, sélection 2020, test 2021 et stabilité jusqu'au
8 septembre 2022. Les conclusions concernent les accidents collectés par cette source.

## Intérêt du projet

L'audit réel porte sur 7 728 394 lignes et 46 colonnes. L'apprentissage utilise
jusqu'à 200 000 événements distincts, sans équilibrer artificiellement les tests.
Le boosting complet obtient une AP de 0,7031 sur 2021, contre 0,3427 pour
la référence sans information et 0,5667 pour la logistique complète.
La tâche présente un signal que ces références ne reproduisent pas.
Cela ne prouve pas qu'aucune formule concurrente ne puisse exister.

La contribution est une évaluation temporelle rigoureuse, l'audit des informations
admissibles et l'analyse du modèle complet, pas une nouvelle méthode de ML.
Les résultats ne prouvent ni que les 33 variables sont toutes nécessaires,
ni un effet causal des variables.

## Livrable

Un notebook en anglais sous 10 Mo avec question, cible documentée, préparation,
modèle complet, références, évaluation temporelle et limites.
Joindre les modules utilisés, pas le dataset. Le notebook du dépôt GitHub
reprend déjà le cadrage du modèle complet. Il reste conservé sans modification directe.
Ce recadrage du récit intervient après les scores et ne les modifie pas.
