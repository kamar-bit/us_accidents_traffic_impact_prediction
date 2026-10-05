# Définition de Severity et cible retenue

## Source officielle

[Fiche du créateur du dataset, section Severity](https://www.kaggle.com/sobhanmoosavi/us-accidents/metadata).
Documentation consultée le 3 octobre 2026. Les proportions ci-dessous proviennent
de notre audit du CSV, pas de la page Kaggle.

Severity prend les valeurs entières **1, 2, 3 et 4**, par impact croissant sur
le trafic. La documentation décrit 1 comme le niveau d'impact minimal avec
retard court et 4 comme un impact important avec retard long. 2 et 3 sont
intermédiaires. Elle ne donne pas de frontières chiffrées en minutes.

Ce n'est pas une mesure de blessures ou de mortalité. L'échelle est ordonnée :
un niveau 4 n'est pas nécessairement deux fois plus important qu'un niveau 2.

## Choix confirmé : Severity >= 3

- `y = 1` pour les niveaux **3 ou 4**.
- `y = 0` pour les niveaux **1 ou 2**.

« Fort impact » désigne ce regroupement des deux niveaux supérieurs.
Le seuil 3 est une **définition d'étude**, pas une frontière opérationnelle
officielle. Le modèle ne distingue pas séparément les quatre niveaux, et
ne détecte pas uniquement la classe 3 ou uniquement la classe 4.

Le regroupement conserve l'ordre de l'échelle et correspond à notre objectif
de priorisation. Il ne doit pas être changé pour obtenir de meilleurs scores.
Ce choix est conservé dans toutes les expériences déjà exécutées.

Le seuil de la cible et le seuil d'alerte sont distincts : `Severity >= 3`
définit ce qu'on prédit. Les 10 % de scores les plus élevés définissent le
budget d'examen rétrospectif utilisé dans l'évaluation.

## Répartition réellement mesurée

Toutes les lignes avant nettoyage et échantillonnage. Agrégation des effectifs
de `monthly_source_severity.csv` par niveau et périmètre.

| Severity | Fichier complet : effectif | Proportion | Source2, 2016-2021 : effectif | Proportion |
| --- | ---: | ---: | ---: | ---: |
| 1 | 67 366 | 0,87 % | 1 258 | 0,04 % |
| 2 | 6 156 981 | 79,67 % | 2 038 974 | 66,07 % |
| 3 | 1 299 337 | 16,81 % | 1 032 841 | 33,47 % |
| 4 | 204 710 | 2,65 % | 12 889 | 0,42 % |

La classe positive représente environ **33,89 %** de Source2 en 2016-2021,
contre **19,46 %** du fichier complet. Dans ce périmètre Source2, environ
**98,77 % des positifs sont de niveau 3**. Les résultats binaires ne prouvent
donc pas à eux seuls la bonne détection du niveau 4. Les proportions des
échantillons nettoyés peuvent différer légèrement.

## Question et limites

Peut-on identifier, au début d'un accident, les incidents classés aux niveaux
3 ou 4 d'impact sur le trafic à partir de leur contexte géographique, routier,
temporel, lumineux et météorologique ? Les performances se maintiennent-elles
l'année suivante ?

Le modèle complet à 33 variables est central. Les règles sont des références
de non-trivialité et les ablations des explications, pas un objectif de réduction.

`Start_Time` définit le début dans cette étude rétrospective. La météo est
masquée sauf si son observation appartient aux deux heures précédentes.
Le fichier ne garantit ni l'heure de publication de chaque information ni
le moment d'attribution de Severity. On prédit l'étiquette enregistrée, pas
des minutes de bouchon ni une évolution future mesurée indépendamment.
Le déploiement au premier signalement n'est pas validé par cette expérience.

Le modèle fournit un score de classement ; ses probabilités n'ont pas fait
l'objet d'une validation de calibration. Ces réserves ne modifient pas la
cible ni les résultats déjà calculés.
