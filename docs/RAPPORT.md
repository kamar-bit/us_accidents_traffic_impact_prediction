# Prédire l'impact sur le trafic au début d'un accident

Le modèle complet à 33 variables atteint sur 2021 une AP de 0.7031, une ROC-AUC de 0.8362 et une précision de 80.13 % parmi les 10 % d'incidents les mieux classés. Sur la période disponible de 2022, son AP reste de 0.7092.

## Résultat principal : modèle complet

Question : peut-on identifier au début d'un accident les incidents classés aux niveaux 3 ou 4 d'impact sur le trafic, à partir du contexte, et conserver les performances l'année suivante ?

Le modèle principal est le boosting complet à 33 variables admissibles. Les règles simples sont des références de non-trivialité ; les ablations expliquent le modèle. La réduction du nombre de variables n'est pas l'objectif central. Ce recadrage intervient après les expériences et ne change ni les scores ni les choix figés.

| name | ap | auc | recall10 | precision10 |
| --- | --- | --- | --- | --- |
| prior | 0.3427 | 0.5 | 0.0977 | 0.335 |
| logistic_full | 0.5667 | 0.7344 | 0.1938 | 0.6643 |
| hgb_full | 0.7031 | 0.8362 | 0.2338 | 0.8013 |

AP et AUC sont des mesures de classement, pas des pourcentages de prédictions correctes. Dans les 10 % de scores les plus élevés sur 2021, 80.13 % des événements sont positifs ; ce budget retrouve 23.38 % de tous les positifs. Le modèle n'identifie donc pas tous les incidents d'impact supérieur. Les probabilités n'ont pas été validées par calibration.

## Documentation de la cible confirmée

Voir [DOCUMENTATION_CIBLE.md](DOCUMENTATION_CIBLE.md). Son contenu est repris ici pour que le rapport reste autonome.

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

## Protocole et données

Les comparaisons temporelles évaluent le modèle complet, pas une recherche de la plus petite liste de variables.

L'audit a lu 7,728,394 lignes et 46 colonnes. La tâche porte sur l'impact trafic conditionnellement à un accident enregistré, avec la cible binaire `Severity >= 3`. Elle ne mesure pas la gravité corporelle.

Source2 uniquement. Train 2016-2019 (199,999 événements), validation 2020 (50,000), test 2021 (99,999), stress 2022 (50,000). Source2 s'arrête au 2022-09-08 dans ce fichier : le stress ne couvre pas toute l'année 2022.

Échantillonnage uniforme d'événements distincts par priorités de hachage, graine 42, sans équilibrage des classes. 2,136 répétitions d'événement et 8 clés contradictoires identifiées sur Source2. Les répétitions sont regroupées ; les clés contradictoires sont exclues. Les tailles peuvent être légèrement inférieures aux plafonds après exclusions. Identifiants dupliqués : 0.

La clé exacte inclut source, début, fin, coordonnées de départ, distance et description. Ces champs servent à reconnaître un enregistrement répété ; ils ne sont pas tous des prédicteurs. Les hachages 64 bits présentent un risque négligeable de collision. La sensibilité aux collisions dans des cellules de 0,001 degré et 15 minutes est rapportée séparément : elles ne prouvent pas l'identité de deux accidents.

Les transformations apprennent uniquement sur le train. La météo est masquée si son observation n'appartient pas à [Start_Time - 120 minutes, Start_Time]. Sur le CSV complet, 49.78 % des observations météo datées sont postérieures au début. Aucune ligne n'est retirée pour ce motif. L'horaire de publication des observations reste inconnu : il s'agit d'une hypothèse de disponibilité, pas d'un simulateur de premier signalement.

`Country` et `Turning_Loop` sont constants dans le fichier. Variables constantes supplémentaires supprimées sur le train : Turning_Loop. Nombre de variables admissibles : 33. Aucun texte libre, adresse fine, identifiant, source, durée ou distance dans les concurrents principaux.

## Diagnostics complémentaires : règles et relations mathématiques

33 règles univariées et 528 paires ont été évaluées sur la même validation. Chaque règle numérique cherche le seuil maximisant l'AP sur le train, avec au moins 1 % des lignes de chaque côté. Les tables catégorielles sont lissées avec 50 observations de poids équivalent vers la moyenne du train. Les valeurs inconnues utilisent cette moyenne. Les arbres de paire ont au plus quatre feuilles et trois niveaux, avec au moins 1 % du train dans chaque feuille.

Meilleure variable pour la règle univariée : **State**. Meilleure paire pour l'arbre simple : **hour, Traffic_Signal**. Une variable catégorielle peut représenter plusieurs modalités : le nombre de variables ne résume pas toute la complexité de la règle. Les taux par catégorie sont dans `results/best_single_category_rates.csv`.

Arbre de la meilleure paire (les modalités sont encodées en indicatrices, les poids indiquent les classes 0 et 1) :

```text
|--- Traffic_Signal <= 0.5000
|   |--- hour <= 11.5000
|   |   |--- hour <= 3.5000
|   |   |   |--- weights: [1660.0000, 1768.0000] class: 1
|   |   |--- hour >  3.5000
|   |   |   |--- weights: [45855.0000, 27683.0000] class: 0
|   |--- hour >  11.5000
|   |   |--- weights: [41300.0000, 34871.0000] class: 0
|--- Traffic_Signal >  0.5000
|   |--- weights: [42371.0000, 4491.0000] class: 0
```

Les régressions logistiques à une variable, deux variables et toutes les variables testent des relations additives. Le boosting sur les cinq meilleures paires teste des relations plus complexes avec très peu d'entrées. Cela évite de conclure que beaucoup de variables sont nécessaires simplement parce qu'un seuil isolé est trop restrictif.

Meilleures variables isolées sur 2020 :

| variable | group | ap |
| --- | --- | --- |
| State | geography | 0.4082 |
| Traffic_Signal | road | 0.3614 |
| hour | time | 0.3524 |
| day_of_week | time | 0.3460 |
| weekend | time | 0.3460 |
| Crossing | road | 0.3394 |
| Start_Lat | geography | 0.3346 |
| Weather_Condition | weather | 0.3227 |
| Junction | road | 0.3170 |
| Astronomical_Twilight | light | 0.3133 |

## Test final 2021

| Modèle | Variables | AP | Rappel, top 10 % | Précision, top 10 % |
| --- | --- | --- | --- | --- |
| prior | 0 | 0.3427 | 9.77 % | 33.50 % |
| best_single_rule | 1 | 0.4118 | 14.25 % | 48.84 % |
| best_pair_rule | 2 | 0.4701 | 15.04 % | 51.55 % |
| logistic_full | 33 | 0.5667 | 19.38 % | 66.43 % |
| hgb_pair_2 | 2 | 0.4991 | 15.06 % | 51.61 % |
| geo_pair_Start_Lat_Start_Lng | 2 | 0.6046 | 19.78 % | 67.78 % |
| pair_plus_geography | 6 | 0.6827 | 22.85 % | 78.31 % |
| hgb_full | 33 | 0.7031 | 23.38 % | 80.13 % |

Référence AP sans information : prévalence de 0.3427. Le rappel à 10 % sélectionne un budget identique pour tous les modèles, avec égalités départagées par une priorité indépendante de la cible. C'est un classement rétrospectif. La probabilité constante conserve le taux du train ; sa calibration peut différer de celle du test.

Face à hgb_pair_2, différence de rappel : 8.32 points, IC 95 % [7.93 ; 8.69]. Différence d'AP : 0.2040, IC [0.1984 ; 0.2091]. Intervalles appariés par rééchantillonnage de semaines, 500 tirages.

Le critère historique d'utilité comparant la complexité avait été fixé à cinq points supplémentaires de rappel, un intervalle de différence excluant zéro et aucune baisse d'AP. Il est conservé pour interpréter cette expérience secondaire, pas comme nouveau critère de réussite de la question prédictive. Il est pédagogique, sans interprétation économique. Un intervalle qui englobe zéro ne démontre pas une équivalence.


## Analyses complémentaires : références à peu de variables

Le screening des paires par arbres de quatre feuilles peut manquer une relation spatiale complexe. La forte performance du groupe géographique sur validation a donc motivé un contrôle supplémentaire de trois paires : latitude-longitude, latitude-État et longitude-État. Ces contrôles sont ajoutés après l'expérience principale et ne remplacent pas sa sélection prédéclarée. Leurs hyperparamètres reprennent ceux du modèle complet ; leur gagnant est choisi exclusivement sur 2020 avant leurs scores de test.

La paire latitude-longitude atteint une AP de 0.6046 sur 2021. Face à ce concurrent, le gain du modèle complet en rappel est de **3.60 points**, IC [3.27 ; 3.85], donc inférieur au seuil de cinq points. Le gain d'AP reste de 0.0985. Sur 2022, le gain en rappel est de 3.45 points.

Il faut ainsi distinguer trois conclusions : les règles mathématiques simples testées ne reproduisent pas le modèle complet ; un modèle non linéaire à deux coordonnées est déjà performant ; toutes les autres variables combinées améliorent encore l'AP, mais leur gain opérationnel marginal dépend du budget d'alertes retenu. Le verdict favorable à la comparaison primaire ne suffit pas à dire que 33 variables sont indispensables.


Une autre sensibilité choisit le concurrent à deux variables sur le rappel de 2020 au lieu de l'AP : hgb_pair_4. Le gain du modèle complet sur 2021 est alors de 4.40 points, IC [3.66 ; 5.12]. Le choix de la métrique de sélection modifie donc l'interprétation opérationnelle.


## Apport des groupes et du volume

- time : gain d'AP de 0.0581 sur 2021, en comparant le modèle complet à sa variante sans ce groupe.
- geography : gain d'AP de 0.1369 sur 2021, en comparant le modèle complet à sa variante sans ce groupe.
- road : gain d'AP de 0.0379 sur 2021, en comparant le modèle complet à sa variante sans ce groupe.
- light : gain d'AP de 0.0005 sur 2021, en comparant le modèle complet à sa variante sans ce groupe.
- weather : gain d'AP de -0.0015 sur 2021, en comparant le modèle complet à sa variante sans ce groupe.

Le modèle à six variables (`hour`, `Traffic_Signal`, `Start_Lat`, `Start_Lng`, `State`, `Timezone`) atteint une AP de 0.6827 et un rappel de 22.85 % sur 2021. Le modèle complet atteint 0.7031 et 23.38 %. Ce diagnostic montre des redondances ; il ne démontre pas un optimum et ne remplace pas le modèle complet comme résultat principal.

Ces différences sont des associations prédictives conditionnelles aux autres entrées, pas des effets causaux de la météo ou de la route. Les variantes utilisent les hyperparamètres du modèle complet, sans réglage spécifique à chaque ablation. Les permutations de groupes peuvent créer des combinaisons irréalistes et doivent être lues avec les ablations réentraînées. L'ordre de la courbe de variables a été fixé sur validation, en partant de la meilleure paire puis des règles univariées les mieux classées. Ce n'est pas une recherche exhaustive de tous les sous-ensembles de cinq ou dix variables.

Les figures `02_feature_budget.png`, `03_group_ablation.png` et `04_learning_curve.png` montrent le gain par budget, par groupe et par taille du train. Le modèle d'indicateurs de manquants est dans la table complète : il peut signaler des particularités de collecte.

## Stress 2022 et contrôles

| Modèle | Variables | AP | Rappel, top 10 % | Précision, top 10 % |
| --- | --- | --- | --- | --- |
| prior | 0 | 0.3470 | 10.19 % | 35.34 % |
| best_single_rule | 1 | 0.4305 | 15.19 % | 52.72 % |
| best_pair_rule | 2 | 0.4768 | 15.29 % | 53.06 % |
| logistic_full | 33 | 0.5727 | 19.43 % | 67.40 % |
| hgb_pair_2 | 2 | 0.5081 | 15.72 % | 54.54 % |
| geo_pair_Start_Lat_Start_Lng | 2 | 0.6149 | 19.89 % | 69.02 % |
| pair_plus_geography | 6 | 0.6937 | 22.84 % | 79.26 % |
| hgb_full | 33 | 0.7092 | 23.35 % | 81.00 % |

Gain utile selon le critère fixé avant les résultats. Face au même concurrent figé : 7.63 points de rappel, IC [7.11 ; 8.11]. Aucun réglage sur cette période. Les résultats mensuels et par État sont dans `monthly_state_scores.csv` ; ne pas comparer des États sans considérer leur prévalence et leur effectif.

Contrôle d'étiquettes mélangées : AP 0.3590, AUC 0.5199 sur 2021. L'AP attendue sans information est proche de la prévalence. Des écarts finis existent ; ce contrôle seul ne garantit pas l'absence de toute fuite.

Cinq graines de mélange donnent sur 2021 une AP comprise entre 0.3445 et 0.3610, et une AUC entre 0.5045 et 0.5199. Ces faibles écarts au hasard ne sont pas exactement nuls, mais restent loin de l'AP du modèle réel. Voir `shuffled_label_replicates.csv`.

Audit des variables exclues sur 2021 :

| name | ap | recall10 |
| --- | --- | --- |
| diagnostic_Distance(mi) | 0.3718 | 0.1522 |
| diagnostic_duration_min | 0.3626 | 0.1254 |
| diagnostic_full_future | 0.7063 | 0.2337 |

Ces scores diagnostiques ne sont pas en compétition avec les variables autorisées. Leur association avec la cible ne prouve pas à elle seule une fuite, et leur faible performance ne garantit pas leur disponibilité initiale. Les règles à quatre classes sur distance et durée sont décrites dans `future_information_4class.csv`.

Sensibilité aux collisions spatiotemporelles :

| split | name | excluded_within_holdout | ap | recall10 |
| --- | --- | --- | --- | --- |
| test | hgb_full | 3723 | 0.6959 | 0.2381 |
| test | hgb_pair_2 | 3723 | 0.4888 | 0.1524 |
| test | near_deduplicated_train | 3723 | 0.6967 | 0.2389 |
| stress | hgb_full | 1473 | 0.7042 | 0.2357 |
| stress | hgb_pair_2 | 1473 | 0.5013 | 0.1585 |
| stress | near_deduplicated_train | 1473 | 0.7047 | 0.2372 |

L'entraînement avec un représentant par cellule est comparé au modèle complet. Les collisions avec le train sont nulles ou rares par construction des périodes disjointes ; la déduplication interne des périodes est le contrôle informatif ici.

## Reproductibilité, originalité et limites

Sélection figée avant calcul des scores de 2021 et 2022 dans `frozen_selection.json`. Modèles sauvegardés dans `frozen_models.joblib`, prédictions et priorités dans `predictions_test.csv` et `predictions_stress.csv`. Hyperparamètres HGB : {"max_leaf_nodes": 31, "max_iter": 200}. Sans poids de classe et sans arrêt anticipé utilisant un split aléatoire. Durée des expériences : 315 secondes. Versions : {"sklearn": "1.7.2", "numpy": "2.3.5", "pandas": "3.0.1"}.

La contribution du projet est une prédiction par modèle complet avec évaluation temporelle, contrôle des informations admissibles et analyse des limites. Les règles et budgets d'information sont des diagnostics de non-trivialité. Le sujet et l'évaluation temporelle existent déjà : [projet comparable](https://github.com/Simranghag22/us-accidents-severity-prediction), [autre projet de cours](https://github.com/JustinJ-Lo/vehicle-severity-ml). Leurs scores ne sont pas directement comparables aux nôtres. Définition de la cible : [fiche du créateur](https://www.kaggle.com/sobhanmoosavi/us-accidents/metadata).

Limites : fournisseur unique, informations de trafic et type exact de route absents, disponibilité au signalement non garantie, observations limitées aux accidents collectés. Une faible performance ne prouve pas l'absence de signal ni l'inutilité universelle du ML. La fréquence du trafic et le nombre d'accidents sans exposition routière ne permettent pas d'estimer le risque de circulation.

Les cellules actualisées en anglais sont proposées dans CELLULES_NOTEBOOK.md. Le notebook existant conserve son ancien récit tant qu'elles ne sont pas intégrées manuellement ; il n'est pas remplacé par défaut. L'archive reste donc un dossier de travail, pas un rendu final recadré. Ajouter les noms des membres, joindre les modules et ne pas joindre le CSV ou les caches. Limite du notebook : 10 Mo.
