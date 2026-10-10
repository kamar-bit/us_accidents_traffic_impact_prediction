# Variables et disponibilité de l'information

Ce dictionnaire correspond au CSV vérifié et aux entrées enregistrées dans
`results/frozen_selection.json`.

## Cible, pas une entrée du modèle

Severity est une échelle ordonnée de 1 à 4 décrivant l'impact sur le trafic.
Le projet prédit `Severity >= 3` : 3-4 contre 1-2, pas les quatre classes.
Voir la [documentation de la cible](DOCUMENTATION_CIBLE.md)
pour la définition officielle et le caractère conventionnel du seuil.

## Les 33 variables admissibles

| Groupe | Nombre | Variables |
| --- | ---: | --- |
| Temps | 4 | `hour`, `day_of_week`, `month`, `weekend`, dérivées de `Start_Time` |
| Géographie | 4 | `Start_Lat`, `Start_Lng`, `State`, `Timezone` |
| Route | 12 | `Amenity`, `Bump`, `Crossing`, `Give_Way`, `Junction`, `No_Exit`, `Railway`, `Roundabout`, `Station`, `Stop`, `Traffic_Calming`, `Traffic_Signal` |
| Luminosité | 4 | `Sunrise_Sunset`, `Civil_Twilight`, `Nautical_Twilight`, `Astronomical_Twilight` |
| Météo | 9 | `Temperature(F)`, `Wind_Chill(F)`, `Humidity(%)`, `Pressure(in)`, `Visibility(mi)`, `Wind_Speed(mph)`, `Precipitation(in)`, `Wind_Direction`, `Weather_Condition` |

Ce sont des entrées logiques, pas nécessairement indépendantes ni 33 colonnes
après encodage. Vocabulaire des catégories et imputations sont appris sur le train.
Les directions NE et NW ne sont pas réduites à leur première lettre.

`Weather_Timestamp` contrôle la disponibilité : la météo hors de
`[Start_Time - 120 minutes, Start_Time]` est masquée sans supprimer l'accident.
49,78 % des observations météo datées du fichier complet sont postérieures
au début. Même une observation antérieure ne garantit pas sa publication à temps.

## Dérive des catégories météo

**Statut : diagnostic exécuté le 6 octobre 2026 dans le
[notebook principal](../notebooks/MAIN_NoteBook.ipynb).** Le vocabulaire et le
modèle historiques restent figés. La comparaison des plafonds de catégories
reste à réaliser ; aucun gain lié à un autre regroupement n'est établi.

### Règle actuelle et risque de perte d'information

Dans [`FeatureBuilder`](../traffic_impact/experiment.py), le vocabulaire est
appris sur l'échantillon d'apprentissage 2016-2019, après le contrôle de
disponibilité météo. Une catégorie doit apparaître au moins **100 fois** pour
être conservée ; pour `Weather_Condition`, seules les **15 plus fréquentes**
parmi celles admissibles sont retenues. Les autres valeurs non manquantes,
y compris les catégories nouvelles des années suivantes, deviennent `Other`.
Les valeurs absentes ou masquées par le contrôle temporel deviennent `Missing`.

Une catégorie rare ou absente pendant l'apprentissage peut devenir fréquente
et informative plus tard. Le modèle figé ne peut pas apprendre son effet
spécifique lorsqu'elle reste regroupée dans `Other`. Cette perte de signal
est un risque à mesurer, pas une explication déjà démontrée des résultats.

### Effectifs mesurés dans les échantillons

Pour `Weather_Condition`, après le contrôle temporel et avant le regroupement :

| Année | Événements | Conservées | Connues regroupées | Inédites | Manquantes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2020 | 50 000 | 23 952 | 1 126 | 6 | 24 916 |
| 2021 | 99 999 | 47 777 | 1 954 | 12 | 50 256 |
| 2022, jusqu'au 8 septembre | 50 000 | 23 915 | 1 081 | 7 | 24 997 |

Les catégories connues regroupées et inédites deviennent toutes `Other`.
Leur part totale est de 2,264 % en 2020, 1,966 % en 2021 et 2,176 % en 2022.
Les catégories inédites seules représentent 0,012 %, 0,012 % et 0,014 % de
tous les événements, soit respectivement 0,0239 %, 0,0241 % et 0,0280 % des
observations météo non manquantes admissibles. Le dénominateur change donc
selon le taux présenté. Les valeurs manquantes concernent environ la moitié
des événements, notamment après masquage de la météo non admissible.

Pour `Wind_Direction`, toutes les directions non manquantes de ces trois
périodes sont conservées : aucune catégorie connue regroupée ou inédite
n'a été observée dans les échantillons.

Les métriques d'un sous-groupe nécessitent au moins 100 événements et les deux
classes de cible. Elles ne sont donc pas calculées pour les 6, 12 et 7 événements
aux conditions météo inédites. Sur 2021, l'AP est de 0,7160 pour les catégories
conservées, de 0,7407 pour les connues regroupées et de 0,6937 pour les valeurs
manquantes ; leurs taux de cible sont respectivement de 34,00 %, 43,86 % et
34,16 %. Ces différences de population empêchent d'attribuer les écarts d'AP
au regroupement seul. Le top 10 % est sélectionné au sein de chaque sous-groupe.

Les exports complets sont `weather_category_drift.csv`,
`weather_category_details.csv` et `weather_subgroup_scores.csv`, sous
`data/main_notebook/reports/`. Ils comprennent les années d'apprentissage,
les fréquences, les taux de cible par catégorie et les métriques calculables.
Ces résultats sont descriptifs et ne démontrent pas une perte de signal
causée par `Other`.

Le regroupement est effectué explicitement par notre préparation, avant
l'encodage. `OneHotEncoder(handle_unknown="ignore")` ne crée pas lui-même
la catégorie `Other` : une valeur inconnue de cet encodeur produit des zéros
pour la variable concernée. Scikit-learn propose aussi `min_frequency` et
`max_categories` pour regrouper les catégories peu fréquentes. Voir la
[documentation de OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).

### État des vérifications

- [x] Conserver le vocabulaire et le modèle appris sur **2016-2019 uniquement**,
  et les appliquer sans réapprentissage à 2020, 2021 et la période disponible
  de 2022. Conserver aussi l'ensemble des catégories observées dans le train
  avant leur regroupement pour distinguer rareté et nouveauté.
- [x] Pour chaque année, compter séparément les catégories conservées, celles
  vues dans le train mais regroupées dans `Other`, celles jamais vues dans
  le train, et les valeurs `Missing`. Utiliser les valeurs après le contrôle
  de disponibilité météo mais avant le regroupement.
- [x] Calculer le **taux de catégories inédites** : nombre d'observations de
  l'année avec une catégorie météo non manquante absente du train, divisé par
  le nombre total d'observations de cette année. Rapporter aussi ce taux parmi
  les seules observations ayant une météo admissible et non manquante, pour
  éviter que le masquage météo ne cache la dérive.
- [x] Rapporter de même la proportion totale de `Other`, en distinguant les
  catégories connues mais regroupées des catégories inédites. Ne pas confondre
  ce taux avec celui des catégories absentes du train.
- [x] Pour chaque catégorie nouvelle ou regroupée, afficher les effectifs,
  fréquences et proportions avec `Severity >= 3`, en les comparant au train
  lorsqu'elle y existe et à la prévalence de l'année. Interpréter prudemment
  les petits effectifs ; ces associations ne sont pas des effets causaux.
- [x] Examiner les performances du modèle figé sur ces sous-groupes, lorsque
  leurs effectifs et leurs classes permettent les métriques, et leur évolution
  annuelle. Une hausse de `Other` peut signaler une dérive temporelle, mais
  ne prouve pas à elle seule qu'elle cause une baisse de performance. Un taux
  faible peut aussi masquer une catégorie importante.
- [ ] Étudier, comme **analyse complémentaire**, des plafonds de 10, 15 et
  25 catégories ou un seuil de fréquence. Pour chaque variante, apprendre
  le vocabulaire uniquement sur le train, sélectionner sur la validation 2020,
  puis figer le choix. Documenter aussi le seuil actuel de 100 observations ;
  ne pas présenter le plafond de 15 comme optimal sans comparaison.
- [x] Distinguer ce diagnostic rétrospectif d'une nouvelle évaluation indépendante.
  Les scores de 2021 et 2022 ont déjà été consultés : une adaptation motivée
  par leur analyse ne permettrait pas de les présenter comme des tests inédits.

Ne pas utiliser les catégories des années futures pour construire le vocabulaire
du modèle historique : cela introduirait de l'information future dans la
préparation et compromettrait le protocole temporel. Un découpage aléatoire
mélangeant les années pourrait masquer ce risque plutôt que le résoudre.

Dans un système déployé, un réentraînement périodique pourrait actualiser le
vocabulaire avec les données historiques devenues disponibles. Il faudrait
alors évaluer chaque version sur une période ultérieure, sans réécrire les
résultats du modèle historique figé.

## Exclusions du modèle principal

- Fin, durée et coordonnées de fin : non admissibles pour notre scénario au début.
- `Distance(mi)` : exclusion conservatrice, disponibilité initiale non établie.
  Une association avec Severity ne prouve pas une fuite.
- `Description` : texte libre hors périmètre, pouvant révéler l'impact.
- `ID` : identité et doublons uniquement.
- `Source` : constante dans l'étude Source2, pas un prédicteur.
- Adresses fines et code de station : exclusions du protocole, pas une preuve
  qu'ils seraient toujours inutiles ou nécessairement des fuites.
- `Country` et `Turning_Loop` : constants dans ce fichier.

La clé de déduplication utilise notamment fin, distance et description pour
reconnaître un événement répété. Cela ne les introduit pas dans les prédicteurs.

Les ablations montrent l'information prédictive supplémentaire de chaque groupe,
pas un effet causal. La faible contribution marginale de certains groupes est
une limite à expliquer, pas un motif pour changer l'objectif du projet.
