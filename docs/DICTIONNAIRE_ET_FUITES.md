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
