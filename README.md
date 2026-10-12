# US Accidents : prédire l'impact d'un accident sur le trafic

**Rendu du groupe : [FINAL_NOTEBOOK.ipynb](notebooks/FINAL_NOTEBOOK.ipynb), code et rapport en anglais.**

Le notebook final réunit les contributions de Yanis et Kamar, une préparation
progressive et les comparaisons communes. Il utilise 33 entrées logiques,
500 000 événements et une actualisation météo dans l'heure, avec le scénario
strict au début en comparaison. Le [mode d'emploi](docs/MODE_EMPLOI_NOTEBOOK_FINAL.md)
explique la reproduction et l'archive de remise.

Projet du module *Hands-on Machine Learning* (Master 1 Informatique - IA).
Membres du groupe : **Yanis Barbieux, Kamar Benbacha, Théophile Melquiot**.

## Étude historique

Les sections ci-dessous décrivent le protocole initial et ses résultats,
reproduits dans [MAIN_NoteBook.ipynb](notebooks/MAIN_NoteBook.ipynb).
Le notebook final utilise le protocole commun décrit ci-dessus.

Peut-on identifier, au début d'un accident, les incidents classés aux niveaux
supérieurs d'impact sur le trafic à partir de leur contexte, et les performances
se maintiennent-elles l'année suivante ?

La cible est **`Severity >= 3`**, soit les niveaux **3 et 4** contre **1 et 2**.
Severity décrit l'impact sur le trafic, pas les blessures. Ce regroupement est
un choix du projet, sans frontière officielle en minutes. La
[documentation de la cible](docs/DOCUMENTATION_CIBLE.md) précise la source,
les proportions et les limites, notamment la rareté du niveau 4.

Le modèle principal combine **33 variables admissibles**. Les règles simples
sont des références de non-trivialité ; les ablations et budgets de variables
sont des diagnostics complémentaires. Le cadrage du récit a été revu après
les expériences, sans modifier les choix figés ni les scores.

## Résultat principal

[US Accidents (Kaggle)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents),
Source2 uniquement : apprentissage 2016-2019, validation 2020, test 2021,
stabilité sur la période disponible de 2022, jusqu'au 8 septembre.

| Modèle, test 2021 (99 999 événements) | AP | ROC-AUC | Rappel, top 10 % | Précision, top 10 % |
| --- | ---: | ---: | ---: | ---: |
| Sans information | 0,3427 | 0,5000 | 9,77 % | 33,50 % |
| Logistique complète, 33 variables | 0,5667 | 0,7344 | 19,38 % | 66,43 % |
| Gradient boosting complet, 33 variables | 0,7031 | 0,8362 | 23,38 % | 80,13 % |

Sur 2022, l'AP du modèle complet est de 0,7092. AP et AUC mesurent le classement,
pas le pourcentage de prédictions correctes. La géographie apporte le plus grand
gain d'AP parmi les ablations réentraînées sur 2021. Dans ce protocole, les gains
marginaux de la lumière (+0,0005) et de la météo (-0,0015) sont faibles ; cela
ne démontre pas leur inutilité générale ni un effet causal. Les probabilités
ne sont pas validées par calibration. Voir le [rapport](docs/RAPPORT.md).

### Dernière reproduction vérifiée

Le 6 octobre 2026, le notebook principal a reproduit ces résultats avec
33 variables, un gradient boosting et une référence constante. Les **16 cellules**
ont terminé sans erreur, les **21 tests unitaires** et les **27 contrôles de
cohérence** passent. L'exécution complète, reconstruction des caches comprise,
a duré **244,26 secondes** dans l'environnement Python 3.10.11 et pandas 2.3.3.
Les caches incompatibles sont désormais reconstruits lorsque Python, pandas ou
NumPy changent. Les scores 2021 et 2022 reproduisent les résultats historiques ;
ces périodes ne constituent pas de nouveaux tests indépendants.
Voir le [bilan de validation](docs/MAIN_NOTEBOOK_VALIDATION.md) et les
[commandes de reproduction du notebook](docs/MODE_EMPLOI.md#exécuter-le-notebook-principal).

## Pourquoi retenir Source2 ?

`Source` désigne l'origine des données brutes dans la
[documentation du créateur](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/metadata).
Le [cadrage](docs/CADRAGE.md) retient Source2 pour limiter l'hétérogénéité
entre fournisseurs. Cette justification explique le choix d'une seule source ;
l'audit ci-dessous apporte des éléments pour préférer Source2 à Source1 dans
notre protocole temporel. Il ne démontre pas que Source2 fournit des données
de meilleure qualité ou les meilleures performances prédictives.

### Évolution de la cible selon la source

| Année | Source1 : proportion avec Severity >= 3 | Source2 : proportion avec Severity >= 3 |
| --- | ---: | ---: |
| 2016 | 28,38 % | 36,85 % |
| 2017 | 32,31 % | 36,41 % |
| 2018 | 33,40 % | 35,97 % |
| 2019 | 21,04 % | 30,29 % |
| 2020 | 10,23 % | 30,65 % |
| 2021 | 2,44 % | 34,33 % |
| 2022 | 2,74 % | 34,46 % |

Ces proportions sont calculées en agrégeant les effectifs de
[`monthly_source_severity.csv`](results/audit/monthly_source_severity.csv),
avant nettoyage et échantillonnage. Pour chaque source et année, le numérateur
est le nombre de lignes de niveaux 3 ou 4, et le dénominateur le nombre total
de lignes. Les périodes correspondent aux observations disponibles : Source2
s'arrête en septembre 2022, alors que Source1 couvre toute cette année et
se poursuit jusqu'en mars 2023.

Source1 présente un changement majeur de distribution : un modèle appris
sur 2016-2019 rencontrerait en 2021 une proportion de positifs beaucoup plus
faible. Source2 reste relativement stable, avec environ 30 à 37 % de positifs
selon l'année. Cette stabilité relative facilite l'interprétation de notre
évaluation temporelle ; elle ne garantit pas la stabilité des autres variables
ni de la manière d'attribuer Severity.

L'évolution de Source1 peut refléter des changements de collecte, de zones
couvertes, d'attribution des niveaux ou des événements eux-mêmes. L'audit
disponible ne permet pas d'en déterminer la cause ni de conclure que cette
source serait mauvaise.

### Comparaison des périmètres possibles

| Périmètre | Intérêt | Limite principale | Position retenue |
| --- | --- | --- | --- |
| Source2 seule | Distribution relativement stable, données suffisantes, protocole et résultats déjà disponibles | Généralisation limitée à cette source ; observations arrêtées en septembre 2022 | Périmètre principal de l'étude |
| Source1 seule | Davantage de lignes, données jusqu'en mars 2023, possibilité d'étudier les changements temporels | Forte variation de la cible ; préparation et expériences propres à cette source nécessaires | Pertinent pour une étude de robustesse face à ces changements |
| Source1 + Source2 | Population étudiée plus large et possibilité d'analyser les différences entre sources | Composition du mélange variable dans le temps ; un score global peut masquer de mauvaises performances sur une source | Extension pertinente avec évaluation séparée par source |

Le mélange n'est pas stable : Source2 représente **80,98 %** des lignes de
Source1 + Source2 en 2018, contre **27,92 %** en 2021. Regrouper les deux
sources combine donc des changements internes aux sources et un changement
de leur poids relatif. Retirer la colonne `Source` des prédicteurs ne suffit
pas à rendre les deux populations homogènes.

Les positifs ne regroupent pas non plus les mêmes proportions de niveaux.
Sur **2016-2021**, le niveau 4 représente **46,67 % des positifs de Source1**,
contre **1,23 % des positifs de Source2**, avant nettoyage. Le modèle Source2
apprend donc essentiellement à reconnaître le niveau 3. Si l'objectif était
spécifiquement la détection du niveau 4, le choix de la cible et du périmètre
devrait être réexaminé. Ces proportions sont également issues de l'audit
mensuel, pas d'une comparaison des performances de modèles.

### Choix principal et extension possible

Nous conservons **Source2 pour l'étude principale**, en justifiant ce périmètre
par l'audit et en limitant nos conclusions à ses accidents enregistrés.
**Source1 constitue une comparaison complémentaire pertinente**, mais aucune
comparaison de modèles disponible ne démontre actuellement la supériorité
prédictive de Source2 sur Source1 ou sur leur mélange.

Une extension comparerait Source1 seule, Source2 seule et leur mélange sur
des périodes communes, avec les mêmes règles de disponibilité des variables
et de sélection des modèles. Elle demanderait un audit des valeurs manquantes
et des doublons pour chaque source, ainsi qu'un contrôle des événements
potentiellement communs aux deux sources. Les performances seraient rapportées
globalement et séparément par source, avec leur prévalence positive pour
interpréter l'AP. Source3, également présente dans le fichier, resterait hors
de cette comparaison limitée à Source1 et Source2.

Cette extension n'a pas été exécutée. Les résultats de 2021 et 2022 déjà
consultés ne pourraient pas être présentés comme des tests inédits si leurs
scores guidaient de nouveaux choix expérimentaux.

## Diagnostic des catégories météo

- [x] L'[analyse de dérive des catégories météo](docs/DICTIONNAIRE_ET_FUITES.md#dérive-des-catégories-météo) est exécutée dans le notebook principal : catégories conservées, connues regroupées, inédites et manquantes, effectifs et taux de cible par année, performances du modèle figé par sous-groupe.
- [ ] Évaluer en analyse complémentaire le plafond actuel de 15 catégories et le seuil de 100 observations : comparer des plafonds de 10, 15 et 25 ou un seuil de fréquence, avec vocabulaire appris sur 2016-2019 et sélection sur 2020 uniquement.

Les catégories `Weather_Condition` inédites représentent 6 événements en 2020,
12 en 2021 et 7 en 2022 dans les échantillons, trop peu pour calculer les métriques
selon le seuil fixé de 100 événements. La comparaison des plafonds reste à faire.
Le diagnostic ne démontre pas une perte de signal causée par `Other` et respecte
le modèle historique figé ainsi que le statut des tests déjà consultés.

## Parcours de lecture

1. [Cadrage](docs/CADRAGE.md) et [définition de la cible](docs/DOCUMENTATION_CIBLE.md).
2. [Variables et disponibilité de l'information](docs/DICTIONNAIRE_ET_FUITES.md).
3. [État de l'art](docs/ETAT_DE_L_ART.md) et [protocole](docs/PROTOCOLE.md).
4. [Résultats détaillés](docs/RAPPORT.md) et [validation du notebook principal](docs/MAIN_NOTEBOOK_VALIDATION.md).
5. [Reproduction et oral](docs/MODE_EMPLOI.md), [consignes du cours](docs/CONSIGNES_PROJET.md).

La [comparaison des deux dossiers](docs/COMPARAISON_DOCUMENTATION.md) explique
les choix de consolidation et les différences conservées.

## Structure du dépôt

| Dossier | Contenu |
| --- | --- |
| `traffic_impact/` | Paquet Python : préparation, modèles, contrôles, génération du rapport ; chemins centralisés dans `paths.py` |
| `scripts/` | Points d'entrée `run_study.py`, `validate_artifacts.py` et `validate_main_notebook.py` |
| `tests/` | Tests unitaires |
| `notebooks/` | [Notebook principal exécuté en anglais](notebooks/MAIN_NoteBook.ipynb) et [notebook historique](notebooks/traffic_impact_study.ipynb) |
| `docs/` | Documentation du projet et cellules proposées pour le notebook |
| `results/` | Petits tableaux versionnés ; `audit/` contient l'audit du CSV |
| `figures/` | Figures du rapport |
| `data/` | CSV, échantillons et sorties du notebook dans `main_notebook/`, ignorés par Git, sauf [data/README.md](data/README.md) |
| `artifacts/` | Prédictions individuelles et modèles sauvegardés, ignorés par Git |
| `dist/` | Archive de rendu générée, ignorée par Git |

## Installer et reproduire

Depuis la racine du dépôt, dans PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -t .
```

Après téléchargement du CSV dans `data/` :

```powershell
.\.venv\Scripts\python.exe scripts/run_study.py
.\.venv\Scripts\python.exe scripts/validate_artifacts.py
```

Une copie neuve du dépôt n'inclut ni les caches ni les modèles : le pipeline
doit donc préparer les données et entraîner les modèles, même sans `--rerun`.
Les résultats sont réutilisés seulement si l'audit correspond au CSV, si les
quatre caches sont présents et si les contrôles requis existent. Voir les
[commandes détaillées](docs/MODE_EMPLOI.md), dont la génération du rapport
à partir des seuls tableaux déjà versionnés.

`--rerun` force le réentraînement selon le protocole enregistré. Après avoir
consulté 2021 ou 2022, ne pas adapter les paramètres puis présenter ces années
comme des tests inédits.

Pour exécuter le notebook principal, installer ses dépendances supplémentaires
et utiliser son validateur depuis la racine du dépôt :

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-main-notebook.txt
.\.venv\Scripts\python.exe scripts/validate_main_notebook.py --execute
```

Le validateur utilise un nouveau noyau Jupyter et enregistre les sorties dans
`notebooks/MAIN_NoteBook.ipynb` après une exécution réussie. Les nouveaux rapports
et figures se trouvent dans `data/main_notebook/`. Voir le
[mode d'emploi](docs/MODE_EMPLOI.md#exécuter-le-notebook-principal).
Les [cellules proposées](docs/CELLULES_NOTEBOOK.md) concernent le notebook historique.
Ajouter les noms des membres et vérifier le rendu avant remise. Le CSV, les
caches et les dépendances locales ne doivent pas être joints au rendu.

## Travailler à plusieurs

Une branche par personne, fusion dans `main` par pull request relue par un
autre membre. Une seule personne modifie le notebook à la fois. Aucun fichier
de données ni fichier volumineux dans Git.

## Références

- [Définition de Severity, fiche du créateur](https://www.kaggle.com/sobhanmoosavi/us-accidents/metadata).
- [Moosavi et al., A Countrywide Traffic Accident Dataset](https://arxiv.org/abs/1906.05409).
- [Moosavi et al., Accident Risk Prediction based on Heterogeneous Sparse Data](https://arxiv.org/abs/1909.09638), tâche de risque différente de notre cible.
