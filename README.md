# US Accidents : prédire l'impact sur le trafic au début d'un accident

Projet du module *Hands-on Machine Learning* (Master 1 Informatique - IA).
Membres du groupe : **[AJOUTER LES TROIS NOMS]**

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

## Parcours de lecture

1. [Cadrage](docs/CADRAGE.md) et [définition de la cible](docs/DOCUMENTATION_CIBLE.md).
2. [Variables et disponibilité de l'information](docs/DICTIONNAIRE_ET_FUITES.md).
3. [État de l'art](docs/ETAT_DE_L_ART.md) et [protocole](docs/PROTOCOLE.md).
4. [Résultats détaillés](docs/RAPPORT.md).
5. [Reproduction et oral](docs/MODE_EMPLOI.md), [consignes du cours](docs/CONSIGNES_PROJET.md).

La [comparaison des deux dossiers](docs/COMPARAISON_DOCUMENTATION.md) explique
les choix de consolidation et les différences conservées.

## Structure du dépôt

| Dossier | Contenu |
| --- | --- |
| `traffic_impact/` | Paquet Python : préparation, modèles, contrôles, génération du rapport ; chemins centralisés dans `paths.py` |
| `scripts/` | Points d'entrée `run_study.py` et `validate_artifacts.py` |
| `tests/` | Tests unitaires |
| `notebooks/` | [Notebook en anglais](notebooks/traffic_impact_study.ipynb), conservé sans modification directe |
| `docs/` | Documentation du projet et cellules proposées pour le notebook |
| `results/` | Petits tableaux versionnés ; `audit/` contient l'audit du CSV |
| `figures/` | Figures du rapport |
| `data/` | CSV et échantillons, ignorés par Git, sauf [data/README.md](data/README.md) |
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

Le notebook GitHub reprend déjà le cadrage du modèle complet. Les
[cellules proposées](docs/CELLULES_NOTEBOOK.md) servent aux mises à jour manuelles.
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
