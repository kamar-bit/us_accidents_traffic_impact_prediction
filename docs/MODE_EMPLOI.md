# Mode d'emploi et présentation

## Lire le dossier

Commencer par le [notebook principal exécuté](../notebooks/MAIN_NoteBook.ipynb),
puis le [cadrage](CADRAGE.md), la [définition de la cible](DOCUMENTATION_CIBLE.md),
le [dictionnaire](DICTIONNAIRE_ET_FUITES.md) et le [protocole](PROTOCOLE.md).
Le [rapport](RAPPORT.md) présente les résultats réels ; l'[état de l'art](ETAT_DE_L_ART.md)
explicite la portée des références consultées. Les [consignes](CONSIGNES_PROJET.md)
reprennent le rappel fourni par le groupe.

Dans ce dépôt, les sorties historiques sont `results/` et `figures/`. Le notebook
principal écrit ses rapports et figures dans `data/main_notebook/`. L'ancien dossier
local `etude_simplicite/` ne doit pas être recréé : ses fichiers ont ici des
emplacements adaptés au paquet `traffic_impact/`.

## Installer, depuis la racine du dépôt

PowerShell, sans activation nécessaire de l'environnement :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -t .
```

Sur macOS ou Linux, utiliser `.venv/bin/python` à la place de
`.\.venv\Scripts\python.exe`.

## Consulter ou régénérer les résultats publiés

Le rapport, le notebook et les figures sont déjà dans Git et peuvent être lus
sans télécharger le CSV ni exécuter l'expérience. Avec les dépendances installées,
les tableaux versionnés permettent de régénérer le rapport et les figures :

```powershell
.\.venv\Scripts\python.exe -m traffic_impact.report
```

Le générateur conserve un notebook existant par défaut. Il réécrit le rapport,
les cellules proposées et les figures à partir de ses textes et des résultats
enregistrés. Relire les documents après génération : les corrections éditoriales
faites directement dans ces fichiers peuvent être remplacées par le générateur.
Les autres documents de cadrage et de méthode sont maintenus manuellement.

## Recalculer depuis le CSV

Télécharger le CSV indiqué dans [data/README.md](../data/README.md) et le placer
dans `data/US_Accidents_March23.csv`.

```powershell
.\.venv\Scripts\python.exe scripts/run_study.py
.\.venv\Scripts\python.exe scripts/validate_artifacts.py
```

On peut aussi pointer vers le CSV déjà présent dans le dossier local parent,
sans le copier dans le clone :

```powershell
.\.venv\Scripts\python.exe scripts/run_study.py --csv ../../US_Accidents_March23.csv
```

Le CSV local fait 3 058 183 727 octets et est lu par blocs. Une copie neuve du
dépôt n'inclut pas `data/cache/` ni `artifacts/` : le pipeline prépare donc
les échantillons et entraîne les modèles même sans `--rerun`. Il réutilise les
résultats seulement si l'audit correspond au chemin, à la taille et à la date
de modification du CSV, si les quatre caches sont présents et si les contrôles
requis existent. `--rerun` force le réentraînement.

Les dépendances locales de l'ancien dossier ne sont pas transférées. Les anciens
modèles sérialisés utilisent des noms de modules différents : ne pas les copier
directement dans `artifacts/`. La reproduction doit utiliser le paquet du dépôt.

## Exécuter le notebook principal

Depuis la racine du dépôt, après création de l'environnement et mise à disposition
du CSV dans `data/US_Accidents_March23.csv` ou `../../US_Accidents_March23.csv` :

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-main-notebook.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
.\.venv\Scripts\python.exe scripts/validate_main_notebook.py --execute
```

Le validateur ouvre un nouveau noyau Jupyter avec le Python utilisé pour cette
commande, exécute toutes les cellules et sauvegarde les sorties dans
`notebooks/MAIN_NoteBook.ipynb` après réussite. Si le notebook est modifié pendant
l'exécution, le validateur refuse d'écraser ces modifications.
Pour vérifier sa structure et ses sorties enregistrées sans réexécution,
utiliser `scripts/validate_main_notebook.py` sans `--execute`.

Le notebook audite le CSV et construit ses propres échantillons sous
`data/main_notebook/data/cache/`. Il réutilise un cache seulement si le CSV,
le code de préparation, les paramètres, les versions de Python/pandas/NumPy et
les empreintes des fichiers correspondent. Les caches anciens, corrompus ou
issus d'un autre environnement sont reconstruits automatiquement.
Les tableaux sont exportés dans `data/main_notebook/reports/` et les figures dans
`data/main_notebook/figures/`. Ces fichiers sont ignorés par Git.

La validation automatisée du 6 octobre 2026 a terminé les 16 cellules
sans erreur, avec 21 tests unitaires et 27 contrôles de cohérence réussis.
Elle a duré 244,26 secondes, dont 4,92 secondes d'entraînement, reconstruction
des caches comprise. La relance manuelle suivante conserve les mêmes scores et
les 27 contrôles réussis, avec 5,53 secondes d'entraînement.
Versions vérifiées : Python 3.10.11, NumPy 2.2.6,
pandas 2.3.3, scikit-learn 1.7.2, matplotlib 3.10.8.
Voir le [bilan de validation](MAIN_NOTEBOOK_VALIDATION.md) et les
[résultats reproduits](RAPPORT.md#reproduction-dans-le-notebook-principal-6-octobre-2026).

## Notebook et remise

Le [notebook principal](../notebooks/MAIN_NoteBook.ipynb) est le point d'entrée
de la reproduction vérifiée du modèle complet et de la référence constante.
Le [notebook historique](../notebooks/traffic_impact_study.ipynb) et ses
[cellules proposées](CELLULES_NOTEBOOK.md) restent disponibles.

`scripts/validate_artifacts.py` vérifie les artefacts et crée
`dist/rendu_sans_donnees.zip`. Il ne réentraîne pas. La remise doit inclure un
notebook en anglais sous 10 Mo, les modules Python utilisés et les noms de tous
les membres. L'archive exclut le CSV, les caches, les prédictions individuelles,
les modèles sauvegardés et les dépendances téléchargées. Vérifier le rendu et
la liste des fichiers avant remise ; cette consolidation ne génère pas l'archive.

## Oral de 10 minutes

| Partie | Durée | Contenu |
| --- | --- | --- |
| Question et cible | 2 min | Niveaux 3-4, source officielle, regroupement choisi |
| Données | 2 min | Source2, découpage temporel, 33 variables, météo et exclusions |
| Modèles | 2 min | Constante, logistique complète, boosting complet |
| Résultats | 3 min | AP, top 10 %, stabilité, ablations et limites |
| Conclusion | 1 min | Signal prédictif, disponibilité et généralisation |

Les règles et budgets de variables sont secondaires. Ne pas promettre des
minutes de bouchon ou une détection exhaustive du niveau 4.
