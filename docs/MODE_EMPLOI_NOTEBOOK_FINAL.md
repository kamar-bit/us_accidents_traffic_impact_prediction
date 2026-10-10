# Utiliser le notebook final

Le fichier à ouvrir est [FINAL_NOTEBOOK.ipynb](../notebooks/FINAL_NOTEBOOK.ipynb).
Il réunit le code et le rapport en anglais. Membres : Yanis Barbieux, Kamar
Benbacha et Théophile Melquiot. Les contributions présentées sont celles de
Yanis et Kamar identifiées dans les travaux disponibles.

## Lire et reproduire

Le notebook contient les sorties de son exécution. Pour le réexécuter :

1. Garder les dossiers `notebooks/`, `traffic_impact/` et
   `results/final_notebook_comparison/` à leur place.
2. Installer les dépendances depuis la racine du projet :

   ```powershell
   python -m pip install -r requirements-final-notebook.txt
   ```

3. Placer `US_Accidents_March23.csv` dans `data/`, ou renseigner `CSV_PATH`
   dans la première cellule de code.
4. Ouvrir le notebook avec cet environnement Python et exécuter toutes les
   cellules dans l'ordre, après redémarrage du noyau.

La première préparation lit le CSV complet par blocs. Les échantillons locaux
existants sont réutilisés seulement après contrôle du fichier, du code, de
l'environnement et de leurs empreintes. Les modèles sont réentraînés dans le
notebook ; les petites tables des comparaisons antérieures servent à expliquer
les réglages, pas à remplacer les résultats de l'exécution.

Les rapports, le dictionnaire et les figures générés sont dans
`data/final_notebook/`. La normalisation en nanosecondes est explicite dans
le diagnostic de proximité ; celui-ci ne supprime pas de lignes.

## Vérifier et préparer le rendu

Depuis la racine du projet :

```powershell
python scripts/validate_final_notebook.py --execute
python scripts/package_final_notebook.py
```

L'archive `dist/US_Accidents_Notebook_Final.zip` contient un seul notebook,
les modules Python, les dépendances, les petites tables nécessaires et ce
mode d'emploi. Elle ne contient ni CSV brut, ni cache, ni modèle volumineux.

Le générateur `scripts/build_final_notebook.py` refuse d'écraser un notebook
existant. Pour produire une nouvelle ébauche, lui fournir un autre chemin avec
`--output`. Les notebooks individuels restent séparés du rendu commun.

Le protocole final diffère du notebook historique : 34 variables logiques,
actualisation météo dans l'heure, 500 000 événements et comparaison boosting/forêt.
Les documents historiques du dépôt conservent leurs propres résultats.
