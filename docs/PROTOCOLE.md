# Protocole et fichiers de resultats

## Question

Peut-on identifier, au debut d'un accident, ceux dont l'impact sur le trafic est eleve (`Severity` >= 3), avec
un modele complet, puis conserver la performance l'annee suivante ? Definition de la cible : `DOCUMENTATION_CIBLE.md`.

## Protocole

- Source2 uniquement. Entrainement 2016-2019 (jusqu'a 200 000 evenements), validation 2020 (50 000),
  test 2021 (100 000), stress 2022 (50 000, la source s'arrete au 8 septembre 2022).
- Les choix (variables, hyperparametres, concurrents simples) sont faits sur 2020 seulement et enregistres
  dans `results/frozen_selection.json` **avant** le calcul des scores de 2021 et 2022.
- Ne pas adapter les parametres apres avoir lu 2021 ou 2022 puis presenter ces annees comme des tests inedits.
- Meteo masquee si l'observation n'est pas dans les deux heures precedant `Start_Time`.
- Les transformations (categories, imputations, seuils) apprennent uniquement sur l'entrainement.
- Le nombre de variables compte les entrees logiques (33), pas les indicatrices des variables categorielles.

## Fichiers

| Emplacement | Contenu |
| --- | --- |
| `results/audit/` | audit du CSV complet : `audit.json`, `columns.csv`, `monthly_source_severity.csv`, `monthly_missing_source2.csv`, `duplicates_by_month.csv` |
| `results/frozen_selection.json` | choix issus exclusivement de 2020 |
| `results/validation_models.csv`, `single_variables.csv`, `pair_rules.csv` | selection et references simples |
| `results/heldout_models.csv` | toutes les variantes figees evaluees sur 2021 et 2022 |
| `results/paired_weekly_intervals.csv` | intervalles par tirage de semaines (500 tirages), apparies entre modeles |
| `results/geographic_pairs_*.csv` | trois paires geographiques supplementaires (sensibilite, selection sur 2020) |
| `results/sensitivity_intervals.csv` | sensibilite au concurrent, a la meteo, a la lumiere et au budget de variables |
| `results/shuffled_label_replicates.csv` | controles par melange des etiquettes d'apprentissage |
| `results/near_duplicate_sensitivity.csv` | collisions dans des cellules de 15 minutes et 0,001 degre |
| `data/cache/` (ignore) | echantillons nettoyes, regenerables |
| `artifacts/` (ignore) | `predictions_test.csv`, `predictions_stress.csv`, `frozen_models.joblib`, regenerables |

## Limites a garder en tete

Les trois paires geographiques ont ete motivees par la bonne performance du groupe geographique en validation :
ce sont des sensibilites, pas une modification de la comparaison principale. La courbe de budget de variables suit un
ordre choisi sur validation, sans recherche exhaustive. La meteo autorisee est datee au plus deux heures avant le debut ;
cela ne garantit pas son heure reelle de publication. Les probabilites n'ont pas fait l'objet d'une calibration.
