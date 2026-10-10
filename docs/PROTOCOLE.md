# Protocole et fichiers de résultats

## Question et statut du cadrage

Identifier au début d'un accident les incidents aux niveaux supérieurs
d'impact sur le trafic, puis tester la stabilité temporelle du modèle.
Cible : `Severity >= 3`, soit 3-4 contre 1-2. Voir la
[définition et sa justification](DOCUMENTATION_CIBLE.md).
Le modèle principal utilise les 33 variables admissibles détaillées dans le
[dictionnaire](DICTIONNAIRE_ET_FUITES.md).

Les choix expérimentaux sont enregistrés dans `results/frozen_selection.json`.
Le recentrage du récit sur le modèle complet intervient après les expériences :
ce document ne constitue pas un nouveau protocole fixé avant leurs résultats.
Il ne change ni les scores ni les choix figés.

## Données et découpage

Source2 uniquement, graine 42, échantillonnage uniforme d'événements distincts
par priorités de hachage, sans équilibrage des classes de validation ou de test.

| Ensemble | Période | Plafond demandé | Événements après nettoyage |
| --- | --- | ---: | ---: |
| Apprentissage | 2016-2019 | 200 000 | 199 999 |
| Validation | 2020 | 50 000 | 50 000 |
| Test | 2021 | 100 000 | 99 999 |
| Stabilité supplémentaire | 2022, jusqu'au 8 septembre | 50 000 | 50 000 |

Les répétitions d'un même événement sont regroupées ; les clés contradictoires
sont exclues. Les catégories, imputations, seuils et suppressions de constantes
sont appris uniquement sur le train. La météo hors de
`[Start_Time - 120 minutes, Start_Time]` est masquée sans retirer l'accident.
Ce contrôle ne garantit pas l'heure réelle de publication de l'observation.

Les hyperparamètres et concurrents sont choisis sur la validation 2020, puis
figés avant les scores de 2021 et 2022. Ces années ont désormais été évaluées :
après des adaptations fondées sur leurs scores, elles ne seraient plus des
tests inédits. Le nombre de variables compte les entrées logiques, pas les
colonnes après encodage catégoriel.

## Comparaison centrale et diagnostics

1. Probabilité constante, référence sans information.
2. Logistique complète, `C` parmi 0,1, 1 et 10.
3. `HistGradientBoostingClassifier` complet : 7, 15 ou 31 feuilles, 100 ou
   200 itérations, taux d'apprentissage 0,1. Choix sur l'AP de 2020 :
   31 feuilles et 200 itérations. Aucun poids de classe ni arrêt anticipé aléatoire.

Les 33 règles univariées, 528 paires d'arbres limités et modèles à peu de
variables sont des références de non-trivialité et des sensibilités.
Les trois paires géographiques supplémentaires ont été motivées par les
performances géographiques sur validation et restent une analyse complémentaire.
Elles ne remplacent pas la comparaison principale. La courbe de budget suit
un ordre choisi sur validation, sans recherche exhaustive des sous-ensembles.

## Évaluation et interprétation

- Average Precision, avec la prévalence positive comme référence.
- Rappel et précision parmi les 10 % de scores les plus élevés.
- ROC-AUC et résultats mensuels et par État.
- Intervalles des différences par bootstrap apparié de semaines, 500 tirages.
- Ablations, courbe d'apprentissage, indicateurs de manquants, mélange des
  étiquettes, doublons et disponibilité météo. L'audit distance/durée est séparé.

AP et AUC ne sont pas des pourcentages de prédictions correctes. Le top 10 %
est un budget de classement rétrospectif. Les probabilités ne sont pas validées
par calibration. Le seuil de cible `Severity >= 3` est distinct d'un seuil
de probabilité d'alerte, qui devrait être choisi sur validation.

Le critère historique de cinq points supplémentaires de rappel, un intervalle
excluant zéro et aucune baisse d'AP jugeait le gain de complexité. Il reste
documenté comme analyse secondaire, sans garantie économique ni nouveau
critère de réussite de la question prédictive.

## Fichiers, chemins relatifs à la racine du dépôt

| Emplacement | Contenu |
| --- | --- |
| `results/audit/` | `audit.json`, `columns.csv`, `monthly_source_severity.csv`, `monthly_missing_source2.csv`, `duplicates_by_month.csv` |
| `results/frozen_selection.json` | Choix issus exclusivement de 2020 |
| `results/validation_models.csv`, `single_variables.csv`, `pair_rules.csv` | Sélection et références simples, dans `results/` |
| `results/heldout_models.csv` | Variantes figées évaluées sur 2021 et 2022 |
| `results/paired_weekly_intervals.csv` | Intervalles par semaines appariées, 500 tirages |
| `results/geographic_pairs_*.csv` | Trois paires géographiques supplémentaires |
| `results/sensitivity_intervals.csv` | Sensibilité au concurrent, à la météo, à la lumière et au budget |
| `results/shuffled_label_replicates.csv` | Contrôles par mélange des étiquettes d'apprentissage |
| `results/near_duplicate_sensitivity.csv` | Collisions dans des cellules de 15 minutes et 0,001 degré |
| `data/cache/`, ignoré | Échantillons nettoyés, régénérables |
| `artifacts/`, ignoré | Prédictions et `frozen_models.joblib`, régénérables |

## Limites

Un seul fournisseur et des accidents enregistrés, sans garantie de disponibilité
au premier signalement ni connaissance du moment d'attribution de Severity.
Les positifs sont surtout de niveau 3 ; les scores binaires ne valident pas
à eux seuls la détection du niveau 4. Les ablations décrivent des associations
prédictives, pas des effets causaux. Calibration, nouveaux tests temporels et
données externes sont des suites possibles, pas des résultats déjà obtenus.
