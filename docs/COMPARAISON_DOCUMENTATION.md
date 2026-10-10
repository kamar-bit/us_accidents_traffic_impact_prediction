# Comparaison et choix de consolidation

Comparaison réalisée le 6 octobre 2026 entre le dossier local
`analyse_us_accidents/` et le dépôt
[kamar-bit/us_accidents_traffic_impact_prediction](https://github.com/kamar-bit/us_accidents_traffic_impact_prediction),
cloné dans `github/us_accidents_traffic_impact_prediction/`.
Base GitHub : commit `5419c0d` (`doc1`), branche `main`.
Branche de consolidation : `Yanis-1`.

## Compatibilité

Les deux dossiers décrivent la même étude : cible `Severity >= 3`, Source2,
33 variables, sélection sur 2020 et mêmes résultats principaux de 2021 et 2022.
Il n'y a pas de divergence scientifique majeure justifiant de suspendre la
consolidation. La comparaison porte sur les fichiers et artefacts existants,
sans nouvelle expérience ni nouvelle vérification des sources externes.

## Choix par document

| Élément | Comparaison | Décision |
| --- | --- | --- |
| README | GitHub décrit mieux la structure ; le local précise mieux la cible et les limites | Réunir ces apports et corriger la conclusion trop générale sur météo/lumière |
| Rapport | Contenu identique après normalisation des fins de ligne | Conserver les résultats ; corriger la note finale sur l'état du notebook GitHub |
| Définition de la cible | Contenu identique après normalisation des fins de ligne | Conserver le fichier GitHub |
| Protocole | GitHub fournit les chemins ; le local détaille davantage sélection, évaluation et limites | Compléter `PROTOCOLE.md` avec les explications locales |
| Cadrage et dictionnaire | Présents seulement dans le dossier local | Importer avec les chemins du dépôt |
| État de l'art | Présent seulement dans le dossier local | Importer en conservant les réserves sur les lectures et les scores non vérifiés |
| Mode d'emploi | Le local utilise l'ancienne structure et un état antérieur du notebook | Adapter les commandes, caches, chemins et préparation de l'oral |
| Consignes du cours | Le local contient un passage d'un projet League of Legends | Conserver les consignes générales et remplacer ce passage par US Accidents |
| Cellules proposées | Même contenu méthodologique, chemins adaptés au paquet GitHub | Garder les chemins GitHub ; corriger les noms des modules dans le texte |
| Notebooks | Le local garde le récit de simplicité ; GitHub est déjà centré sur le modèle complet | Conserver les deux fichiers sans les modifier |

## Contrôle des résultats

Sur les 26 fichiers de résultats communs, 23 sont strictement identiques par
empreinte SHA-256, dont `frozen_selection.json`, `heldout_models.csv` et les
intervalles appariés. Les trois autres différences sont :

- `future_information_4class.csv` : fins de ligne uniquement.
- `shuffled_label_replicates.csv` : deux valeurs de Brier diffèrent d'environ
  3 × 10^-17, sans changement des AP, AUC, rappels ou précisions.
- `artifact_validation.json` : chemins et empreintes du code réorganisé,
  taille du notebook et état de la relecture des prédictions sauvegardées.

Ces différences n'affectent pas les conclusions documentées. Aucun fichier de
résultats, modèle, donnée ou code Python n'est remplacé par cette consolidation.
Les anciennes sorties locales hors de `etude_simplicite/` ne servent pas de
référence pour les scores de l'étude actuelle.

## Points restant à finaliser pour la remise

Ajouter les noms des membres, vérifier le rendu du notebook et préparer les
slides si souhaité. Le générateur peut réécrire les textes du rapport et des
cellules proposées : relire ces fichiers après une régénération, comme expliqué
dans le [mode d'emploi](MODE_EMPLOI.md). Cette branche rassemble la documentation ;
elle ne modifie pas les choix expérimentaux ni les notebooks.
