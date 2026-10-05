# Impact d'un accident sur le trafic (US Accidents)

Projet du module *Hands-on Machine Learning* (Master 1 Informatique - IA).
Membres du groupe : **[AJOUTER LES TROIS NOMS]**

**Question.** Au debut d'un accident, peut-on identifier ceux dont l'impact sur la circulation est eleve
(`Severity` >= 3), a partir du contexte (lieu, heure, route, lumiere, meteo), et la performance se
maintient-elle l'annee suivante ? `Severity` mesure l'impact sur le trafic, pas la gravite des blessures.

**Donnees.** [US Accidents (Kaggle)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents), source "Source2"
uniquement. Entrainement 2016-2019, validation 2020, test 2021, "stress" 2022.

## Resultat principal (test 2021, 99 999 accidents)

| Modele | AP | AUC | Precision dans le top 10 % |
| --- | --- | --- | --- |
| Sans information (prevalence 34,3 %) | 0,343 | 0,500 | 33,5 % |
| Regression logistique, 33 variables | 0,567 | 0,734 | 66,4 % |
| Gradient boosting, 33 variables | 0,703 | 0,836 | 80,1 % |

Le modele apprend surtout **ou** se produit l'accident : la geographie apporte l'essentiel du gain, la meteo
et la lumiere n'apportent rien. Details et limites : [docs/RAPPORT.md](docs/RAPPORT.md).

## Structure du depot

```
.
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- traffic_impact/          code reutilisable (paquet Python)
|   |-- paths.py             tous les chemins, definis une seule fois
|   |-- prepare.py           audit du CSV complet et echantillonnage
|   |-- experiment.py        variables, modeles, metriques, protocole fige
|   |-- verify_results.py    controles supplementaires (etiquettes melangees, sensibilites)
|   |-- check_geographic_pairs.py   paires geographiques (sensibilite)
|   `-- report.py            genere le rapport, les figures et les cellules du notebook
|-- scripts/                 points d'entree en ligne de commande
|   |-- run_study.py         pipeline complet (audit, etude, controles, rapport)
|   `-- validate_artifacts.py   coherence du depot + archive de rendu sans donnees
|-- tests/                   tests unitaires
|-- notebooks/               notebook de rendu (anglais)
|-- docs/                    rapport, definition de la cible, protocole, cellules du notebook
|-- results/                 tableaux de resultats (petits, versionnes) ; audit/ = audit du CSV
|-- figures/                 figures du rapport
|-- data/                    CSV et echantillons (ignores par Git, voir data/README.md)
`-- artifacts/               predictions individuelles et modeles sauvegardes (ignores par Git)
```

## Installer et reproduire

```bash
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt

python -m unittest discover -s tests -t .              # tests unitaires

# placer le CSV dans data/ (voir data/README.md), puis :
python scripts/run_study.py                            # audit + rapport (reutilise les resultats existants)
python scripts/run_study.py --rerun                    # reentraine les modeles figes et recalcule le test
python scripts/validate_artifacts.py                   # coherence + archive dist/rendu_sans_donnees.zip
```

`--rerun` ne doit pas etre utilise pour "ameliorer" les scores : les choix ont ete figes sur 2020 avant
de lire 2021 et 2022 (voir [docs/PROTOCOLE.md](docs/PROTOCOLE.md)).

## Ce qui est dans Git, et ce qui n'y est pas

Versionne : code, tests, docs, notebook, petits tableaux de `results/`, figures.
Ignore : le CSV, `data/cache/`, `artifacts/`, `dist/`. Ils se regenerent avec `scripts/run_study.py`.

## Travailler a plusieurs

Une branche par personne, fusion dans `main` par pull request relue par un autre membre. Un seul notebook,
modifie par une seule personne. Aucun fichier de donnees ni de plus de quelques Mo dans Git.

## References

- Definition de `Severity` : [fiche du createur du dataset](https://www.kaggle.com/sobhanmoosavi/us-accidents/metadata)
- Moosavi et al., *A Countrywide Traffic Accident Dataset* ([arXiv:1906.05409](https://arxiv.org/abs/1906.05409))
- Moosavi et al., *Accident Risk Prediction based on Heterogeneous Sparse Data* ([arXiv:1909.09638](https://arxiv.org/abs/1909.09638))
