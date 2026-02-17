# Modelisation-de-Risques-de-Sante-
🏥 Prédiction des Risques de Réadmission Hospitalière (Machine Learning)

Ce projet implémente un pipeline complet de Data Science pour prédire le risque de réadmission des patients sous 30 jours, en s'appuyant sur des indicateurs cliniques et démographiques inspirés des standards de données de santé (type MIMIC-III).

🎯 Objectifs du Projet

Modélisation prédictive : Identifier les patients à haut risque via un classifieur Random Forest.

Interprétabilité clinique : Analyser les facteurs déterminants (Feature Importance) pour aider à la décision médicale.

Rigueur logicielle : Garantir la stabilité du code par des tests unitaires intégrés.

🛠️ Stack Technique

Langage : Python 3.x

Librairies Data : Pandas, NumPy, Scikit-learn

Tests : Unittest

Méthodologie : Programmation orientée objet (POO) pour le pipeline de modélisation.

📂 Structure du Code

generate_health_data() : Simulateur de données synthétiques (Âge, BMI, comorbidités, durée de séjour).

HospitalReadmissionModel : Classe gérant l'entraînement, l'évaluation (ROC AUC) et l'importance des variables.

TestHealthPipeline : Suite de tests vérifiant l'intégrité des données et la performance du modèle.

📊 Résultats & Interprétation

Le modèle permet d'isoler des variables clés telles que le score de comorbidité et le nombre d'admissions antérieures, souvent corrélés avec une fragilité accrue du patient. L'utilisation du score ROC AUC permet d'évaluer la capacité du modèle à distinguer les classes dans un contexte de données potentiellement déséquilibrées.

🚀 Utilisation

python health_risk_modeling.py
