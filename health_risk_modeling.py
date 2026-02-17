import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import unittest

# =================================================================
# 1. GÉNÉRATION DE DONNÉES SYNTHÉTIQUES (STYLE MIMIC-III)
# =================================================================
def generate_health_data(n_patients=1000):
    """Simule des données de santé : âge, comorbidités, durée de séjour."""
    np.random.seed(42)
    data = {
        'patient_id': range(n_patients),
        'age': np.random.randint(18, 95, n_patients),
        'bmi': np.random.normal(27, 5, n_patients),
        'comorbidity_score': np.random.poisson(2, n_patients),
        'stay_duration_days': np.random.exponential(5, n_patients),
        'prior_admissions': np.random.binomial(5, 0.2, n_patients)
    }
    df = pd.DataFrame(data)
    
    # Logique de réadmission (Cible) : Risque accru si âge élevé et beaucoup de comorbidités
    logit = -3 + 0.02 * df['age'] + 0.5 * df['comorbidity_score'] + 0.1 * df['prior_admissions']
    prob = 1 / (1 + np.exp(-logit))
    df['readmitted'] = np.random.binomial(1, prob)
    
    return df

# =================================================================
# 2. PIPELINE DE MODÉLISATION
# =================================================================
class HospitalReadmissionModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.features = ['age', 'bmi', 'comorbidity_score', 'stay_duration_days', 'prior_admissions']

    def train(self, df):
        X = df[self.features]
        y = df['readmitted']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        # Évaluation
        y_pred = self.model.predict(X_test)
        auc = roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1])
        return auc, classification_report(y_test, y_pred)

    def get_feature_importance(self):
        """Interprétation clinique des résultats."""
        return pd.Series(self.model.feature_importances_, index=self.features).sort_values(ascending=False)

# =================================================================
# 3. TESTS UNITAIRES (ROBUSTESSE)
# =================================================================
class TestHealthPipeline(unittest.TestCase):
    def test_data_generation_shape(self):
        df = generate_health_data(100)
        self.assertEqual(len(df), 100)
        self.assertIn('readmitted', df.columns)

    def test_model_training(self):
        df = generate_health_data(500)
        pipeline = HospitalReadmissionModel()
        auc, _ = pipeline.train(df)
        self.assertGreater(auc, 0.5) # Le modèle doit faire mieux que le hasard

# =================================================================
# EXECUTION
# =================================================================
if __name__ == "__main__":
    print("--- Phase 1 : Collecte et Préparation ---")
    raw_df = generate_health_data(2000)
    
    print("--- Phase 2 : Entraînement et Interprétation ---")
    readmission_pipeline = HospitalReadmissionModel()
    auc_score, report = readmission_pipeline.train(raw_df)
    
    print(f"ROC AUC Score : {auc_score:.2f}")
    print("\nImportance des variables cliniques :")
    print(readmission_pipeline.get_feature_importance())
    
    print("\n--- Phase 3 : Validation Technique (Tests) ---")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHealthPipeline)
    unittest.TextTestRunner(verbosity=1).run(suite)