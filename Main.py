# Importation des bibliothèques supplémentaires pour SMOTE
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc

# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ===== Partie 1 : Analyse des fichiers CSV et préparation des données ===== #
# Étape 1 : Charger et fusionner les données
def load_and_prepare_data():
    general_data = pd.read_csv('general_data.csv')
    manager_data = pd.read_csv('manager_survey_data.csv')
    employee_data = pd.read_csv('employee_survey_data.csv')

    # Fusion des fichiers via EmployeeID
    data = pd.merge(general_data, manager_data, on='EmployeeID', how='inner')
    data = pd.merge(data, employee_data, on='EmployeeID', how='inner')

    # Étape 2 : Gérer les valeurs manquantes
    data['NumCompaniesWorked'] = data['NumCompaniesWorked'].fillna(data['NumCompaniesWorked'].median())
    data['TotalWorkingYears'] = data['TotalWorkingYears'].fillna(data['TotalWorkingYears'].median())
    data['EnvironmentSatisfaction'] = data['EnvironmentSatisfaction'].fillna(data['EnvironmentSatisfaction'].median())
    data['JobSatisfaction'] = data['JobSatisfaction'].fillna(data['JobSatisfaction'].median())
    data['WorkLifeBalance'] = data['WorkLifeBalance'].fillna(data['WorkLifeBalance'].median())

    # Étape 3 : Encoder la variable cible et les variables catégoriques
    data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})
    data_encoded = pd.get_dummies(data, drop_first=True)

    # Étape 4 : Séparer les variables explicatives (X) et la cible (y)
    cols_to_drop = ['EmployeeID', 'StandardHours', 'Over18', 'EmployeeCount', 'Attrition']
    cols_to_drop = [col for col in cols_to_drop if col in data_encoded.columns]

    X = data_encoded.drop(columns=cols_to_drop)
    y = data_encoded['Attrition']

    # Normalisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Étape 5 : Appliquer SMOTE pour équilibrer les classes
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    return X_resampled, y_resampled, X


# ===== Partie 2 : Application d'un algorithme ===== #
# Étape spécifique pour chaque membre (Régression logistique pour toi)
def logistic_regression(X_resampled, y_resampled, X):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    # Division des données en entraînement et test
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Entraînement du modèle de régression logistique
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    # Prédictions et évaluation
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    print("Classification Report :")
    print(classification_report(y_test, y_pred))

    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))

    print(f"AUC-ROC : {roc_auc_score(y_test, y_pred_prob):.2f}")

    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"AUC-ROC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Courbe ROC après application de SMOTE')
    plt.legend(loc="lower right")
    plt.show()

    # Importance des variables
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_[0]
    }).sort_values(by='Coefficient', ascending=False)

    print("\nImportance des variables :")
    print(coefficients.head(10))


# ===== Exécution ===== #
if __name__ == "__main__":
    # Partie commune
    X_resampled, y_resampled, X_original = load_and_prepare_data()

    # Partie spécifique (Régression logistique pour toi)
    logistic_regression(X_resampled, y_resampled, X_original)
