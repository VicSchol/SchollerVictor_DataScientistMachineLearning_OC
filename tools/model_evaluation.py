from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score, accuracy_score

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def evaluate_classification_model(model, X_train, y_train, X_test, y_test, average='weighted', threshold=0.5):
    """
    Évalue un modèle de classification en affichant matrices de confusion et métriques
    avec possibilité d'ajuster le seuil de décision.

    Args:
        model : modèle sklearn entraîné
        X_train, y_train : données d'entraînement
        X_test, y_test : données de test
        average : moyenne pour les métriques ('weighted', 'macro', 'micro', None)
        threshold : seuil de décision (par défaut 0.5)

    Returns:
        dict : dictionnaire avec métriques et prédictions
    """
    results = {}

    # ---------------- Prédictions avec seuil ----------------
    if hasattr(model, "predict_proba"):
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
        y_train_pred = (y_train_proba >= threshold).astype(int)
        y_test_pred = (y_test_proba >= threshold).astype(int)
    else:
        # fallback si le modèle n'a pas predict_proba
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

    results['y_train_pred'] = y_train_pred
    results['y_test_pred'] = y_test_pred

    # ---------------- Matrices de confusion côte à côte ----------------
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_xlabel("Prédit")
    axes[0].set_ylabel("Réel")
    axes[0].set_title(f"Train (Seuil={threshold})")
    
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_xlabel("Prédit")
    axes[1].set_ylabel("Réel")
    axes[1].set_title(f"Test (Seuil={threshold})")
    
    plt.tight_layout()
    plt.show()

    results['conf_matrix_train'] = cm_train
    results['conf_matrix_test'] = cm_test

    # ---------------- Précision, rappel, F1-score ----------------
    results['precision_train'] = precision_score(y_train, y_train_pred, average=average, zero_division=0)
    results['recall_train'] = recall_score(y_train, y_train_pred, average=average, zero_division=0)
    results['f1_train'] = f1_score(y_train, y_train_pred, average=average, zero_division=0)

    results['precision_test'] = precision_score(y_test, y_test_pred, average=average, zero_division=0)
    results['recall_test'] = recall_score(y_test, y_test_pred, average=average, zero_division=0)
    results['f1_test'] = f1_score(y_test, y_test_pred, average=average, zero_division=0)

    # ---------------- Affichage synthétique ----------------
    if average is None:
        metrics_df = pd.DataFrame({
            'Precision (Train)': results['precision_train'],
            'Recall (Train)': results['recall_train'],
            'F1 (Train)': results['f1_train'],
            'Precision (Test)': results['precision_test'],
            'Recall (Test)': results['recall_test'],
            'F1 (Test)': results['f1_test']
        }, index=[f'Class {i}' for i in range(len(results['precision_train']))])
    else:
        metrics_df = pd.DataFrame({
            'Precision': [results['precision_train'], results['precision_test']],
            'Recall': [results['recall_train'], results['recall_test']],
            'F1': [results['f1_train'], results['f1_test']]
        }, index=['Train', 'Test'])
    
    print(metrics_df)
    
    return results



def plot_precision_recall_curve(model, X_test, y_test):
    """
    Trace la courbe précision-rappel pour un modèle de classification binaire.
    
    Args:
        model : modèle entraîné (doit avoir predict_proba ou decision_function)
        X_test : features de test
        y_test : labels de test
    """
    # Vérifier si le modèle supporte predict_proba ou decision_function
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]  # probas de la classe positive
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
    else:
        raise ValueError("Le modèle doit avoir predict_proba ou decision_function")

    # Calcul précision et rappel
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    avg_precision = average_precision_score(y_test, y_scores)

    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"PR curve (AP = {avg_precision:.2f})", color="blue")
    plt.xlabel("Recall (Rappel)")
    plt.ylabel("Precision (Précision)")
    plt.title("Courbe Précision–Rappel")
    plt.legend()
    plt.grid(True)
    plt.show()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

def perform_cross_validation(X, y, model, cross_val_type):
    """
    Validation croisée avec métriques sur TRAIN et TEST pour détecter overfit.
    Precision et recall séparés pour chaque classe (0 et 1).
    """
    classes = np.unique(y)

    metrics_per_fold = {
        "train_accuracy": [], "test_accuracy": []
    }
    for c in classes:
        metrics_per_fold[f"train_precision_class_{c}"] = []
        metrics_per_fold[f"train_recall_class_{c}"] = []
        metrics_per_fold[f"test_precision_class_{c}"] = []
        metrics_per_fold[f"test_recall_class_{c}"] = []

    use_iloc = hasattr(X, "iloc") and hasattr(y, "iloc")

    for train_idx, test_idx in cross_val_type.split(X, y):
        if use_iloc:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        else:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)

        # Prédictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Accuracy
        metrics_per_fold["train_accuracy"].append(accuracy_score(y_train, y_train_pred))
        metrics_per_fold["test_accuracy"].append(accuracy_score(y_test, y_test_pred))

        # Precision / recall par classe
        train_prec = precision_score(y_train, y_train_pred, average=None, zero_division=0)
        train_rec = recall_score(y_train, y_train_pred, average=None, zero_division=0)
        test_prec = precision_score(y_test, y_test_pred, average=None, zero_division=0)
        test_rec = recall_score(y_test, y_test_pred, average=None, zero_division=0)

        for i, c in enumerate(classes):
            metrics_per_fold[f"train_precision_class_{c}"].append(train_prec[i])
            metrics_per_fold[f"train_recall_class_{c}"].append(train_rec[i])
            metrics_per_fold[f"test_precision_class_{c}"].append(test_prec[i])
            metrics_per_fold[f"test_recall_class_{c}"].append(test_rec[i])

    # Moyenne et std
    metrics_summary = {}
    for name, values in metrics_per_fold.items():
        metrics_summary[name] = {
            "mean": np.mean(values),
            "std": np.std(values)
        }

    return metrics_summary
