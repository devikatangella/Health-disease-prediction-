"""
Model Evaluation and Diagnostic Visualization Module
Computes:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion Matrix with clinical annotation (True Negatives, False Positives, False Negatives, True Positives)
- ROC Curves
- Precision-Recall Curves
- Multi-Model Comparison Plots
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report
)

# Professional medical styling
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15
})

def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes standard healthcare ML evaluation metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.5
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    # Specificity = TN / (TN + FP)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": cm.tolist()
    }

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ["Negative (No Disease)", "Positive (Disease)"],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plots annotated confusion matrix with clinical emphasis."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Custom annotations
    group_names = ["True Negative (TN)", "False Positive (FP)\n[Type I Error]", "False Negative (FN)\n[Type II Error]", "True Positive (TP)"]
    group_counts = [f"{v:0.0f}" for v in cm.flatten()]
    group_percentages = [f"{v:.1%}" for v in cm.flatten() / np.sum(cm)]
    
    labels = [f"{v1}\n{v2}\n({v3})" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax,
                xticklabels=class_names, yticklabels=class_names, linewidths=1.5, linecolor="#e0e0e0")
    
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontweight="semibold")
    ax.set_ylabel("True Ground Truth", fontweight="semibold")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
    return fig

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Classifier",
    title: str = "Receiver Operating Characteristic (ROC) Curve",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plots ROC Curve with AUC score."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(fpr, tpr, color="#1e3a8a", lw=2.5, label=f"{model_name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Chance Baseline (AUC = 0.50)")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="semibold")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontweight="semibold")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
    return fig

def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Classifier",
    title: str = "Precision-Recall Curve",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plots Precision-Recall Curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(recall, precision, color="#0284c7", lw=2.5, label=f"{model_name}")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontweight="semibold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontweight="semibold")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
    return fig

def plot_model_comparison(
    models_metrics: Dict[str, Dict[str, float]],
    metric_keys: List[str] = ["accuracy", "recall", "f1", "roc_auc"],
    title: str = "Model Performance Benchmark Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plots grouped bar chart comparing multiple trained algorithms."""
    data = []
    for model_name, metrics in models_metrics.items():
        for m in metric_keys:
            if m in metrics:
                data.append({
                    "Model": model_name,
                    "Metric": m.replace("_", " ").upper(),
                    "Score": metrics[m]
                })
                
    df_plot = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    palette = ["#1e3a8a", "#0284c7", "#0d9488", "#f59e0b"]
    sns.barplot(data=df_plot, x="Metric", y="Score", hue="Model", palette=palette, ax=ax)
    
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title, fontweight="bold", pad=14)
    ax.set_ylabel("Score (0.0 to 1.0)", fontweight="semibold")
    ax.set_xlabel("Evaluation Metric", fontweight="semibold")
    ax.legend(title="Algorithm", frameon=True, bbox_to_anchor=(1.02, 1), loc="upper left")
    
    # Annotate bars with values
    for p in ax.patches:
        h = p.get_height()
        if h > 0.05:
            ax.annotate(f"{h:.2f}",
                        (p.get_x() + p.get_width() / 2., h / 2),
                        ha='center', va='center',
                        color='white', fontweight='bold', fontsize=9, rotation=90)
                        
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
    return fig
