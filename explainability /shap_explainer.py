
# shap_explainer.py

import shap
import matplotlib.pyplot as plt

def explain_predictions(model, X_sample, max_display=5):
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)

    print("Generating SHAP summary plot...")
    shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png")
    print("SHAP plot saved as shap_summary_plot.png")
    return shap_values
