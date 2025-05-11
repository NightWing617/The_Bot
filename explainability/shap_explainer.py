import logging
import shap
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

class ModelExplainer:
    def __init__(self, model: RandomForestClassifier):
        self.model = model
        self.explainer = None
        self.feature_names = None
        
    def _initialize_explainer(self, X: pd.DataFrame) -> bool:
        """Initialize SHAP explainer with background data."""
        try:
            self.feature_names = X.columns.tolist()
            self.explainer = shap.TreeExplainer(self.model)
            return True
        except Exception as e:
            logger.error(f"Error initializing explainer: {str(e)}")
            return False
            
    def generate_explanations(
        self,
        X: pd.DataFrame,
        horse_names: Optional[List[str]] = None
    ) -> Dict[str, dict]:
        """
        Generate SHAP explanations for predictions.
        
        Args:
            X: Feature matrix
            horse_names: Optional list of horse names for labeling
            
        Returns:
            Dictionary of explanations per horse
        """
        try:
            logger.info("Generating SHAP explanations...")
            
            # Initialize explainer if needed
            if not self.explainer:
                success = self._initialize_explainer(X)
                if not success:
                    raise ValueError("Failed to initialize explainer")
                    
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                # For binary classification, take positive class values
                shap_values = shap_values[1]
            
            # Convert to numpy array if needed
            if not isinstance(shap_values, np.ndarray):
                shap_values = np.array(shap_values)
                
            # Prepare feature names
            feature_names = (
                self.feature_names if self.feature_names
                else X.columns.tolist()
            )
            
            # Get expected value (handle both single value and array)
            expected_value = self.explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = expected_value[1]  # For binary classification
            base_value = float(expected_value)
            
            # Generate explanations for each prediction
            explanations = {}
            for i in range(len(X)):
                horse_name = horse_names[i] if horse_names else f"Horse_{i}"
                
                # Get feature importance for this prediction
                importance_dict = {}
                for j, name in enumerate(feature_names):
                    value = shap_values[i, j]
                    # Handle multi-dimensional SHAP values
                    if isinstance(value, np.ndarray):
                        value = float(np.mean(value))
                    else:
                        value = float(value)
                    importance_dict[name] = value
                    
                # For the shap_values list, handle potential multi-dimensional arrays
                shap_value_list = []
                for value in shap_values[i]:
                    if isinstance(value, np.ndarray):
                        shap_value_list.append(float(np.mean(value)))
                    else:
                        shap_value_list.append(float(value))
                    
                explanations[horse_name] = {
                    'feature_importance': importance_dict,
                    'base_value': base_value,
                    'shap_values': shap_value_list
                }
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error generating explanations: {str(e)}")
            raise
            
    def generate_natural_language_explanation(
        self,
        horse_name: str,
        explanation: Dict[str, Any]
    ) -> str:
        """Generate natural language explanation from SHAP values."""
        try:
            importances = explanation['feature_importance']
            base_value = explanation['base_value']
            
            # Sort features by absolute importance
            sorted_features = sorted(
                importances.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            # Generate explanation
            lines = [f"Analysis for {horse_name}:"]
            
            # Add base probability
            base_prob = 1 / (1 + np.exp(-base_value))
            lines.append(f"Base win probability: {base_prob:.1%}")
            
            # Add top contributing factors
            lines.append("\nTop contributing factors:")
            for feature, importance in sorted_features[:3]:
                if abs(importance) < 0.01:
                    continue
                direction = "increases" if importance > 0 else "decreases"
                feature_name = feature.replace("_", " ").title()
                lines.append(
                    f"- {feature_name} {direction} win probability "
                    f"by {abs(importance):.2f}"
                )
                
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error generating natural language explanation: {str(e)}")
            raise

def explain_predictions(model, X_sample, max_display=5):
    """Generate SHAP explanations for model predictions.
    
    Args:
        model: Trained model instance
        X_sample: Feature matrix to explain
        max_display: Maximum number of features to show in plots
    
    Returns:
        SHAP values for the predictions
    """
    try:
        # Create tree explainer for Random Forest
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values and ensure proper conversion to dense array
        shap_values = explainer(X_sample)
        if hasattr(shap_values, 'values'):
            shap_values = shap_values.values  # Convert from object to numpy array
        
        # Convert to dense array if sparse
        if isinstance(shap_values, np.ndarray) and hasattr(shap_values, 'todense'):
            shap_values = shap_values.todense()
            
        # Handle multi-class case
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]  # Take positive class values
            
        print("Generating SHAP summary plot...")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X_sample,
            max_display=max_display,
            show=False,
            plot_type="bar"
        )
        plt.tight_layout()
        plt.savefig("shap_summary_plot.png")
        plt.close()
        print("SHAP plot saved as shap_summary_plot.png")
        
        return shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values
        
    except Exception as e:
        logger.error(f"Error generating SHAP explanations: {str(e)}")
        raise