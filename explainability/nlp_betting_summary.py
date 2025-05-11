import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class BettingSummaryGenerator:
    def __init__(self):
        self.min_confidence = 0.6  # Minimum probability to consider "strong favorite"
        self.min_value_threshold = 0.1  # Minimum Kelly value for "value bet"
        
    def _validate_predictions(self, predictions: List[Dict[str, Any]]) -> bool:
        """Validate prediction data structure."""
        if not predictions:
            return True  # Empty predictions are valid but will get empty summary
            
        required_fields = ['horse', 'probability', 'odds']
        for pred in predictions:
            if not all(field in pred for field in required_fields):
                return False
        return True
        
    def _generate_key_insights(self, predictions: List[Dict[str, Any]]) -> List[str]:
        """Generate key betting insights."""
        insights = []
        
        if not predictions:
            insights.append("No predictions available for this race.")
            return insights
            
        # Find strong favorites
        favorites = [
            p for p in predictions
            if p.get('probability', 0) >= self.min_confidence
        ]
        if favorites:
            horses = ', '.join(p['horse'] for p in favorites)
            insights.append(
                f"Strong favorites identified: {horses} "
                f"({', '.join(f'{p['probability']:.1%}' for p in favorites)} win probability)"
            )
            
        # Identify value bets
        value_bets = [
            p for p in predictions
            if p.get('kelly_value', 0) >= self.min_value_threshold
        ]
        if value_bets:
            horses = ', '.join(p['horse'] for p in value_bets)
            insights.append(
                f"Value betting opportunities: {horses} "
                f"(Kelly values: {', '.join(f'{p['kelly_value']:.2f}' for p in value_bets)})"
            )
            
        # Risk assessment
        high_risk = any(p.get('kelly_value', 0) > 0.5 for p in predictions)
        if high_risk:
            insights.append(
                "High-risk betting scenario detected. "
                "Consider reducing stakes."
            )
            
        return insights
        
    def _generate_individual_analysis(
        self,
        predictions: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate per-horse analysis."""
        analysis = []
        
        for pred in predictions:
            horse = pred.get('horse', 'Unknown')
            prob = pred.get('probability', 0)
            odds = pred.get('odds', 0)
            kelly = pred.get('kelly_value', 0)
            
            lines = [f"\n{horse}:"]
            lines.append(f"- Win Probability: {prob:.1%}")
            lines.append(f"- Current Odds: {odds:.1f}")
            
            if kelly > 0:
                lines.append(
                    f"- Recommended Stake: {kelly:.1%} of bankroll"
                )
                
                # Add reasoning
                if prob >= self.min_confidence:
                    lines.append("- Strong winning chance")
                if kelly >= self.min_value_threshold:
                    lines.append("- Good value bet")
                    
            else:
                lines.append("- No betting value found")
                
            analysis.extend(lines)
            
        return analysis
        
    def generate_summary(
        self,
        predictions: List[Dict[str, Any]],
        bankroll: float = 1000.0
    ) -> str:
        """Generate comprehensive betting summary.
        
        Args:
            predictions: List of prediction dictionaries
            bankroll: Available bankroll
            
        Returns:
            String containing formatted betting summary
        """
        try:
            if not self._validate_predictions(predictions):
                return "Error: Invalid prediction format"
                
            if not predictions:
                return (
                    "Race Analysis Summary\n\n"
                    "No predictions available for this race.\n"
                    "Unable to generate betting recommendations."
                )
                
            # Generate sections
            sections = []
            
            # 1. Header
            sections.append("Race Analysis Summary\n")
            
            # 2. Key Insights
            insights = self._generate_key_insights(predictions)
            if insights:
                sections.append("Key Insights:")
                sections.extend(f"• {insight}" for insight in insights)
                
            # 3. Individual Analysis
            sections.append("\nIndividual Horse Analysis:")
            analysis = self._generate_individual_analysis(predictions)
            sections.extend(analysis)
            
            # 4. Bankroll Management
            total_stake = sum(
                p.get('kelly_value', 0) * bankroll
                for p in predictions
            )
            sections.append(
                f"\nTotal Recommended Stakes: £{total_stake:.2f} "
                f"({(total_stake/bankroll):.1%} of bankroll)"
            )
            
            return "\n".join(sections)
            
        except Exception as e:
            logger.error(f"Error generating betting summary: {str(e)}")
            return f"Error generating betting summary"

def generate_summary(
    predictions: List[Dict[str, Any]],
    explanations: Optional[Dict] = None,
    bankroll: float = 1000.0
) -> str:
    """
    Main entry point for generating betting summaries.
    
    Args:
        predictions: List of prediction dictionaries
        explanations: Optional SHAP explanations
        bankroll: Total bankroll for context
    
    Returns:
        Formatted summary string
    """
    generator = BettingSummaryGenerator()
    return generator.generate_summary(predictions, explanations, bankroll)