# live_odds_monitor.py

import time
import requests
from datetime import datetime
from typing import Dict, List, Optional
import backoff  # Add to requirements.txt
from utils.logger import get_logger

logger = get_logger(__name__)

class OddsMonitor:
    def __init__(self, base_url: str, api_key: str, retry_attempts: int = 3):
        self.base_url = base_url
        self.api_key = api_key
        self.retry_attempts = retry_attempts
        self.last_update = None
        self.cache = {}
        
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, ValueError),
        max_tries=3
    )
    def _fetch_odds(self, race_id: str) -> dict:
        """Fetch latest odds with exponential backoff retry."""
        try:
            headers = {'Authorization': f'Bearer {self.api_key}'}
            response = requests.get(
                f"{self.base_url}/odds/{race_id}",
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            self.last_update = datetime.now()
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching odds: {str(e)}")
            raise
        
    def _validate_odds(self, odds_data: dict) -> bool:
        """Validate odds data structure and values."""
        if not isinstance(odds_data, dict):
            return False
            
        for horse, data in odds_data.items():
            if not isinstance(data, dict):
                return False
            if 'odds' not in data:
                return False
            try:
                odds = float(data['odds'])
                if odds <= 1.0:  # Odds must be greater than 1
                    return False
            except (ValueError, TypeError):
                return False
        
        return True
    
    def _detect_significant_changes(
        self,
        old_odds: Dict[str, float],
        new_odds: Dict[str, float],
        threshold: float = 0.2
    ) -> List[dict]:
        """Detect significant odds changes."""
        changes = []
        for horse, new_odd in new_odds.items():
            if horse in old_odds:
                old_odd = old_odds[horse]
                pct_change = abs(new_odd - old_odd) / old_odd
                if pct_change > threshold:
                    changes.append({
                        'horse': horse,
                        'old_odds': old_odd,
                        'new_odds': new_odd,
                        'pct_change': pct_change
                    })
        return changes

def update_with_live_data(
    predictions: List[dict],
    race_id: Optional[str] = None,
    live_updates: Optional[dict] = None
) -> List[dict]:
    """
    Update predictions with live odds data.
    
    Args:
        predictions: List of prediction dictionaries
        race_id: Optional race identifier for fetching live odds
        live_updates: Optional manual updates (for testing/fallback)
    
    Returns:
        Updated list of predictions
    """
    try:
        logger.info("Starting live odds update...")
        
        if not predictions:
            raise ValueError("No predictions provided")
        
        # Use provided updates or try to fetch live data
        updates = live_updates or {}
        if not updates and race_id:
            try:
                # Initialize with environment variables in production
                monitor = OddsMonitor(
                    base_url="http://api.odds.example.com",
                    api_key="test_key"
                )
                updates = monitor._fetch_odds(race_id)
            except Exception as e:
                logger.error(f"Failed to fetch live odds: {str(e)}")
                return predictions  # Return original predictions on error
        
        # Process updates
        updated = []
        for pred in predictions:
            horse = pred['horse']
            new_pred = pred.copy()
            
            if horse in updates:
                try:
                    horse_updates = updates[horse]
                    
                    # Update odds if available
                    if 'odds' in horse_updates:
                        new_odds = float(horse_updates['odds'])
                        if new_odds > 1.0:  # Validate odds
                            new_pred['odds'] = new_odds
                            logger.info(f"Updated odds for {horse}: {new_odds}")
                    
                    # Handle scratched horses
                    if horse_updates.get('scratched', False):
                        logger.warning(f"Horse scratched: {horse}")
                        continue
                    
                    # Add any additional live updates
                    new_pred['last_updated'] = datetime.now().isoformat()
                    
                except (ValueError, TypeError) as e:
                    logger.error(f"Error processing updates for {horse}: {str(e)}")
                    new_pred = pred  # Keep original on error
            
            updated.append(new_pred)
        
        if not updated:
            logger.warning("No valid predictions after updates")
            return predictions
        
        # Sort by updated probabilities
        updated.sort(key=lambda x: x.get('probability', 0), reverse=True)
        
        logger.info(f"Successfully updated {len(updated)} predictions")
        return updated
        
    except Exception as e:
        logger.error(f"Error in update_with_live_data: {str(e)}")
        return predictions  # Return original predictions on error
