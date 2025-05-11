
# kelly_calculator.py

def calculate_kelly_bets(predictions, bankroll=1000.0, kelly_fraction=1.0):
    bets = []
    for pred in predictions:
        p = pred['probability']
        odds = pred['odds']
        b = odds - 1
        q = 1 - p

        kelly_value = (b * p - q) / b
        stake = max(kelly_value * bankroll * kelly_fraction, 0)
        bets.append({
            'horse': pred['horse'],
            'stake': round(stake, 2),
            'kelly_value': round(kelly_value, 3),
            'probability': p,
            'odds': odds
        })
    return bets
