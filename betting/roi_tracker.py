
# roi_tracker.py

def calculate_roi(history):
    total_staked = sum([h['stake'] for h in history])
    total_return = 0
    for h in history:
        if h['win']:
            total_return += h['stake'] * h['odds']
    net_profit = total_return - total_staked
    roi = (net_profit / total_staked) * 100 if total_staked > 0 else 0
    return round(roi, 2), round(net_profit, 2)
