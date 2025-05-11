
# reallocation_logic.py

def reallocate_bankroll(bets, new_bankroll):
    total_stake = sum(b['stake'] for b in bets)
    scale = new_bankroll / total_stake if total_stake > 0 else 0

    for bet in bets:
        bet['stake'] = round(bet['stake'] * scale, 2)

    return bets
