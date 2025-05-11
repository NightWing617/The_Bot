
# nlp_betting_summary.py

def generate_summary(bets):
    summary_lines = []
    for bet in bets:
        line = (
            f"Horse '{bet['horse']}' has a predicted win probability of {bet['probability']*100:.1f}%, "
            f"odds of {bet['odds']}, and a recommended stake of R{bet['stake']:.2f} "
            f"(Kelly fraction: {bet['kelly_value']:.2f})."
        )
        if bet['kelly_value'] > 0.1:
            line += " High confidence bet."
        elif bet['kelly_value'] < 0.01:
            line += " Caution: Low edge."
        summary_lines.append(line)
    return "\n".join(summary_lines)
