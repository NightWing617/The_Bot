
# obsidian_exporter.py

def export_to_obsidian(bets, file_path="RaceReport.md"):
    with open(file_path, "w") as f:
        f.write("# Race Day Betting Summary\n\n")
        for bet in bets:
            f.write(f"## {bet['horse']}\n")
            f.write(f"- Probability: {bet['probability']*100:.2f}%\n")
            f.write(f"- Odds: {bet['odds']}\n")
            f.write(f"- Stake: R{bet['stake']}\n")
            f.write(f"- Kelly Score: {bet['kelly_value']}\n\n")
    print(f"Exported summary to {file_path}")
