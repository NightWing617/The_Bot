
# bankroll_manager.py

class BankrollManager:
    def __init__(self, initial_bankroll=1000.0):
        self.bankroll = initial_bankroll
        self.history = []

    def update(self, results):
        for result in results:
            if result['win']:
                payout = result['stake'] * result['odds']
                self.bankroll += payout - result['stake']
            else:
                self.bankroll -= result['stake']
            self.history.append({
                'horse': result['horse'],
                'stake': result['stake'],
                'odds': result['odds'],
                'win': result['win'],
                'bankroll': round(self.bankroll, 2)
            })

    def get_current_bankroll(self):
        return round(self.bankroll, 2)

    def get_history(self):
        return self.history
