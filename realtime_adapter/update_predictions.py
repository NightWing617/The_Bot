
# update_predictions.py

def re_rank_predictions(predictions):
    # Sort horses by probability
    return sorted(predictions, key=lambda x: x['probability'], reverse=True)
