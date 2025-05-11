
# app.py

import streamlit as st
from betting.kelly_calculator import calculate_kelly_bets
from explainability.nlp_betting_summary import generate_summary

def present_results(predictions, explanations=None):
    st.title("AI Horse Racing Analyst")
    
    st.subheader("Top Betting Recommendations")
    bets = calculate_kelly_bets(predictions)
    for bet in bets:
        st.markdown(f"**{bet['horse']}**: R{bet['stake']} at {bet['odds']} odds (p={bet['probability']})")
    
    st.subheader("Natural Language Summary")
    summary = generate_summary(bets)
    st.text_area("Betting Summary", summary, height=200)
    
    if explanations:
        st.subheader("Feature Importance (SHAP)")
        st.image("shap_summary_plot.png")
