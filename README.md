
# 🧠 AI Horse Racing Analyst

A high-performance AI system designed to analyze horse racing data, predict outcomes with statistical confidence, and intelligently size bets using the Kelly Criterion. The architecture is modular, interpretable, and designed for real-time adaptability.

---

## 🎯 Goal

To maximize betting profitability and prediction confidence using a rich blend of structured racecard data, machine learning models, and real-time market signals.

---

## 💡 Features

- PDF & API-based racecard ingestion
- Feature engineering for form, speed, and sentiment
- Probabilistic modeling with scikit-learn
- Kelly Criterion-based bet sizing
- Real-time updates (odds, scratches, weather)
- SHAP-based explainability
- Streamlit UI, Discord bot, and Obsidian export
- Modular folder structure with reusable components

---

## 🧱 Folder Structure

```
horse-racing-ai/
├── data/
├── ingestion/
├── preprocessing/
├── modeling/
├── betting/
├── explainability/
├── interface/
├── realtime_adapter/
├── utils/
├── notebooks/
├── main.py
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

1. Clone the repo and install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the full pipeline:

```bash
python main.py
```

3. Or launch the Streamlit dashboard:

```bash
streamlit run interface/app.py
```

---

## ⚙️ Config

Edit `utils/config.yaml` to customize bankroll, Kelly fraction, and data paths.

---

## 🤖 GitHub Copilot Usage

This codebase is structured with modular functions and docstrings to help GitHub Copilot provide intelligent code suggestions. Each component follows a single responsibility principle, making it easy for Copilot to:
- Suggest additional feature transformations
- Tune models
- Add UI enhancements
- Detect and correct bugs

---

## 📌 TODOs for Collaboration

- [ ] Improve odds scraping robustness
- [ ] Add LSTM or XGBoost support
- [ ] Integrate with bookmaker APIs for auto-betting
- [ ] Add unit tests across modules

---

## 🧠 Inspired By

Horse racing prediction models, sports betting research, and AI interpretability frameworks (SHAP, LIME).

---

## 📜 License

MIT License

---

## 🙋‍♂️ Maintainer

Built by an AI development partner for high-confidence, data-driven horse racing analysis.

