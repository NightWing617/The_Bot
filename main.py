import sys
from pathlib import Path
import yaml
from utils.logger import get_logger
from data_ingestion.pdf_parser import parse_racecard
from preprocessing.clean_data import clean_race_data
from preprocessing.feature_engineering import engineer_features
from predictive_modeling.model_train import load_model, train_model
from predictive_modeling.predict import predict_outcomes
from betting.kelly_calculator import calculate_kelly_bets
from explainability.shap_explainer import explain_predictions
from explainability.nlp_betting_summary import generate_summary
from interface.app import present_results
from realtime_adapter.live_odds_monitor import update_with_live_data

logger = get_logger(__name__)

def load_config():
    """Load configuration with error handling."""
    try:
        with open("utils/config.yaml", 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("Configuration file not found")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration: {str(e)}")
        raise

def validate_data(data, stage):
    """Validate data at each pipeline stage."""
    if data is None:
        raise ValueError(f"Received None data at stage: {stage}")
    if hasattr(data, 'empty') and data.empty:
        raise ValueError(f"Received empty DataFrame at stage: {stage}")
    return True

def main():
    """Main pipeline with comprehensive error handling."""
    try:
        logger.info("Starting Horse Racing AI System...")
        
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")

        # Step 1: Ingest racecard data
        try:
            racecard_path = Path(config['racecard_pdf'])
            if not racecard_path.exists():
                raise FileNotFoundError(f"Racecard not found at {racecard_path}")
            
            raw_data = parse_racecard(str(racecard_path))
            validate_data(raw_data, "PDF parsing")
            logger.info("Racecard data ingested successfully")
        except Exception as e:
            logger.error(f"Error during racecard ingestion: {str(e)}")
            raise

        # Step 2: Clean and preprocess data
        try:
            cleaned_data = clean_race_data(raw_data)
            validate_data(cleaned_data, "data cleaning")
            
            features_df = engineer_features(cleaned_data)
            validate_data(features_df, "feature engineering")
            logger.info("Data preprocessing completed successfully")
        except Exception as e:
            logger.error(f"Error during data preprocessing: {str(e)}")
            raise

        # Step 3: Load or train model
        try:
            model_path = Path(config.get('model_path', 'models/race_model.pkl'))
            if model_path.exists():
                model = load_model(model_path)
                logger.info(f"Model loaded from {model_path}")
            else:
                logger.warning(f"No model found at {model_path}, training new model")
                historical_data = config.get('historical_data_path')
                if not historical_data:
                    raise ValueError("No historical data path configured for training")
                model, _ = train_model(
                    features_df,
                    target_col='won',
                    save_path=str(model_path)
                )
            logger.info("Model preparation completed successfully")
        except Exception as e:
            logger.error(f"Error during model preparation: {str(e)}")
            raise

        # Step 4: Generate predictions
        try:
            predictions = predict_outcomes(model, features_df)
            validate_data(predictions, "prediction generation")
            logger.info("Predictions generated successfully")
        except Exception as e:
            logger.error(f"Error during prediction generation: {str(e)}")
            raise

        # Step 5: Calculate optimal bets
        try:
            bankroll = config.get('bankroll', 1000.0)
            kelly_fraction = config.get('kelly_fraction', 0.5)
            
            bets = calculate_kelly_bets(
                predictions,
                bankroll=bankroll,
                kelly_fraction=kelly_fraction
            )
            validate_data(bets, "bet calculation")
            logger.info("Betting calculations completed successfully")
        except Exception as e:
            logger.error(f"Error during bet calculation: {str(e)}")
            raise

        # Step 6: Generate explanations
        try:
            explanations = explain_predictions(
                model,
                features_df,
                horse_names=features_df['horse'].tolist()
            )
            summary = generate_summary(bets, explanations, bankroll=bankroll)
            logger.info("Explanations generated successfully")
        except Exception as e:
            logger.error(f"Error generating explanations: {str(e)}")
            raise

        # Step 7: Present results
        try:
            present_results(bets, explanations, summary)
            logger.info("Results presented successfully")
        except Exception as e:
            logger.error(f"Error presenting results: {str(e)}")
            raise

        # Step 8: Monitor for live updates
        try:
            if config.get('enable_live_monitoring', False):
                logger.info("Starting live odds monitoring...")
                updated_bets = update_with_live_data(
                    predictions,
                    race_id=config.get('race_id')
                )
                if updated_bets:
                    present_results(
                        updated_bets,
                        explanations,
                        generate_summary(updated_bets, explanations, bankroll)
                    )
        except Exception as e:
            logger.error(f"Error during live monitoring: {str(e)}")
            # Don't raise here - live monitoring is optional

        logger.info("Pipeline completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())