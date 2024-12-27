import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

class DraftBasedPredictor:
    """
    Predicts match outcomes based on completed team drafts and historical performance
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the predictor with historical match data
        
        Args:
            data_path: Path to the processed match data CSV
        """
        self.df = pd.read_csv(data_path)
        self.champion_encoders = {}
        self.model = None
        self._prepare_data()

    def _prepare_data(self):
        """Prepare and encode the draft data"""
        # Encode champions for each pick position
        for i in range(1, 6):
            pick_col = f'pick{i}'
            self.champion_encoders[pick_col] = LabelEncoder()
            self.df[f'{pick_col}_encoded'] = self.champion_encoders[pick_col].fit_transform(self.df[pick_col])

        # Create feature matrix
        self.features = [
            # Encoded picks
            'pick1_encoded', 'pick2_encoded', 'pick3_encoded', 
            'pick4_encoded', 'pick5_encoded',
            
            # Historical performance
            'winrate_pick1', 'winrate_pick2', 'winrate_pick3', 
            'winrate_pick4', 'winrate_pick5',
            
            # Pick frequency
            'count_pick1', 'count_pick2', 'count_pick3', 
            'count_pick4', 'count_pick5'
        ]

        self.X = self.df[self.features]
        self.y = self.df['result']

    def train_model(self, test_size: float = 0.2):
        """
        Train the prediction model
        
        Args:
            test_size: Proportion of data to use for testing
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=42, 
            stratify=self.y
        )

        # Initialize and train model
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)

        # Print performance metrics
        print("\nModel Performance Metrics:")
        print(f"Cross-validation score: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Plot feature importance
        self._plot_feature_importance()

    def _plot_feature_importance(self):
        """Visualize feature importance"""
        importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance.head(10), x='importance', y='feature')
        plt.title('Top 10 Most Important Features for Prediction')
        plt.tight_layout()
        plt.show()

    def get_champion_stats(self, team_name: str, picks: list[str]) -> list[dict]:
        """
        Get detailed statistics for each champion pick
        
        Args:
            team_name: Name of the team
            picks: List of champion picks
            
        Returns:
            List of dictionaries containing champion statistics
        """
        stats = []
        for i, pick in enumerate(picks, 1):
            # Get team-specific stats for specific pick order
            team_pick_stats = self.df[
                (self.df['teamname'] == team_name) & 
                (self.df[f'pick{i}'] == pick)
            ]
            
            # Get overall stats for specific pick order
            overall_stats = self.df[self.df[f'pick{i}'] == pick]
            
            # Get team stats for this champion regardless of pick order
            team_all_pick_stats = self.df[
                (self.df['teamname'] == team_name) & 
                (self.df[['pick1', 'pick2', 'pick3', 'pick4', 'pick5']] == pick).any(axis=1)
            ]
            
            # Get overall stats for this champion regardless of pick order
            overall_all_pick_stats = self.df[
                (self.df[['pick1', 'pick2', 'pick3', 'pick4', 'pick5']] == pick).any(axis=1)
            ]
            
            stats.append({
                'position': i,
                'champion': pick,
                # Stats for specific pick order
                'team_games': len(team_pick_stats) if not team_pick_stats.empty else 0,
                'team_winrate': team_pick_stats['result'].mean() if not team_pick_stats.empty else 0,
                'overall_games': len(overall_stats) if not overall_stats.empty else 0,
                'overall_winrate': overall_stats['result'].mean() if not overall_stats.empty else 0,
                # Stats regardless of pick order
                'team_all_pick_games': len(team_all_pick_stats) if not team_all_pick_stats.empty else 0,
                'team_all_pick_winrate': team_all_pick_stats['result'].mean() if not team_all_pick_stats.empty else 0,
                'overall_all_pick_games': len(overall_all_pick_stats) if not overall_all_pick_stats.empty else 0,
                'overall_all_pick_winrate': overall_all_pick_stats['result'].mean() if not overall_all_pick_stats.empty else 0
            })
        
        return stats
    

    def predict_match(self, 
                     team1_name: str,
                     team1_picks: list[str],
                     team2_name: str,
                     team2_picks: list[str]) -> dict:
        """
        Predict match outcome based on completed drafts
        
        Args:
            team1_name: Name of first team
            team1_picks: List of 5 champions picked by team1
            team2_name: Name of second team
            team2_picks: List of 5 champions picked by team2
            
        Returns:
            Dictionary containing prediction details
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        # Validate inputs
        if len(team1_picks) != 5 or len(team2_picks) != 5:
            raise ValueError("Each team must have exactly 5 picks")

        # Process team drafts
        team1_features = self._process_team_draft(team1_name, team1_picks)
        team2_features = self._process_team_draft(team2_name, team2_picks)

        # Get win probabilities
        team1_prob = self.model.predict_proba([team1_features])[0][1]
        team2_prob = self.model.predict_proba([team2_features])[0][1]

        # Normalize probabilities
        total = team1_prob + team2_prob
        team1_prob /= total
        team2_prob /= total

        # Create result dictionary
        result = {
            'team1': {
                'name': team1_name,
                'picks': team1_picks,
                'win_probability': team1_prob,
                'champion_stats': self.get_champion_stats(team1_name, team1_picks)
            },
            'team2': {
                'name': team2_name,
                'picks': team2_picks,
                'win_probability': team2_prob,
                'champion_stats': self.get_champion_stats(team2_name, team2_picks)
            }
            
        }

        return result

    def print_detailed_prediction(self, result: dict):
        """
        Print detailed prediction results including champion statistics
        
        Args:
            result: Prediction result dictionary
        """
        print("\n=== Match Prediction ===")
        print(f"\nOverall Win Probabilities:")
        print(f"{result['team1']['name']}: {result['team1']['win_probability']:.1%}")
        print(f"{result['team2']['name']}: {result['team2']['win_probability']:.1%}")
        
        print("\n=== Champion Statistics ===")
        
        # Print team-specific champion stats
        for team in ['team1', 'team2']:
            print(f"\n{result[team]['name']} Draft:")
            for stat in result[team]['champion_stats']:
                print(f"\nPosition {stat['position']} - {stat['champion']}:")
                print(f"  Team Stats (Pick {stat['position']}) : {stat['team_winrate']:.1%} win rate ({stat['team_games']} games)")
                print(f"  Team Stats (All Picks)     : {stat['team_all_pick_winrate']:.1%} win rate ({stat['team_all_pick_games']} games)")
                print(f"  Overall Stats (Pick {stat['position']}) : {stat['overall_winrate']:.1%} win rate ({stat['overall_games']} games)")
                print(f"  Overall Stats (All Picks)   : {stat['overall_all_pick_winrate']:.1%} win rate ({stat['overall_all_pick_games']} games)")
        
    def _process_team_draft(self, team_name: str, picks: list[str]) -> list:
        """Process draft picks into model features"""
        features = []

        # Encode picks
        for i, pick in enumerate(picks, 1):
            try:
                encoded_pick = self.champion_encoders[f'pick{i}'].transform([pick])[0]
                features.append(encoded_pick)
            except ValueError:
                raise ValueError(f"Unknown champion '{pick}' for {team_name}'s pick {i}")

        # Get historical performance stats
        for i, pick in enumerate(picks, 1):
            team_pick_stats = self.df[
                (self.df['teamname'] == team_name) & 
                (self.df[f'pick{i}'] == pick)
            ]
            
            winrate = team_pick_stats[f'winrate_pick{i}'].mean() if not team_pick_stats.empty else 0
            count = team_pick_stats[f'count_pick{i}'].mean() if not team_pick_stats.empty else 0
            
            features.extend([winrate, count])

        return features
