#Class chứa các hàm phục vụ cho mô hình dự đoán

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

#Models
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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
            'count_pick4', 'count_pick5',
            
            # New performance features
            # Overall performance
            'kills', 'deaths', 'assists', 'team kpm', 'ckpm','gspd','gpr','gamelength',
            # Objectives
            'firstblood', 'firstdragon', 'dragons', 'elementaldrakes',
            'firstherald', 'heralds', 'firstbaron', 'barons', 'firsttower',
            'firstmidtower','firsttothreetowers','turretplates',
            # Economy
            'earned gpm','goldat15', 'goldat20', 'goldat25',
            'golddiffat15', 'golddiffat20','golddiffat25', 'xpdiffat20', 'xpdiffat25',
            # Vision
            'wardsplaced', 'visionscore', 'wardskilled','controlwardsbought',
            # Farm
            'cspm', 'minionkills', 'monsterkills', 'csat15', 'csdiffat15',
            'csat20', 'csdiffat20', 'csat25','csdiffat25',
            #Combat
            'damagetochampions', 'damagetakenperminute', 'damagemitigatedperminute'   
            
        ]
        
        # Kiểm tra số lượng đặc trưng
        print(f"Total features: {len(self.features)}")

        self.X = self.df[self.features]
        self.y = self.df['result']
        
    def _get_team_recent_stats(self, team_name: str, n_matches: int = 10) -> dict:
        """
        Get average stats from N most recent matches for a team based on date
        
        Args:
            team_name: Name of the team
            n_matches: Number of recent matches to consider
            
        Returns:
            Dictionary containing average stats from recent matches
        """
        # Convert date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(self.df['date']):
            self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Get team matches and sort by date
        team_matches = self.df[self.df['teamname'] == team_name]
        
        # Validate if team exists
        if team_matches.empty:
            raise ValueError(f"No matches found for team: {team_name}")
        
        # Get n most recent matches
        n_available = len(team_matches)
        if n_available < n_matches:
            print(f"Warning: Only {n_available} matches found for {team_name} (requested {n_matches})")
            n_matches = n_available
        
        recent_matches = team_matches.sort_values('date', ascending=False).head(n_matches)
        
        # Calculate stats
        stats = {}
        for feature in self.features:
            if feature not in ['pick1_encoded', 'pick2_encoded', 'pick3_encoded', 
                            'pick4_encoded', 'pick5_encoded']:
                try:
                    stats[feature] = recent_matches[feature].mean()
                except KeyError:
                    print(f"Warning: Feature '{feature}' not found in data")
                    stats[feature] = 0
                except Exception as e:
                    print(f"Error calculating {feature}: {str(e)}")
                    stats[feature] = 0
        
        # Optional: Add date range info for debugging
        stats['_date_range'] = {
            'newest_match': recent_matches['date'].max().strftime('%Y-%m-%d'),
            'oldest_match': recent_matches['date'].min().strftime('%Y-%m-%d'),
            'matches_used': len(recent_matches)
        }
        
        return stats

    def train_model(self, test_size: float = 0.2):
        """
        Train and evaluate multiple models, then fine-tune the best performing one
        
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
        
        print(f"Training features shape: {self.X.shape}")  # In ra để kiểm tra
        print("Features used in training:")
        for f in self.features:
            print(f"- {f}")

        # Define base models
        models = {
            "XGBoost": xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            "MLP": MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=300,
                random_state=42
            ),
            #  "Stacking": StackingClassifier(
            #     estimators=[
            #         ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            #         ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42)),
            #         ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', alpha=0.001, learning_rate='adaptive', max_iter=300, random_state=42))
            #     ],
            #     final_estimator=LogisticRegression(),
            #     cv=5
            # ),
            "RandomForest": RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                random_state=42
            )
        }
        
        # Create pipelines for each model
        pipelines = {
            name: Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ]) for name, model in models.items()
        }
        
        # Train and evaluate each model
        print("=== Initial Model Evaluation ===")
        results = {}
        for name, pipeline in pipelines.items():
            print(f"\nTraining {name}...")
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'pipeline': pipeline,
                'accuracy': accuracy
            }
            print(f"{name} Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            # Vẽ biểu đồ đánh giá overfitting
            train_sizes, train_scores, test_scores = learning_curve(
                pipeline, self.X, self.y, cv=5, scoring='accuracy',
                n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            plt.figure()
            plt.plot(train_sizes, train_mean, 'o-', color="r",
                     label="Training score")
            plt.plot(train_sizes, test_mean, 'o-', color="g",
                     label="Cross-validation score")
            plt.fill_between(train_sizes, train_mean - train_std,
                             train_mean + train_std, alpha=0.1, color="r")
            plt.fill_between(train_sizes, test_mean - test_std,
                             test_mean + test_std, alpha=0.1, color="g")
            plt.title(f'Learning Curve for {name}')
            plt.xlabel('Training Size')
            plt.ylabel('Accuracy Score')
            plt.legend(loc="best")
            plt.grid()
            plt.show()
            
        # Find best model
        best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
        print(f"\nBest performing model: {best_model_name}")
        
        # def create_stacking_model():
        #     estimators = [
        #         ('rf',RandomForestClassifier())
        #         ('xgb',xgb.XGBClassifier())
        #         ('mlp',MLPClassifier())
        #     ]
            
        #     stacking=StackingClassifier(
        #         estimators=estimators,
        #         final_estimator=LogisticRegression(),
        #         cv=5
        #     )
        #     return stacking
        
        # Define hyperparameter grids for each model
        param_grids = {
            "XGBoost": {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [4, 6, 8],
                'classifier__learning_rate': [0.01, 0.1, 0.3],
                'classifier__min_child_weight': [1, 3, 5],
                'classifier__subsample': [0.8, 0.9, 1.0],
                'classifier__colsample_bytree': [0.8, 0.9, 1.0]
            },
            "MLP": {
                'classifier__hidden_layer_sizes': [
                    (64,32), 
                    (128,64), 
                    (256,128),
                    (128,64,32)
                ],
                'classifier__activation': ['relu', 'tanh'],
                'classifier__alpha': [0.0001, 0.001, 0.01],
                'classifier__learning_rate_init': [0.001, 0.01],
                'classifier__batch_size': [32, 64, 128],
                'classifier__max_iter': [300, 500],
                'classifier__early_stopping': [True],
                'classifier__validation_fraction': [0.1]
            },
            "RandomForest": {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [4, 6, 8],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__max_features': ['sqrt', 'log2']
            },
            # "Stacking": {
            #     'classifier__rf__n_estimators': [100, 200],
            #     'classifier__rf__max_depth': [4, 6],
            #     'classifier__xgb__n_estimators': [100, 200],
            #     'classifier__xgb__max_depth': [4, 6],
            #     'classifier__mlp__hidden_layer_sizes': [(128, 64), (64, 32)],
            #     'classifier__mlp__activation': ['relu', 'tanh'],
            #     'classifier__mlp__alpha': [0.0001, 0.001],
            #     'classifier__mlp__learning_rate_init': [0.001, 0.01],
            #     'classifier__final_estimator__C': [0.1, 1.0]
            # }
        }
        
        # Fine-tune best model
        print(f"\n=== Fine-tuning {best_model_name} ===")
        grid_search = GridSearchCV(
            pipelines[best_model_name],
            param_grids[best_model_name],
            cv=5,
            n_jobs=-1,
            verbose=2,
            scoring='accuracy'
        )
        
        grid_search.fit(X_train, y_train)
        
        # Print results
        print("\nBest parameters found:")
        print(grid_search.best_params_)
        print(f"\nBest cross-validation accuracy: {grid_search.best_score_:.4f}")
        
        # Final evaluation on test set
        y_pred = grid_search.predict(X_test)
        print("\nFinal Model Performance on Test Set:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save the best model
        self.model = grid_search.best_estimator_
        
        # Plot feature importance for the best model
        self._plot_feature_importance()
        
        # Thêm vào sau phần "Initial Model Evaluation"
        print("\n=== K-fold Cross Validation Results ===")
        for name, pipeline in pipelines.items():
            cv_scores = cross_val_score(pipeline, self.X, self.y, cv=5, scoring='accuracy')
            print(f"\n{name}:")
            print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"Individual fold scores: {cv_scores}")
        
        return self.model


    def _plot_feature_importance(self):
        """Visualize feature importance of the best model"""
        if hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.features,
                'importance': self.model.named_steps['classifier'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(12, 6))
            sns.barplot(data=importance.head(15), x='importance', y='feature')
            plt.title('Top 15 Most Important Features for Prediction')
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
    

    def predict_match(self, team1_name: str, team1_picks: list[str],
                    team2_name: str, team2_picks: list[str]) -> dict:
        """Predict match outcome"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        print(f"\nPredicting match: {team1_name} vs {team2_name}")
        print(f"Model expects {len(self.features)} features")

        try:
            # Get recent stats for both teams
            team1_recent_stats = self._get_team_recent_stats(team1_name)
            team2_recent_stats = self._get_team_recent_stats(team2_name)
            
            
            # Process team drafts
            team1_features = self._process_team_draft(team1_name, team1_picks)
            team2_features = self._process_team_draft(team2_name, team2_picks)

            # Verify dimensions
            if len(team1_features) != len(self.features):
                print("\nFeature mismatch details:")
                print(f"Got {len(team1_features)} features:")
                print(f"Expected {len(self.features)} features:")
                raise ValueError(f"Feature count mismatch")

            # Make predictions
            team1_prob = self.model.predict_proba([team1_features])[0][1]
            team2_prob = self.model.predict_proba([team2_features])[0][1]

            # Normalize probabilities
            total = team1_prob + team2_prob
            team1_prob /= total
            team2_prob /= total

            return {
                'team1': {
                    'name': team1_name,
                    'picks': team1_picks,
                    'win_probability': team1_prob,
                    'champion_stats': self.get_champion_stats(team1_name, team1_picks),
                    'recent_stats': team1_recent_stats
                },
                'team2': {
                    'name': team2_name,
                    'picks': team2_picks,
                    'win_probability': team2_prob,
                    'champion_stats': self.get_champion_stats(team2_name, team2_picks),
                    'recent_stats': team2_recent_stats
                }
            }
        except Exception as e:
            raise ValueError(f"Error making prediction: {str(e)}")

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
        
    def _process_team_draft(self, team_name: str, picks: list[str]) -> np.ndarray:
        """Process draft picks into model features"""
        features = []
        
        print("\nProcessing features for prediction:")
        
        # 1. Encode picks (5 features)
        for i, pick in enumerate(picks, 1):
            encoded_pick = self.champion_encoders[f'pick{i}'].transform([pick])[0]
            features.append(encoded_pick)
            # print(f"Added pick{i}_encoded: {encoded_pick}")
        
        # 2. Get draft stats (10 features)
        for i, pick in enumerate(picks, 1):
            team_pick_stats = self.df[
                (self.df['teamname'] == team_name) & 
                (self.df[f'pick{i}'] == pick)
            ]
            winrate = team_pick_stats[f'winrate_pick{i}'].mean() if not team_pick_stats.empty else 0
            count = team_pick_stats[f'count_pick{i}'].mean() if not team_pick_stats.empty else 0
            features.extend([winrate, count])
            # print(f"Added winrate_pick{i}: {winrate}, count_pick{i}: {count}")
        
        # 3. Get recent team stats
        recent_stats = self._get_team_recent_stats(team_name)
        
        # Tạo set các đặc trưng đã thêm để tránh trùng lặp
        added_features = set([f'pick{i}_encoded' for i in range(1, 6)] +
                            [f'winrate_pick{i}' for i in range(1, 6)] +
                            [f'count_pick{i}' for i in range(1, 6)])
        
        # Chỉ thêm các đặc trưng còn lại từ self.features
        for feature in self.features:
            if feature not in added_features:
                value = recent_stats.get(feature, 0)
                features.append(value)
                # print(f"Added {feature}: {value}")
        
        features = np.array(features)
        # print(f"\nTotal features created: {len(features)}")
        # print(f"Expected features: {len(self.features)}")
        
        # Kiểm tra số lượng đặc trưng
        if len(features) != len(self.features):
            print("\nFeature mismatch details:")
            print("Features in model:", len(self.features))
            print("Features created:", len(features))
            print("\nFeatures already added:", sorted(added_features))
            print("\nAll features expected:", sorted(self.features))
            raise ValueError(f"Feature count mismatch: got {len(features)}, expected {len(self.features)}")
        
        return features