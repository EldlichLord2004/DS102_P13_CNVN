from predictor_new import DraftBasedPredictor
import numpy as np

class DraftOptimizer:
    def __init__(self, predictor: DraftBasedPredictor):
        """
        Khởi tạo DraftOptimizer với một predictor đã được huấn luyện
        
        Args:
            predictor: Đối tượng DraftBasedPredictor đã được train
        """
        self.predictor = predictor
        # Lấy danh sách tướng từ dữ liệu của predictor
        self.champion_pool = self._get_valid_champions()
        
    def _get_valid_champions(self):
        """Lấy danh sách tất cả tướng hợp lệ từ dữ liệu"""
        valid_champions = set()
        for i in range(1, 6):
            valid_champions.update(self.predictor.df[f'pick{i}'].unique())
        return list(valid_champions)

    def suggest_improvements(self, team1_name, team1_picks, team2_name, team2_picks, top_n=5):
        """Đề xuất cải thiện cho đội yếu hơn"""
        try:
            # Kiểm tra tỉ lệ thắng hiện tại
            current_result = self.predictor.predict_match(team1_name, team1_picks, team2_name, team2_picks)
            
            # Xác định đội yếu hơn
            team1_prob = current_result['team1']['win_probability']
            team2_prob = current_result['team2']['win_probability']
            weaker_team = 1 if team1_prob < team2_prob else 2
            
            team_name = team1_name if weaker_team == 1 else team2_name
            current_picks = team1_picks.copy() if weaker_team == 1 else team2_picks.copy()
            opponent_picks = team2_picks.copy() if weaker_team == 1 else team1_picks.copy()
            opponent_name = team2_name if weaker_team == 1 else team1_name
            current_prob = team1_prob if weaker_team == 1 else team2_prob
            
            # Chỉ lấy những tướng đã có trong dữ liệu training
            valid_champions = set()
            for i in range(1, 6):
                pick_col = f'pick{i}'
                valid_champions.update(self.predictor.champion_encoders[pick_col].classes_)
            
            # Lọc danh sách tướng hợp lệ
            valid_new_champs = [champ for champ in valid_champions 
                              if champ not in current_picks and champ not in opponent_picks]
            
            improvements = []
            for pos in range(5):
                position_improvements = []
                current_champ = current_picks[pos]
                
                for new_champ in valid_new_champs:
                    try:
                        new_picks = current_picks.copy()
                        new_picks[pos] = new_champ
                        
                        if weaker_team == 1:
                            new_result = self.predictor.predict_match(
                                team_name, new_picks, opponent_name, opponent_picks
                            )
                            new_prob = new_result['team1']['win_probability']
                        else:
                            new_result = self.predictor.predict_match(
                                opponent_name, opponent_picks, team_name, new_picks
                            )
                            new_prob = new_result['team2']['win_probability']
                        
                        if new_prob > current_prob:
                            position_improvements.append({
                                'position': pos + 1,
                                'old_champion': current_champ,
                                'new_champion': new_champ,
                                'new_win_probability': new_prob,
                                'improvement': new_prob - current_prob
                            })
                    except Exception as e:
                        continue
                
                if position_improvements:
                    best_position_improvement = max(
                        position_improvements,
                        key=lambda x: x['improvement']
                    )
                    improvements.append(best_position_improvement)
                    print(f"Tìm thấy cải thiện: {current_champ} -> "
                          f"{best_position_improvement['new_champion']} "
                          f"(+{best_position_improvement['improvement']:.2%})")
            
            # Sắp xếp theo mức độ cải thiện
            improvements.sort(key=lambda x: x['improvement'], reverse=True)
            
            return {
                'current_win_probability': current_prob,
                'team_to_improve': team_name,
                'best_improvements': improvements[:top_n] if improvements else []
            }
            
        except Exception as e:
            print(f"Lỗi khi tìm cải thiện: {str(e)}")
            return None