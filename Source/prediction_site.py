"""
prediction_site.py - Streamlit application for LCK Match Prediction
"""
import streamlit as st
import pandas as pd
import joblib
from predictor_new import DraftBasedPredictor
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="LCK Match Predictor",
    page_icon="🎮",
    layout="wide"
)

# Initialize predictor
@st.cache_resource
def load_predictor():
    """Load and initialize the predictor with model"""
    try:
        model_data = joblib.load('../Models/draft_predictor_best_model.joblib')
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load champion list
@st.cache_data
def load_champion_list():
    """Load and cache the list of available champions"""
    df = pd.read_csv("../Data/LCK_Tournament.csv")
    champions = sorted(list(set(
        df['pick1'].unique().tolist() +
        df['pick2'].unique().tolist() +
        df['pick3'].unique().tolist() +
        df['pick4'].unique().tolist() +
        df['pick5'].unique().tolist()
    )))
    return champions

def create_radar_chart(team_stats: dict, team_name: str):
    """
    Tạo biểu đồ radar cho các chỉ số quan trọng của team với các chỉ số được cải tiến
    """
    categories = [
        'Combat Power', 'Objective Control', 'Economy', 
        'Vision Control', 'Early Game', 'Late Game'
    ]
    
    # Chuẩn hóa các chỉ số thành điểm từ 0-100
    values = [
        # Combat Power (KDA + Damage)
        min(100, (
            (team_stats['kills'] * 3 + team_stats['assists']) / max(1, team_stats['deaths']) * 10 +  # KDA impact
            team_stats['damagetochampions'] / 1000  # Damage impact
        ) / 2),
        
        # Objective Control (First objectives + Total objectives)
        min(100, (
            # First objectives
            (team_stats['firstdragon'] * 10) +     
            (team_stats['firstherald'] * 15) +    
            (team_stats['firstbaron'] * 30) +
            (team_stats['firsttower']) * 20 +    
            (team_stats['firsttothreetowers']) * 20 +
            # Total objectives
            (team_stats['dragons'] - 1)  * 10 +    
            (team_stats['barons'] / 3) * 20       
        )),
        
        # Economy (Earned GPM - Gold Per Minute)
        min(100, (
            (team_stats['earned gpm'] / 1500) * 100  # Earned GPM (max ~2000): 100% weight
        )),
        
        # Vision Control (Vision Score)
        min(100, (
            (team_stats['visionscore'] / 400) * 100  # Vision Score (max ~300): 100% weight
        )),
        
        # Early Game (First 15 minutes)
        min(100, 50 + (
            (team_stats['golddiffat15'] / 1000) * 20 +  # Gold diff at 15
            team_stats['firstblood'] * 5 +            # First blood bonus
            team_stats['firsttower'] * 15 +            # First tower bonus
            (team_stats['csdiffat15'] / 10) * 3        # CS diff at 15
        )),
        
        # Late Game (Post 20 minutes performance)
        min(100, (
            # Gold difference scaling
            ((team_stats['golddiffat25'] / 5000) * 25) + 50 +  # Gold diff at 25 (normalized around 50)
            
            # Late game objectives
            (team_stats['barons'] / 3) * 15 +                   
            
            # Late game economy
            (team_stats['earned gpm'] / 2000) * 20 +  
            
            # Late game combat
            ((team_stats['damagetochampions'] / team_stats['gamelength']) / 2000) * 15  # DPM in late game
        )),
    ]

    # Ensure all values are between 0 and 100
    values = [max(0, min(100, v)) for v in values]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=team_name,
        line=dict(color='#1f77b4', width=2),  # Customize line style
        fillcolor='rgba(31, 119, 180, 0.3)'   # Semi-transparent fill
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10),
                gridcolor='rgba(0,0,0,0.1)',
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color='black'),
                gridcolor='rgba(0,0,0,0.1)',
            ),
            bgcolor='white'
        ),
        showlegend=True,
        title=dict(
            text=f"Team Performance Analysis - {team_name}",
            x=0,
            y=0.95,
            font=dict(size=16)
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(t=100, b=50, l=50, r=50)
    )
    
    return fig

# def create_win_probability_gauge(probability: float, team_name: str):
    """
    Tạo biểu đồ đồng hồ cho xác suất thắng
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        title = {'text': f"{team_name} Win Probability"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    return fig

def show_key_factors(result: dict):
    """
    Hiển thị các yếu tố chính ảnh hưởng đến dự đoán cho cả hai đội
    """
    st.subheader("Thông số về draft pick đang lựa chọn")
    
    # Tạo hai hàng columns cho hai đội
    st.write(f"### {result['team1']['name']}")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Tỉ lệ thắng trung bình của draft của các đội",
            f"{(result['team1']['champion_stats'][0]['overall_winrate'] + result['team1']['champion_stats'][1]['overall_winrate'])/2:.1%}"
        )
    
    with col2:
        st.metric(
            "Số ván đấu các đội đã sử dụng draft này",
            sum(stat['team_games'] for stat in result['team1']['champion_stats'])
        )

    # Thông tin cho đội 2
    st.write(f"### {result['team2']['name']}")
    col4, col5 = st.columns(2)
    
    with col4:
        st.metric(
            "Tỉ lệ thắng trung bình của draft của các đội",
            f"{(result['team2']['champion_stats'][0]['overall_winrate'] + result['team2']['champion_stats'][1]['overall_winrate'])/2:.1%}"
        )
    
    with col5:
        st.metric(
            "Số ván đấu các đội đã sử dụng draft này",
            sum(stat['team_games'] for stat in result['team2']['champion_stats'])
        )

def main():
    st.title("Dự đoán kết quả LCK 🎮")
    st.write("Dự đoán tỉ lệ thắng dựa trên đội tuyển và lượt pick tướng")
    
    # Load data
    predictor = load_predictor()
    champions = load_champion_list()
    
    # Create two columns for team inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Đội 1 (Đội xanh)")
        team1_name = st.text_input("Tên đội:", value="T1")
        st.write("Chọn tướng:")
        team1_picks = []
        for i in range(5):
            pick = st.selectbox(
                f"Pick {i+1}",
                options=champions,
                key=f"team1_pick{i}"
            )
            team1_picks.append(pick)

    with col2:
        st.subheader("Đội 2 (Đội đỏ)")
        team2_name = st.text_input("Tên đội:", value="Gen.G")
        st.write("Chọn tướng:")
        team2_picks = []
        for i in range(5):
            pick = st.selectbox(
                f"Pick {i+1}",
                options=champions,
                key=f"team2_pick{i}"
            )
            team2_picks.append(pick)

    # Predict button
    if st.button("Kết quả"):
        try:
            # Get prediction
            result = predictor.predict_match(
                team1_name=team1_name,
                team1_picks=team1_picks,
                team2_name=team2_name,
                team2_picks=team2_picks
            )
            
            # Display results
            st.header("Kết quả dự đoán")
            
            # Win probabilities
            st.subheader("Tỉ lệ thắng mỗi đội")
            prob_col1, prob_col2 = st.columns(2)
            with prob_col1:
                st.metric(
                    team1_name,
                    f"{result['team1']['win_probability']:.1%}"
                )
            with prob_col2:
                st.metric(
                    team2_name,
                    f"{result['team2']['win_probability']:.1%}"
                )
                
            show_key_factors(result)
            
            # Thêm phân tích chi tiết
            st.header("Phân tích chi tiết")
            
            # 1. Biểu đồ Radar cho phân tích team
            st.subheader("Phân tích hiệu suất đội tuyển")
            radar_col1, radar_col2 = st.columns(2)
            
            with radar_col1:
                if 'recent_stats' in result['team1']:
                    radar1 = create_radar_chart(result['team1']['recent_stats'], team1_name)
                    st.plotly_chart(radar1, use_container_width=True)
                else:
                    st.warning(f"Không có dữ liệu gần đây cho {team1_name}")
            
            with radar_col2:
                if 'recent_stats' in result['team2']:
                    radar2 = create_radar_chart(result['team2']['recent_stats'], team2_name)
                    st.plotly_chart(radar2, use_container_width=True)
                else:
                    st.warning(f"Không có dữ liệu gần đây cho {team2_name}")
            
            # # 2. Biểu đồ xác suất thắng
            # st.subheader("Xác suất chiến thắng")
            # gauge_col1, gauge_col2 = st.columns(2)
            
            # with gauge_col1:
            #     gauge1 = create_win_probability_gauge(result['team1']['win_probability'], team1_name)
            #     st.plotly_chart(gauge1, use_container_width=True)
            
            # with gauge_col2:
            #     gauge2 = create_win_probability_gauge(result['team2']['win_probability'], team2_name)
            #     st.plotly_chart(gauge2, use_container_width=True)
            
            # 3. Thống kê chi tiết
            st.subheader("Thống kê chi tiết thông số trong 10 ván đấu gần nhất")
            stats_col1, stats_col2 = st.columns(2)
            
            with stats_col1:
                st.write(f"### {team1_name}")
                if 'recent_stats' in result['team1']:
                    stats = result['team1']['recent_stats']
                    st.metric("First Dragon Rate", f"{stats['firstdragon'] * 100:.1f}%")
                    st.metric("First Herald Rate", f"{stats['firstherald'] * 100:.1f}%")
                    st.metric("First Tower Rate", f"{stats['firsttower'] * 100:.1f}%")
                    st.metric("GPM", f"{stats['earned gpm']:.1f}")
                    st.metric("Vision Score", f"{stats['visionscore']:.1f}")
                    st.metric("First Blood Rate", f"{stats['firstblood'] * 100:.1f}%")
                    st.metric("Gold Diff @15", f"{stats['golddiffat15']:.0f}")
                    st.metric("Gold Diff @20", f"{stats['golddiffat20']:.0f}")
            
            with stats_col2:
                st.write(f"### {team2_name}")
                if 'recent_stats' in result['team2']:
                    stats = result['team2']['recent_stats']
                    st.metric("First Dragon Rate", f"{stats['firstdragon'] * 100:.1f}%")
                    st.metric("First Herald Rate", f"{stats['firstherald'] * 100:.1f}%")
                    st.metric("First Tower Rate", f"{stats['firsttower'] * 100:.1f}%")
                    st.metric("GPM", f"{stats['earned gpm']:.1f}")
                    st.metric("Vision Score", f"{stats['visionscore']:.1f}")
                    st.metric("First Blood Rate", f"{stats['firstblood'] * 100:.1f}%")
                    st.metric("Gold Diff @15", f"{stats['golddiffat15']:.0f}")
                    st.metric("Gold Diff @20", f"{stats['golddiffat20']:.0f}")
            
            st.subheader("Thông số các tướng lựa chọn")
            
            # Team 1 Stats
            st.write(f"\n{team1_name} Draft:")
            for stat in result['team1']['champion_stats']:
                with st.expander(f"Lượt {stat['position']} - {stat['champion']}"):
                    st.write(f"Winrate của đội khi pick ở lượt {stat['position']}: {stat['team_winrate']:.1%} ({stat['team_games']} games)")
                    st.write(f"Winrate của đội khi pick tướng này: {stat['team_all_pick_winrate']:.1%} ({stat['team_all_pick_games']} games)")
                    st.write(f"Winrate của tướng này khi pick ở lượt {stat['position']}: {stat['overall_winrate']:.1%} ({stat['overall_games']} games)")
                    st.write(f"Winrate tổng của tướng này: {stat['overall_all_pick_winrate']:.1%} ({stat['overall_all_pick_games']} games)")
            
            # Team 2 Stats
            st.write(f"\n{team2_name} Draft:")
            for stat in result['team2']['champion_stats']:
                with st.expander(f"Lượt {stat['position']} - {stat['champion']}"):
                    st.write(f"Winrate của đội khi pick ở lượt {stat['position']}: {stat['team_winrate']:.1%} ({stat['team_games']} games)")
                    st.write(f"Winrate của đội khi pick tướng này: {stat['team_all_pick_winrate']:.1%}({stat['team_all_pick_games']} games)")
                    st.write(f"Winrate của tướng này khi pick ở lượt {stat['position']}: {stat['overall_winrate']:.1%} ({stat['overall_games']} games)")
                    st.write(f"Winrate tổng của tướng này: {stat['overall_all_pick_winrate']:.1%} ({stat['overall_all_pick_games']} games)")
            

        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()