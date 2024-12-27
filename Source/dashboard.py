import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple

class LOLDashboard:
    """Enhanced Dashboard for League of Legends match analysis"""
    
    def __init__(self) -> None:
        """Initialize dashboard with improved configuration"""
        st.set_page_config(
            page_title="LCK Tournament Analysis",
            page_icon="🎮",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        self.data: pd.DataFrame = self._load_and_preprocess_data()
        
    def _load_and_preprocess_data(self) -> pd.DataFrame:
        """Load and preprocess the tournament data"""
        try:
            df = pd.read_csv("../Data/LCK_Tournament.csv")
            
            # Calculate additional metrics
            df["kda"] = (df["kills"] + df["assists"]) / df["deaths"].replace(0, 1)
            df["vision_per_min"] = df["visionscore"] / (df["gamelength"] / 60)
            df["gold_per_min"] = df["totalgold"] / (df["gamelength"] / 60)
            df["cs_per_min"] = (df["minionkills"] + df["monsterkills"]) / (df["gamelength"] / 60)
            
            # Add GPR analysis helpers
            df["gpr_category"] = pd.cut(df["gpr"], 
                                    bins=[-float("inf"), -4, -2, 0, 2, 4, float("inf")],
                                    labels=["Very Poor", "Poor", "Below Average", "Above Average", "Good", "Excellent"])
          
            
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()

    def create_sidebar_filters(self) -> Tuple[int, str, str]:
        """Create and handle sidebar filters"""
        with st.sidebar:
            st.header("Filters")
            
            # Year filter
            years = sorted(self.data["year"].unique())
            year = st.selectbox("Select Year", years, index=len(years)-1)
            
            # Split filter
            splits = ["All Splits"] + sorted(self.data["split"].unique().tolist())
            split = st.selectbox("Select Split", splits)
            
            # Team filter
            teams = sorted(self.data["teamname"].unique())
            team = st.selectbox("Select Team", teams)
            
        return year, split, team

    def create_time_progression_plot(self, data: pd.DataFrame) -> None:
        """Create time progression analysis plots"""
        st.subheader("Game Progression Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gold difference progression
            fig_gold = go.Figure()
            for time in [10, 15, 20]:
                fig_gold.add_box(
                    y=data[f"golddiffat{time}"],
                    name=f"{time} min"
                )
            fig_gold.update_layout(
                title="Gold Difference Progression",
                yaxis_title="Gold Difference",
                showlegend=False
            )
            st.plotly_chart(fig_gold, use_container_width=True)
            
        with col2:
            # Kills progression
            fig_kills = go.Figure()
            for time in [10, 15, 20]:
                fig_kills.add_box(
                    y=data[f"killsat{time}"],
                    name=f"{time} min"
                )
            fig_kills.update_layout(
                title="Kills Progression",
                yaxis_title="Kills",
                showlegend=False
            )
            st.plotly_chart(fig_kills, use_container_width=True)

    def create_objective_control_analysis(self, data: pd.DataFrame) -> None:
        """Create objective control analysis section"""
        st.subheader("Objective Control")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Dragon control
            avg_dragons = data["dragons"].mean()
            avg_opp_dragons = data["opp_dragons"].mean()
            
            fig_dragons = go.Figure(data=[
                go.Bar(name="Team Dragons", y=[avg_dragons]),
                go.Bar(name="Enemy Dragons", y=[avg_opp_dragons])
            ])
            fig_dragons.update_layout(title="Average Dragon Control")
            st.plotly_chart(fig_dragons, use_container_width=True)
            
        with col2:
            # Tower control
            avg_towers = data["towers"].mean()
            avg_opp_towers = data["opp_towers"].mean()
            
            fig_towers = go.Figure(data=[
                go.Bar(name="Team Towers", y=[avg_towers]),
                go.Bar(name="Enemy Towers", y=[avg_opp_towers])
            ])
            fig_towers.update_layout(title="Average Tower Control")
            st.plotly_chart(fig_towers, use_container_width=True)

    def create_game_length_analysis(self, data: pd.DataFrame) -> None:
        """Create game length analysis section"""
        st.subheader("Game Duration Analysis")
        
        # Convert game length to minutes
        data["game_minutes"] = data["gamelength"] / 60
        
        fig = px.histogram(
            data,
            x="game_minutes",
            nbins=20,
            title="Game Duration Distribution"
        )
        fig.update_layout(
            xaxis_title="Game Duration (minutes)",
            yaxis_title="Number of Games"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def create_gpr_performance_analysis(self, data: pd.DataFrame) -> None:
        """Create enhanced GPR performance analysis section"""
        st.subheader("GPR Performance Analysis")

        # 1. GPR Distribution and Game Length Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # GPR Distribution
            fig_dist = px.histogram(
                data,
                x="gpr",
                nbins=20,
                title="GPR Distribution",
                color="result",  # Color by win/loss
                barmode="overlay",
                color_discrete_map={0: "#EF553B", 1: "#636EFA"}  # Red for loss, Blue for win
            )
            fig_dist.update_layout(
                xaxis_title="GPR",
                yaxis_title="Number of Games",
                showlegend=True,
                legend=dict(
                    title="Result",
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # GPR by Game Length
            fig_scatter = px.scatter(
                data,
                x="gamelength",
                y="gpr",
                color="result",
                title="GPR by Game Length",
                color_discrete_map={0: "#EF553B", 1: "#636EFA"},
                trendline="ols"  # Add trend line
            )
            fig_scatter.update_layout(
                xaxis_title="Game Length (seconds)",
                yaxis_title="GPR",
                showlegend=True,
                legend=dict(
                    title="Result",
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # 2. GPR Statistics Table and Visualization
        st.subheader("GPR Performance Categories")
        
        # Calculate GPR statistics
        gpr_stats = data.groupby("gpr_category").agg({
            "result": ["count", "mean"],
            "gpr": ["mean", "std", "min", "max"],
            "kills": "mean",
            "deaths": "mean",
            "assists": "mean",
            "totalgold": "mean"
        }).round(3)
        
        # Remove rows with NaN values
        gpr_stats = gpr_stats.dropna()

        # Rename columns for clarity
        gpr_stats.columns = [
            "Games Played",
            "Win Rate",
            "Avg GPR",
            "GPR Std Dev",
            "Min GPR",
            "Max GPR",
            "Avg Kills",
            "Avg Deaths",
            "Avg Assists",
            "Avg Gold"
        ]

        # Convert win rate to percentage
        gpr_stats["Win Rate"] = (gpr_stats["Win Rate"] * 100).round(1).astype(str) + '%'
        
        # Format gold values
        gpr_stats["Avg Gold"] = gpr_stats["Avg Gold"].apply(lambda x: f"{x:,.0f}")
        
        # Reset index to make gpr_category a column
        gpr_stats = gpr_stats.reset_index()

        # Create two columns for stats
        col1, col2 = st.columns([2, 3])

        with col1:
            # Display statistics table with styling
            st.dataframe(
                gpr_stats,
                use_container_width=True,
                height=300,
                column_config={
                    "gpr_category": st.column_config.Column(
                    "Performance Category",
                    help="GPR performance category",
                    width="medium"
                    ),
                    "Games Played": st.column_config.NumberColumn(
                        help="Number of games played in this category"
                    ),
                    "Win Rate": st.column_config.TextColumn(
                        help="Percentage of games won in this category"
                    ),
                    "Avg GPR": st.column_config.NumberColumn(
                        help="Average Gold Percentage Rating",
                        format="%.2f"
                    ),
                    "GPR Std Dev": st.column_config.NumberColumn(
                        help="Standard deviation of GPR",
                        format="%.2f"
                    )
                }
            )

        with col2:
            # Create visualization of GPR categories
            fig = go.Figure()
            
            # Add Win Rate bars
            fig.add_trace(go.Bar(
                name='Win Rate',
                x=gpr_stats.index,
                y=[float(x.strip('%')) for x in gpr_stats['Win Rate']],
                yaxis='y',
                marker_color='#1f77b4',
                opacity=0.7
            ))

            # Add Average GPR line
            fig.add_trace(go.Scatter(
                name='Avg GPR',
                x=gpr_stats.index,
                y=gpr_stats['Avg GPR'],
                yaxis='y2',
                line=dict(color='#ff7f0e', width=3),
                mode='lines+markers'
            ))

            # Update layout
            fig.update_layout(
                title='Win Rate and Average GPR by Performance Category',
                yaxis=dict(
                    title='Win Rate (%)',
                    titlefont=dict(color='#1f77b4'),
                    tickfont=dict(color='#1f77b4'),
                    side='left'
                ),
                yaxis2=dict(
                    title='Average GPR',
                    titlefont=dict(color='#ff7f0e'),
                    tickfont=dict(color='#ff7f0e'),
                    overlaying='y',
                    side='right'
                ),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)

        # 3. GPR Insights
        st.subheader("GPR Performance Insights")
        
        # Calculate key insights
        best_category = gpr_stats.sort_values("Win Rate", ascending=False).index[0]
        highest_gpr = float(gpr_stats["Avg GPR"].max())
        most_games = gpr_stats["Games Played"].idxmax()
        
        # Display insights
        insights_col1, insights_col2, insights_col3 = st.columns(3)
        
        with insights_col1:
            st.info(f"Best Performing Category: **{best_category}**")
        with insights_col2:
            st.info(f"Highest Average GPR: **{highest_gpr:.2f}**")
        with insights_col3:
            st.info(f"Most Common Category: **{most_games}**")

        # Add correlation analysis
        if len(data) > 1:
            correlation = data["gpr"].corr(data["result"])
            st.write(f"GPR-Win Correlation: **{correlation:.3f}**")
    
    def create_overview_metrics(self, data: pd.DataFrame) -> None:
        """Create overview metrics section with GPR"""
        st.header("Team Performance Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            win_rate = (data["result"].mean() * 100).round(2)
            st.metric("Win Rate", f"{win_rate}%")
            
        with col2:
            avg_kda = data["kda"].mean().round(2)
            st.metric("Average KDA", avg_kda)
            
        with col3:
            avg_gpm = data["gold_per_min"].mean().round(2)
            st.metric("Gold per Minute", avg_gpm)
            
        with col4:
            avg_vision = data["vision_per_min"].mean().round(2)
            st.metric("Vision Score per Minute", avg_vision)
            
        with col5:
            avg_gpr = data["gpr"].mean().round(2)
            st.metric("Average GPR", avg_gpr)

    def render_dashboard(self) -> None:
        """Render the complete dashboard"""
        st.title("LCK Tournament Analysis Dashboard")
        
        # Apply filters
        year, split, team = self.create_sidebar_filters()
        
        # Filter data
        filtered_data = self.data[self.data["teamname"] == team]
        if split != "All Splits":
            filtered_data = filtered_data[filtered_data["split"] == split]
        filtered_data = filtered_data[filtered_data["year"] == year]
        
        if filtered_data.empty:
            st.warning("No data available for selected filters.")
            return
            
        # Create dashboard sections
        self.create_overview_metrics(filtered_data)
        self.create_gpr_performance_analysis(filtered_data)
        self.create_time_progression_plot(filtered_data)
        self.create_objective_control_analysis(filtered_data)
        self.create_game_length_analysis(filtered_data)

def main() -> None:
    """Main function to run the dashboard"""
    dashboard = LOLDashboard()
    dashboard.render_dashboard()

if __name__ == "__main__":
    main()