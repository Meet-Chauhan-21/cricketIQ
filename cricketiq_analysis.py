"""
CricketIQ - Data Analytics Project

This Python script performs comprehensive analysis on IPL cricket matches dataset
including data cleaning, exploratory data analysis, visualization, and predictive modeling.

Author: College Data Science Project
Dataset: IPL Cricket Matches (2008-2020)
"""

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Set plot style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def load_data(file_path="matches.csv"):
    """
    Load the cricket matches dataset from CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing match data
        
    Returns:
    --------
    df : pandas.DataFrame
        Loaded dataset
    """
    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)
    
    df = pd.read_csv(file_path)
    
    print(f"✓ Dataset loaded successfully!")
    print(f"  Total records: {len(df)}")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Shape: {df.shape}")
    print()
    
    return df


def clean_data(df):
    """
    Clean the dataset by handling missing values and converting data types.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw dataset
        
    Returns:
    --------
    df : pandas.DataFrame
        Cleaned dataset
    """
    print("=" * 60)
    print("CLEANING DATASET")
    print("=" * 60)
    
    # Display missing values before cleaning
    missing_before = df.isnull().sum().sum()
    print(f"Missing values before cleaning: {missing_before}")
    
    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # Fill categorical columns with mode or 'Unknown'
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()
            if not mode_value.empty:
                df[col] = df[col].fillna(mode_value[0])
            else:
                df[col] = df[col].fillna("Unknown")
    
    # Display missing values after cleaning
    missing_after = df.isnull().sum().sum()
    print(f"Missing values after cleaning: {missing_after}")
    print("✓ Data cleaning complete!")
    print()
    
    return df


def perform_eda(df):
    """
    Perform Exploratory Data Analysis on the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned dataset
    """
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Univariate Analysis
    print("\n--- UNIVARIATE ANALYSIS ---\n")
    
    print("Top 10 Winning Teams:")
    print(df["winner"].value_counts().head(10))
    print()
    
    print("Top 10 Player of the Match Winners:")
    print(df["player_of_match"].value_counts().head(10))
    print()
    
    print("Matches per Season:")
    print(df["season"].value_counts().sort_index())
    print()
    
    print("Toss Decision Distribution:")
    print(df["toss_decision"].value_counts())
    print()
    
    # Bivariate Analysis
    print("\n--- BIVARIATE ANALYSIS ---\n")
    
    # Toss winner vs match winner
    df["toss_match_same"] = (df["toss_winner"] == df["winner"]).astype(int)
    toss_impact = df["toss_match_same"].value_counts()
    toss_percentage = (toss_impact.get(1, 0) / len(df) * 100)
    
    print(f"Toss Winner also won match: {toss_percentage:.2f}%")
    print()
    
    # Multivariate Analysis
    print("\n--- MULTIVARIATE ANALYSIS ---\n")
    
    print("Correlation Matrix (Numeric Features):")
    numeric_features = df.select_dtypes(include=["number"]).columns.tolist()
    correlation_matrix = df[numeric_features].corr()
    print(correlation_matrix)
    print()


def create_visualizations(df):
    """
    Create comprehensive visualizations for data analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned dataset
    """
    print("=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Visualization 1: Matches per Season
    plt.figure(figsize=(14, 6))
    season_counts = df["season"].value_counts().sort_index()
    sns.barplot(x=season_counts.index, y=season_counts.values, palette="viridis")
    plt.title("Matches Per Season", fontsize=16, fontweight='bold')
    plt.xlabel("Season", fontsize=12)
    plt.ylabel("Number of Matches", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("visualization_1_matches_per_season.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Visualization 1: Matches per Season - Saved")
    
    # Visualization 2: Top 10 Winning Teams
    plt.figure(figsize=(14, 6))
    top_winners = df["winner"].value_counts().head(10)
    sns.barplot(x=top_winners.index, y=top_winners.values, palette="magma")
    plt.title("Top 10 Winning Teams", fontsize=16, fontweight='bold')
    plt.xlabel("Team", fontsize=12)
    plt.ylabel("Number of Wins", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("visualization_2_top_winning_teams.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Visualization 2: Top Winning Teams - Saved")
    
    # Visualization 3: Player of the Match Frequency
    plt.figure(figsize=(14, 6))
    pom_counts = df["player_of_match"].value_counts().head(10)
    sns.barplot(x=pom_counts.index, y=pom_counts.values, palette="cubehelix")
    plt.title("Top 10 Player of the Match Winners", fontsize=16, fontweight='bold')
    plt.xlabel("Player Name", fontsize=12)
    plt.ylabel("Number of Awards", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("visualization_3_player_of_match.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Visualization 3: Player of the Match - Saved")
    
    # Visualization 4: Matches per City
    plt.figure(figsize=(14, 6))
    city_counts = df["city"].value_counts().head(15)
    sns.barplot(x=city_counts.index, y=city_counts.values, palette="coolwarm")
    plt.title("Top 15 Cities Hosting Matches", fontsize=16, fontweight='bold')
    plt.xlabel("City", fontsize=12)
    plt.ylabel("Number of Matches", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("visualization_4_matches_per_city.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Visualization 4: Matches per City - Saved")
    
    # Visualization 5: Toss Decision Analysis
    plt.figure(figsize=(8, 6))
    toss_decision_counts = df["toss_decision"].value_counts()
    colors = ['#ff9999', '#66b3ff']
    plt.pie(toss_decision_counts.values, labels=toss_decision_counts.index, 
            autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 12})
    plt.title("Toss Decision Distribution", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("visualization_5_toss_decision.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Visualization 5: Toss Decision - Saved")
    
    # Visualization 6: Toss Winner vs Match Winner
    plt.figure(figsize=(8, 6))
    toss_vs_win = df["toss_match_same"].value_counts().rename({1: "Yes", 0: "No"})
    sns.barplot(x=toss_vs_win.index, y=toss_vs_win.values, palette="Set2")
    plt.title("Did Toss Winner Also Win the Match?", fontsize=16, fontweight='bold')
    plt.xlabel("Toss Winner = Match Winner", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.tight_layout()
    plt.savefig("visualization_6_toss_impact.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Visualization 6: Toss Impact - Saved")
    
    # Visualization 7: Distribution of Win by Runs
    plt.figure(figsize=(12, 6))
    sns.histplot(df["win_by_runs"], bins=30, kde=True, color="teal")
    plt.title("Distribution of Win By Runs", fontsize=16, fontweight='bold')
    plt.xlabel("Win By Runs", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.savefig("visualization_7_win_by_runs.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Visualization 7: Win by Runs Distribution - Saved")
    
    # Visualization 8: Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_features = df.select_dtypes(include=["number"]).columns.tolist()
    corr = df[numeric_features].corr()
    sns.heatmap(corr, annot=True, cmap="RdYlBu", fmt=".2f", center=0)
    plt.title("Correlation Heatmap (Numeric Features)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("visualization_8_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Visualization 8: Correlation Heatmap - Saved")
    
    # Visualization 9: Top Teams Performance Across Seasons
    plt.figure(figsize=(14, 6))
    top_teams = df["winner"].value_counts().head(5).index
    team_season_perf = df[df["winner"].isin(top_teams)].groupby(["season", "winner"]).size().reset_index(name="wins")
    sns.lineplot(data=team_season_perf, x="season", y="wins", hue="winner", marker="o", linewidth=2)
    plt.title("Top 5 Teams Performance Across Seasons", fontsize=16, fontweight='bold')
    plt.xlabel("Season", fontsize=12)
    plt.ylabel("Number of Wins", fontsize=12)
    plt.legend(title="Team", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("visualization_9_team_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Visualization 9: Team Performance - Saved")
    
    # Visualization 10: Top Venues
    plt.figure(figsize=(14, 6))
    venue_counts = df["venue"].value_counts().head(10)
    sns.barplot(x=venue_counts.values, y=venue_counts.index, palette="rocket")
    plt.title("Top 10 Venues by Matches Hosted", fontsize=16, fontweight='bold')
    plt.xlabel("Number of Matches", fontsize=12)
    plt.ylabel("Venue", fontsize=12)
    plt.tight_layout()
    plt.savefig("visualization_10_top_venues.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Visualization 10: Top Venues - Saved")
    
    print("\n✓ All visualizations created and saved successfully!")
    print()


def feature_engineering(df):
    """
    Create new features for model improvement.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned dataset
        
    Returns:
    --------
    df : pandas.DataFrame
        Dataset with new features
    """
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    # Feature 1: Toss winner advantage
    df["toss_winner_advantage"] = (df["toss_winner"] == df["winner"]).astype(int)
    print("✓ Created: toss_winner_advantage")
    
    # Feature 2: Total wins per team
    team_total_wins = df["winner"].value_counts().to_dict()
    df["total_wins_per_team"] = df["winner"].map(team_total_wins).fillna(0).astype(int)
    print("✓ Created: total_wins_per_team")
    
    # Feature 3: Match year
    df["match_year"] = df["date"].dt.year
    print("✓ Created: match_year")
    
    print("\n✓ Feature engineering complete!")
    print()
    
    return df


def train_regression_model(df):
    """
    Train Linear Regression model to predict win_by_runs.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with features
        
    Returns:
    --------
    model : sklearn.Pipeline
        Trained regression model
    metrics : dict
        Model evaluation metrics
    """
    print("=" * 60)
    print("TRAINING REGRESSION MODEL (Win by Runs Prediction)")
    print("=" * 60)
    
    # Prepare data
    reg_features = ["toss_decision", "dl_applied", "season", "toss_winner_advantage"]
    reg_target = "win_by_runs"
    
    reg_df = df[reg_features + [reg_target]].dropna().copy()
    X_reg = reg_df[reg_features]
    y_reg = reg_df[reg_target]
    
    print(f"Dataset shape: {X_reg.shape}")
    
    # Define preprocessing
    reg_cat_cols = ["toss_decision"]
    reg_num_cols = ["dl_applied", "season", "toss_winner_advantage"]
    
    reg_preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), reg_cat_cols),
            ("num", "passthrough", reg_num_cols)
        ]
    )
    
    # Create pipeline
    reg_model = Pipeline(steps=[
        ("preprocessor", reg_preprocessor),
        ("model", LinearRegression())
    ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Train model
    reg_model.fit(X_train, y_train)
    y_pred = reg_model.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Model Evaluation ---")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    print()
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'y_test': y_test,
        'y_pred': y_pred
    }
    
    return reg_model, metrics


def train_classification_model(df):
    """
    Train Logistic Regression model to predict match winner.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with features
        
    Returns:
    --------
    model : sklearn.Pipeline
        Trained classification model
    metrics : dict
        Model evaluation metrics
    """
    print("=" * 60)
    print("TRAINING CLASSIFICATION MODEL (Match Winner Prediction)")
    print("=" * 60)
    
    # Prepare data
    clf_df = df.copy()
    clf_df = clf_df[clf_df["winner"].isin(clf_df[["team1", "team2"]].stack().unique())].copy()
    
    clf_df["match_result_class"] = np.where(
        clf_df["winner"] == clf_df["team1"],
        "Team1 Win",
        "Team2 Win"
    )
    
    clf_features = [
        "team1", "team2", "toss_winner", "toss_decision",
        "dl_applied", "season", "toss_winner_advantage"
    ]
    target_col = "match_result_class"
    
    model_df = clf_df[clf_features + [target_col]].dropna().copy()
    X_clf = model_df[clf_features]
    y_clf = model_df[target_col]
    
    print(f"Dataset shape: {X_clf.shape}")
    print(f"Class distribution:")
    print(y_clf.value_counts())
    
    # Define preprocessing
    clf_cat_cols = ["team1", "team2", "toss_winner", "toss_decision"]
    clf_num_cols = ["dl_applied", "season", "toss_winner_advantage"]
    
    clf_preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), clf_cat_cols),
            ("num", "passthrough", clf_num_cols)
        ]
    )
    
    # Create pipeline
    clf_model = Pipeline(steps=[
        ("preprocessor", clf_preprocessor),
        ("model", LogisticRegression(max_iter=2000, random_state=42))
    ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Train model
    clf_model.fit(X_train, y_train)
    y_pred = clf_model.predict(X_test)
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=["Team1 Win", "Team2 Win"])
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Team1 Win", "Team2 Win"],
                yticklabels=["Team1 Win", "Team2 Win"],
                cbar_kws={'label': 'Count'})
    plt.title("Confusion Matrix - Logistic Regression", fontsize=16, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Confusion matrix saved as 'confusion_matrix.png'")
    print()
    
    metrics = {
        'accuracy': acc,
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_pred': y_pred
    }
    
    return clf_model, metrics


def evaluate_models(reg_metrics, clf_metrics, df):
    """
    Evaluate and display final model performance and insights.
    
    Parameters:
    -----------
    reg_metrics : dict
        Regression model metrics
    clf_metrics : dict
        Classification model metrics
    df : pandas.DataFrame
        Original dataset
    """
    print("=" * 80)
    print("FINAL MODEL EVALUATION AND PROJECT INSIGHTS")
    print("=" * 80)
    print()
    
    # Model Performance
    print("🤖 MODEL PERFORMANCE:")
    print(f"   • Linear Regression (Win by Runs):")
    print(f"      - MSE: {reg_metrics['mse']:.2f}")
    print(f"      - RMSE: {reg_metrics['rmse']:.2f}")
    print(f"      - R² Score: {reg_metrics['r2']:.4f}")
    print(f"   • Logistic Regression (Match Winner):")
    print(f"      - Accuracy: {clf_metrics['accuracy']:.4f} ({clf_metrics['accuracy']*100:.2f}%)")
    print()
    
    # Dataset Insights
    most_wins_team = df["winner"].value_counts().idxmax()
    most_wins_count = df["winner"].value_counts().max()
    toss_effect_pct = round(df["toss_winner_advantage"].mean() * 100, 2)
    most_matches_city = df["city"].value_counts().idxmax()
    top_stadium = df["venue"].value_counts().idxmax()
    top_player = df["player_of_match"].value_counts().idxmax()
    top_player_count = df["player_of_match"].value_counts().max()
    
    print("📊 DATASET INSIGHTS:")
    print(f"   • Total matches analyzed: {len(df)}")
    print(f"   • Seasons covered: {df['season'].min()} to {df['season'].max()}")
    print()
    
    print("🏆 KEY FINDINGS:")
    print(f"   • Most successful team: {most_wins_team} ({most_wins_count} wins)")
    print(f"   • Toss winner also won match: {toss_effect_pct}% of the time")
    print(f"   • City hosting most matches: {most_matches_city}")
    print(f"   • Most popular stadium: {top_stadium}")
    print(f"   • Top Player of the Match: {top_player} ({top_player_count} awards)")
    print()
    
    print("=" * 80)
    print()
    print("✓ CricketIQ Analysis Complete!")
    print()


def main():
    """
    Main function to execute the complete analysis pipeline.
    """
    print("\n")
    print("=" * 80)
    print(" " * 20 + "CRICKETIQ DATA ANALYTICS PROJECT")
    print("=" * 80)
    print()
    
    # Step 1: Load data
    df = load_data("matches.csv")
    
    # Step 2: Clean data
    df = clean_data(df)
    
    # Step 3: Perform EDA
    perform_eda(df)
    
    # Step 4: Create visualizations
    create_visualizations(df)
    
    # Step 5: Feature engineering
    df = feature_engineering(df)
    
    # Step 6: Train regression model
    reg_model, reg_metrics = train_regression_model(df)
    
    # Step 7: Train classification model
    clf_model, clf_metrics = train_classification_model(df)
    
    # Step 8: Evaluate models and display insights
    evaluate_models(reg_metrics, clf_metrics, df)
    
    print("=" * 80)
    print("All analysis steps completed successfully!")
    print("Check the current directory for saved visualizations and reports.")
    print("=" * 80)


# Entry point
if __name__ == "__main__":
    main()
