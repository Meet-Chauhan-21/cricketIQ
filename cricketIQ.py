'''Understanding IPL match trends and building a predictive model is useful
for cricket analysts, fans, and strategy planning. In this project, we analyze
IPL match data using Python to discover key insights and build a machine
learning model to predict match winners.

We aim to answer questions such as:

1. How are matches distributed across seasons? (Step 6)
2. Which teams win most frequently? (Step 6)
3. Which cities host the most matches? (Step 7)
4. What is the toss decision distribution? (Step 7)
5. How often does toss winner also win the match? (Step 8)
6. Which teams perform best over time? (Step 8)
7. Can we predict match winners from pre-match information? (Step 12)
8. How accurate is the winner prediction model? (Step 13)
9. Which teams are hardest to predict? (Step 14)
10. What practical conclusions can be drawn from the analysis? (Step 15)
'''


# Step 1: Importing necessary Python libraries.

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Step 2: Creating the data frame.

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "matches.csv"
IMAGE_DIR = BASE_DIR / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

dataframe = pd.read_csv(DATA_FILE)

# display all column names
print("------------- Column Names ----------------")
print(dataframe.columns)

# setting pandas display option to display all columns
pd.set_option("display.max_columns", None)

# display top few lines of data
print("\n------------- Top Records ----------------")
print(dataframe.head())

# display shape of data
print("\n------------- Dataset Shape ----------------")
print(dataframe.shape)


# Step 3: Data Cleaning and Preparation
'''Before moving further we need to clean and process the data.
1. Convert date column to datetime format for better time analysis.
2. Fill missing values to avoid issues in EDA and model training.
3. Keep only rows with valid winner/team fields for prediction tasks.
'''

print("\n------------- Data Cleaning ----------------")

# Convert date to datetime, invalid values become NaT
if "date" in dataframe.columns:
    dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")
    print("Converted date column to datetime format")

# Fill categorical missing values
categorical_cols = dataframe.select_dtypes(include=["object", "string"]).columns
for col in categorical_cols:
    dataframe[col] = dataframe[col].fillna("Unknown")

# Fill numeric missing values with median
numeric_cols = dataframe.select_dtypes(include=["number"]).columns
for col in numeric_cols:
    dataframe[col] = dataframe[col].fillna(dataframe[col].median())

# Remove rows that cannot be used for winner prediction
dataframe = dataframe.dropna(subset=["winner", "team1", "team2"])

print("Data cleaning complete")

# Step 4: Getting summary of the dataframe using df.info()
print("\n------------- Dataset Summary ----------------")
dataframe.info()

# Step 4.1: Getting description of the dataframe using df.describe()
print("\n------------- Dataset Description ----------------")
print(dataframe.describe(include="all"))


# Step 5: Checking for missing or null values to identify any data gaps (EDA)
print("\n------------- Missing Values ----------------")
print(dataframe.isnull().sum())


# Step 6: Univariate analysis
# Question 1: How are matches distributed across seasons?
season_counts = dataframe["season"].value_counts().sort_index()
print("\n------------- Matches Per Season ----------------")
print(season_counts)

plt.figure(figsize=(10, 5))
sns.barplot(x=season_counts.index, y=season_counts.values, color="#1f77b4")
plt.title("Matches Per Season")
plt.xlabel("Season")
plt.ylabel("Number of Matches")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "step6_matches_per_season.png", dpi=160)
plt.show()

# Question 2: Which teams win most frequently?
top_winners = dataframe["winner"].value_counts().head(10)
print("\n------------- Top Winning Teams ----------------")
print(top_winners)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_winners.values, y=top_winners.index, color="#ff7f0e")
plt.title("Top 10 Winning Teams")
plt.xlabel("Number of Wins")
plt.ylabel("Team")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "step6_top_winning_teams.png", dpi=160)
plt.show()


# Step 7: Exploring match characteristics
# Question 3: Which cities host the most matches?
city_counts = dataframe["city"].value_counts().head(10)
print("\n------------- Top Match Cities ----------------")
print(city_counts)

plt.figure(figsize=(10, 5))
sns.barplot(x=city_counts.values, y=city_counts.index, color="#2ca02c")
plt.title("Top 10 Cities Hosting IPL Matches")
plt.xlabel("Number of Matches")
plt.ylabel("City")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "step7_top_match_cities.png", dpi=160)
plt.show()

# Question 4: What is the toss decision distribution?
toss_distribution = dataframe["toss_decision"].value_counts()
print("\n------------- Toss Decision Distribution ----------------")
print(toss_distribution)

plt.figure(figsize=(7, 5))
plt.pie(toss_distribution.values, labels=toss_distribution.index, autopct="%1.1f%%", startangle=90)
plt.title("Toss Decision Distribution")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "step7_toss_decision_distribution.png", dpi=160)
plt.show()


# Step 8: Relationship analysis
# Question 5: How often does toss winner also win the match?
dataframe["toss_match_same"] = (dataframe["toss_winner"] == dataframe["winner"]).astype(int)
toss_match_pct = dataframe["toss_match_same"].mean() * 100
print("\n------------- Toss Winner Also Won Match ----------------")
print(f"Percentage: {toss_match_pct:.2f}%")

impact_counts = dataframe["toss_match_same"].value_counts().rename({1: "Yes", 0: "No"})
plt.figure(figsize=(7, 5))
sns.barplot(x=impact_counts.index, y=impact_counts.values, color="#9467bd")
plt.title("Did Toss Winner Also Win?")
plt.xlabel("Toss Winner = Match Winner")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "step8_toss_impact.png", dpi=160)
plt.show()

# Question 6: Which teams perform best over time?
best_teams = dataframe["winner"].value_counts().head(5).index
team_perf = dataframe[dataframe["winner"].isin(best_teams)].groupby(["season", "winner"]).size().reset_index(name="wins")

plt.figure(figsize=(11, 5))
sns.lineplot(data=team_perf, x="season", y="wins", hue="winner", marker="o")
plt.title("Top 5 Teams Performance Across Seasons")
plt.xlabel("Season")
plt.ylabel("Wins")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "step8_top_teams_performance.png", dpi=160)
plt.show()


# Step 9: Prepare features for model
'''Machine learning models require clearly defined features and target.
We use pre-match columns that are available before result is known.
'''

model_columns = [
    "team1",
    "team2",
    "toss_winner",
    "toss_decision",
    "venue",
    "city",
    "winner",
]

model_df = dataframe[model_columns].copy().dropna()
print("\n------------- Model Dataset Shape ----------------")
print(model_df.shape)

X = model_df.drop("winner", axis=1)
y = model_df["winner"]


# Step 10: Split data into train and test
print("\n------------- Splitting Data for Training and Testing ----------------")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Training set: {len(X_train)} samples")
print(f"Testing set: {len(X_test)} samples")


# Step 11: Build preprocessing + model pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), X.columns.tolist())
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=250, random_state=42)),
    ]
)


# Step 12: Train model
print("\n------------- Training Winner Prediction Model ----------------")
model.fit(X_train, y_train)
print("Model Training Completed Successfully!")


# Step 13: Evaluate predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n------------- Model Performance Metrics ----------------")
print(f"Accuracy: {accuracy:.4f}")

if accuracy > 0.70:
    print("Model Performance: Excellent! Strong winner prediction capability.")
elif accuracy > 0.50:
    print("Model Performance: Good. Reasonable predictive performance.")
else:
    print("Model Performance: Needs improvement. Try additional engineered features.")

print("\nClassification report (first 10 labels shown by sklearn behavior):")
print(classification_report(y_test, y_pred, zero_division=0))


# Step 14: Confusion matrix for top teams
# Question 9: Which teams are hardest to predict?
top_labels = y_test.value_counts().head(10).index.tolist()
cm = confusion_matrix(y_test, y_pred, labels=top_labels)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, cmap="Blues", xticklabels=top_labels, yticklabels=top_labels)
plt.title("Confusion Matrix (Top 10 Teams)")
plt.xlabel("Predicted Winner")
plt.ylabel("Actual Winner")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "step14_confusion_matrix_top10.png", dpi=160)
plt.show()


# Step 15: Final summary
# Question 10: What practical conclusions can be drawn from the analysis?
print("\n" + "=" * 70)
print("CRICKETIQ ANALYSIS AND PREDICTION COMPLETE!")
print("=" * 70)
print(f"Total matches analyzed: {len(dataframe)}")
print(f"Prediction accuracy: {accuracy:.2%}")
print(f"Image outputs folder: {IMAGE_DIR}")
print("=" * 70)

print("\nKey conclusions:")
print("1. Match volume and team dominance patterns are clearly visible by season.")
print("2. Toss decision preferences are consistent and measurable.")
print("3. Toss-winning correlation exists but does not fully determine outcomes.")
print("4. Pre-match categorical features can provide useful winner predictions.")
