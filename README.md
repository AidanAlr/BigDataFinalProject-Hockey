# NHL Game Outcome Prediction with PySpark ML

A distributed machine learning pipeline that predicts NHL game outcomes using historical event data and advanced hockey analytics. Built with Apache Spark MLlib.

## Problem Statement

Predict whether an NHL team will **win** a game based on their historical performance metrics computed from in-game events.

**Binary Classification:**

- **1 (Win)**: Team earned 2 points (regulation, OT, or shootout win)
- **0 (Not Win)**: Team earned 0 or 1 points (regulation loss or OT/shootout loss)

## NHL Points System

| Outcome          | Points | Label       |
| ---------------- | ------ | ----------- |
| Regulation Win   | 2      | 1 (Win)     |
| OT/Shootout Win  | 2      | 1 (Win)     |
| OT/Shootout Loss | 1      | 0 (Not Win) |
| Regulation Loss  | 0      | 0 (Not Win) |

## Data

### Event Data (`NHL_EventData.csv`)

Event-level data capturing individual game events with advanced analytics:

| Column         | Description                                                    |
| -------------- | -------------------------------------------------------------- |
| `Season`       | NHL season (e.g., 20152016)                                    |
| `GameID`       | Unique game identifier                                         |
| `EventTeam`    | Team that generated the event                                  |
| `Venue`        | "Home" or "Away"                                               |
| `Corsi`        | Shot attempt indicator (includes goals, shots, blocks, misses) |
| `Fenwick`      | Unblocked shot attempt (goals + shots + misses)                |
| `Shot`         | Shot on goal indicator                                         |
| `Goal`         | Goal scored indicator                                          |
| `ShotDistance` | Distance from net (feet)                                       |
| `ShotAngle`    | Angle of shot relative to goal                                 |
| `xG_F`         | Expected goals for (probability of scoring)                    |

### Results Data (`results.csv`)

Game outcome labels:

| Column    | Description                          |
| --------- | ------------------------------------ |
| `Game Id` | Unique game identifier               |
| `Season`  | NHL season                           |
| `Date`    | Game date                            |
| `Ev_Team` | Team name                            |
| `Is_Home` | Whether team is home (1) or away (0) |
| `Goal`    | Goals scored by team                 |
| `Win`     | Whether team won (1 or 0)            |
| `Points`  | Points earned (0, 1, or 2)           |
| `xG`      | Expected goals                       |

## Feature Engineering

### Per-Game Aggregations

Raw events are aggregated per game per team:

| Feature               | Calculation                                  |
| --------------------- | -------------------------------------------- |
| `game_corsi`          | SUM(Corsi) - total shot attempts             |
| `game_fenwick`        | SUM(Fenwick) - total unblocked shot attempts |
| `game_shots`          | SUM(Shot) - total shots on goal              |
| `game_xg`             | SUM(xG_F) - total expected goals             |
| `game_avg_shot_dist`  | AVG(ShotDistance) - average shot distance    |
| `game_avg_shot_angle` | AVG(ShotAngle) - average shot angle          |

### Rolling Historical Features

For each game, features are computed from the team's **previous games within the current season only** (no cross-season leakage, no data from future games):

| Feature               | Description                             |
| --------------------- | --------------------------------------- |
| `hist_goals_avg`      | Season-to-date average goals per game   |
| `hist_win_pct`        | Season-to-date win percentage           |
| `hist_points_avg`     | Season-to-date average points per game  |
| `hist_corsi_avg`      | Season-to-date average Corsi per game   |
| `hist_fenwick_avg`    | Season-to-date average Fenwick per game |
| `hist_shots_avg`      | Season-to-date average shots per game   |
| `hist_xg_avg`         | Season-to-date average expected goals   |
| `hist_shot_dist_avg`  | Season-to-date average shot distance    |
| `hist_shot_angle_avg` | Season-to-date average shot angle       |
| `recent_win_pct`      | Win percentage in last 5 games          |
| `recent_goals_avg`    | Average goals in last 5 games           |

For early-season games with no history, league average defaults are used.

### Matchup Features

Each prediction sample combines home and away team features:

**Home team features (10):**

- `home_goals_avg`, `home_win_pct`, `home_points_avg`, `home_corsi_avg`, `home_fenwick_avg`, `home_shots_avg`, `home_xg_avg`, `home_recent_form`, `home_recent_goals`, `home_games_played`

**Away team features (10):**

- `away_goals_avg`, `away_win_pct`, `away_points_avg`, `away_corsi_avg`, `away_fenwick_avg`, `away_shots_avg`, `away_xg_avg`, `away_recent_form`, `away_recent_goals`, `away_games_played`

**Differential features (5):**

- `win_pct_diff` = home_win_pct - away_win_pct
- `goals_avg_diff` = home_goals_avg - away_goals_avg
- `xg_diff` = home_xg_avg - away_xg_avg
- `corsi_diff` = home_corsi_avg - away_corsi_avg
- `recent_form_diff` = home_recent_form - away_recent_form

**Total: 25 features**

## Models

| Model                  | Configuration                                             |
| ---------------------- | --------------------------------------------------------- |
| Random Forest          | 200 trees, max depth 10, seed 42                          |
| Logistic Regression    | Binomial, maxIter 100, regParam 0.01, elasticNetParam 0.8 |
| Gradient Boosted Trees | 100 iterations, max depth 8, seed 42                      |
| Multilayer Perceptron  | Layers [25, 64, 32, 2], maxIter 100, blockSize 128        |

## Results

Test set: 2016-2017 season (254 games)

Training set: 9,150 games | Test set distribution: 154 wins (60.6%), 100 not wins (39.4%)

### Baseline Comparisons

| Baseline        | Accuracy | Description                              |
| --------------- | -------- | ---------------------------------------- |
| Majority Class  | 0.6063   | Always predict "Win"                     |
| Weighted Random | 0.5226   | Random guess matching class distribution |
| Coin Flip       | 0.5000   | Random 50/50 guess                       |

### Model Performance

| Model                  | Accuracy | AUC    | Precision | Recall | F1     |
| ---------------------- | -------- | ------ | --------- | ------ | ------ |
| Multilayer Perceptron  | 0.5709   | 0.5378 | 0.6380    | 0.6753 | 0.6562 |
| Logistic Regression    | 0.5512   | 0.4925 | 0.6020    | 0.7662 | 0.6743 |
| Random Forest          | 0.5394   | 0.5140 | 0.6108    | 0.6623 | 0.6355 |
| Gradient Boosted Trees | 0.5315   | 0.5528 | 0.6241    | 0.5714 | 0.5966 |

### Confusion Matrices

**Random Forest:**

|                  | Predicted: Not Win | Predicted: Win |
| ---------------- | ------------------ | -------------- |
| Actual: Not Win  | 35                 | 65             |
| Actual: Win      | 52                 | 102            |

**Logistic Regression:**

|                  | Predicted: Not Win | Predicted: Win |
| ---------------- | ------------------ | -------------- |
| Actual: Not Win  | 22                 | 78             |
| Actual: Win      | 36                 | 118            |

**Gradient Boosted Trees:**

|                  | Predicted: Not Win | Predicted: Win |
| ---------------- | ------------------ | -------------- |
| Actual: Not Win  | 47                 | 53             |
| Actual: Win      | 66                 | 88             |

**Multilayer Perceptron:**

|                  | Predicted: Not Win | Predicted: Win |
| ---------------- | ------------------ | -------------- |
| Actual: Not Win  | 41                 | 59             |
| Actual: Win      | 50                 | 104            |

### Analysis

All models perform below the majority class baseline (60.6%), which highlights the difficulty of predicting NHL games.

- **High randomness**: Hockey has significant game-to-game variance due to factors that are difficult to capture in historical statistics.
- **Competitive balance**: NHL teams are relatively evenly matched , making outcomes harder to predict.
- **Limited signal**: The features based on season-to-date averages may not capture short-term factors like player injuries, back-to-back games, or travel fatigue.

The **Multilayer Perceptron** achieved the highest accuracy (57.09%), suggesting that non-linear relationships between features may provide some predictive value. However, no model substantially outperforms random guessing, indicating that pre-game statistics alone are insufficient for reliable game prediction. However, sports betting models that consistently achieve 52.4% can be profitable over time, thus even small improvements over random chance can be valuable in practice.

### Feature Importance

**Random Forest:**

| Feature             | Importance |
| ------------------- | ---------- |
| `corsi_diff`        | 0.0544     |
| `win_pct_diff`      | 0.0513     |
| `home_goals_avg`    | 0.0490     |
| `goals_avg_diff`    | 0.0487     |
| `home_points_avg`   | 0.0445     |
| `away_shots_avg`    | 0.0441     |
| `away_goals_avg`    | 0.0440     |
| `home_shots_avg`    | 0.0436     |
| `away_recent_goals` | 0.0435     |
| `away_win_pct`      | 0.0421     |

**Gradient Boosted Trees:**

| Feature             | Importance |
| ------------------- | ---------- |
| `home_goals_avg`    | 0.0517     |
| `away_shots_avg`    | 0.0504     |
| `home_recent_goals` | 0.0504     |
| `home_shots_avg`    | 0.0490     |
| `away_goals_avg`    | 0.0488     |
| `home_games_played` | 0.0481     |
| `corsi_diff`        | 0.0479     |
| `away_recent_goals` | 0.0453     |
| `goals_avg_diff`    | 0.0452     |
| `home_points_avg`   | 0.0448     |

Both models show relatively flat importance distributions, with no single feature dominating. Key observations:

- **Differential features** (`corsi_diff`, `goals_avg_diff`, `win_pct_diff`) appear in the top features, suggesting that relative team strength matters more than absolute statistics.
- **Corsi** is as a strong predictor, validating its as an advanced hockey metric.
- **Recent form** features (`home_recent_goals`, `away_recent_goals`) rank highly in GBT, indicating that momentum plays a role.
- The relatively even distribution of importance across features suggests that no single statistic is a reliable predictor.

## Usage

```bash
# Run with spark-submit
spark-submit --driver-memory 4g --executor-memory 4g \
  experiment.py \
  --events NHL_EventData.csv \
  --results results.csv

# Run with subset data for testing
spark-submit --driver-memory 4g --executor-memory 4g \
  experiment.py \
  --events NHL_EventData_subset.csv \
  --results results_subset.csv
```

## Project Structure

```
.
├── experiment.py            # Main PySpark ML pipeline
├── NHL_EventData.csv        # Raw event data
├── results.csv              # Game outcomes (labels)
├── NHL_EventData_subset.csv # Subset for testing
├── results_subset.csv       # Subset labels
└── README.md
```
