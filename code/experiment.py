import argparse
from itertools import chain

from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    GBTClassifier,
    LogisticRegression,
    MultilayerPerceptronClassifier,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
)
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    avg as spark_avg,
)
from pyspark.sql.functions import (
    coalesce,
    col,
    lit,
    regexp_replace,
    row_number,
    to_date,
    trim,
    upper,
    when,
)
from pyspark.sql.functions import (
    max as spark_max,
)
from pyspark.sql.functions import (
    sum as spark_sum,
)
from pyspark.sql.types import DoubleType, IntegerType, LongType


def print_feature_importance(model, feature_names, model_name):
    """Print feature importances for tree-based models."""
    try:
        # Get the actual classifier from the pipeline
        classifier = model.stages[-1]
        if hasattr(classifier, "featureImportances"):
            importances = classifier.featureImportances.toArray()
            feat_imp = list(zip(feature_names, importances))
            feat_imp.sort(key=lambda x: x[1], reverse=True)

            print(f"\n{model_name} Feature Importances:")
            print("-" * 40)
            for fname, imp in feat_imp[:10]:  # Top 10
                print(f"  {fname:<30} {imp:.4f}")
    except Exception as e:
        print(f"Could not extract feature importances: {e}")


# ---------------- TEAM NAME MAPPING -----------------------
TEAM_MAP = {
    # ANAHEIM DUCKS
    "Anaheim": "ANA",
    "Anaheim Ducks": "ANA",
    "Mighty Ducks of Anaheim": "ANA",
    "Mighty Ducks": "ANA",
    "ANA": "ANA",
    # ARIZONA / PHOENIX COYOTES
    "Arizona": "ARI",
    "Arizona Coyotes": "ARI",
    "Coyotes": "ARI",
    "Phoenix": "ARI",
    "Phoenix Coyotes": "ARI",
    "PHX": "ARI",
    "ARI": "ARI",
    # BOSTON BRUINS
    "Boston": "BOS",
    "Boston Bruins": "BOS",
    "BOS": "BOS",
    # BUFFALO SABRES
    "Buffalo": "BUF",
    "Buffalo Sabres": "BUF",
    "BUF": "BUF",
    # CAROLINA HURRICANES
    "Carolina": "CAR",
    "Carolina Hurricanes": "CAR",
    "Hurricanes": "CAR",
    "Hartford": "CAR",
    "Hartford Whalers": "CAR",
    "CAR": "CAR",
    # COLUMBUS BLUE JACKETS
    "Columbus": "CBJ",
    "Columbus Blue Jackets": "CBJ",
    "CBJ": "CBJ",
    # CALGARY FLAMES
    "Calgary": "CGY",
    "Calgary Flames": "CGY",
    "CGY": "CGY",
    # CHICAGO BLACKHAWKS
    "Chicago": "CHI",
    "Chicago Blackhawks": "CHI",
    "Blackhawks": "CHI",
    "CHI": "CHI",
    # COLORADO AVALANCHE
    "Colorado": "COL",
    "Colorado Avalanche": "COL",
    "Avalanche": "COL",
    "Quebec": "COL",
    "Quebec Nordiques": "COL",
    "COL": "COL",
    # DALLAS STARS
    "Dallas": "DAL",
    "Dallas Stars": "DAL",
    "Stars": "DAL",
    "Minnesota North Stars": "DAL",
    "DAL": "DAL",
    # DETROIT RED WINGS
    "Detroit": "DET",
    "Detroit Red Wings": "DET",
    "Red Wings": "DET",
    "DET": "DET",
    # EDMONTON OILERS
    "Edmonton": "EDM",
    "Edmonton Oilers": "EDM",
    "Oilers": "EDM",
    "EDM": "EDM",
    # FLORIDA PANTHERS
    "Florida": "FLA",
    "Florida Panthers": "FLA",
    "Panthers": "FLA",
    "FLA": "FLA",
    # LOS ANGELES KINGS
    "L.A.": "LAK",
    "LA": "LAK",
    "L.A": "LAK",
    "Los Angeles": "LAK",
    "Los Angeles Kings": "LAK",
    "Kings": "LAK",
    "LAK": "LAK",
    # MINNESOTA WILD
    "Minnesota": "MIN",
    "Minnesota Wild": "MIN",
    "Wild": "MIN",
    "MIN": "MIN",
    # MONTREAL CANADIENS
    "Montréal": "MTL",
    "Montreal": "MTL",
    "Montreal Canadiens": "MTL",
    "Canadiens": "MTL",
    "MTL": "MTL",
    # NASHVILLE PREDATORS
    "Nashville": "NSH",
    "Nashville Predators": "NSH",
    "Predators": "NSH",
    "NSH": "NSH",
    # NEW JERSEY DEVILS
    "N.J.": "NJD",
    "N.J": "NJD",
    "NJ": "NJD",
    "New Jersey": "NJD",
    "New Jersey Devils": "NJD",
    "Devils": "NJD",
    "NJD": "NJD",
    # NEW YORK ISLANDERS
    "N.Y. I": "NYI",
    "N.Y. Islanders": "NYI",
    "NY Islanders": "NYI",
    "New York Islanders": "NYI",
    "Islanders": "NYI",
    "NYI": "NYI",
    # NEW YORK RANGERS
    "N.Y. R": "NYR",
    "N.Y. Rangers": "NYR",
    "NY Rangers": "NYR",
    "New York Rangers": "NYR",
    "Rangers": "NYR",
    "NYR": "NYR",
    # OTTAWA SENATORS
    "Ottawa": "OTT",
    "Ottawa Senators": "OTT",
    "Senators": "OTT",
    "OTT": "OTT",
    # PHILADELPHIA FLYERS
    "Philadelphia": "PHI",
    "Philadelphia Flyers": "PHI",
    "Flyers": "PHI",
    "PHI": "PHI",
    # PITTSBURGH PENGUINS
    "Pittsburgh": "PIT",
    "Pittsburgh Penguins": "PIT",
    "Penguins": "PIT",
    "PIT": "PIT",
    # SAN JOSE SHARKS
    "S.J.": "SJS",
    "SJ": "SJS",
    "San Jose": "SJS",
    "San Jose Sharks": "SJS",
    "Sharks": "SJS",
    "SJS": "SJS",
    # SEATTLE KRAKEN
    "Seattle": "SEA",
    "Seattle Kraken": "SEA",
    "Kraken": "SEA",
    "SEA": "SEA",
    # ST. LOUIS BLUES
    "St. Louis": "STL",
    "St Louis": "STL",
    "St. Louis Blues": "STL",
    "St Louis Blues": "STL",
    "Blues": "STL",
    "STL": "STL",
    # TAMPA BAY LIGHTNING
    "T.B.": "TBL",
    "TB": "TBL",
    "Tampa Bay": "TBL",
    "Tampa Bay Lightning": "TBL",
    "Lightning": "TBL",
    "TBL": "TBL",
    # TORONTO MAPLE LEAFS
    "Toronto": "TOR",
    "Toronto Maple Leafs": "TOR",
    "Maple Leafs": "TOR",
    "Leafs": "TOR",
    "TOR": "TOR",
    # VANCOUVER CANUCKS
    "Vancouver": "VAN",
    "Vancouver Canucks": "VAN",
    "Canucks": "VAN",
    "VAN": "VAN",
    # VEGAS GOLDEN KNIGHTS
    "Vegas": "VGK",
    "Vegas Golden Knights": "VGK",
    "Golden Knights": "VGK",
    "VGK": "VGK",
    # WINNIPEG JETS / ATLANTA THRASHERS
    "Winnipeg": "WPG",
    "Winnipeg Jets": "WPG",
    "Jets": "WPG",
    "Atlanta": "WPG",
    "Atlanta Thrashers": "WPG",
    "Thrashers": "WPG",
    "ATL": "WPG",
    "WPG": "WPG",
    # WASHINGTON CAPITALS
    "Washington": "WSH",
    "Washington Capitals": "WSH",
    "Capitals": "WSH",
    "WSH": "WSH",
}


def main():
    parser = argparse.ArgumentParser(
        description="NHL Hockey Game Outcome Prediction using Pre-Game Features"
    )
    parser.add_argument(
        "--events", required=True, help="Path to NHL_EventData.csv (local or GCS path)"
    )
    parser.add_argument(
        "--results", required=True, help="Path to results.csv (local or GCS path)"
    )
    args = parser.parse_args()

    events_path = args.events
    results_path = args.results

    spark = (
        SparkSession.builder.appName("HockeyML_PreGame")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "50")
        .getOrCreate()
    )

    from pyspark.sql.functions import create_map

    team_map_expr = create_map([lit(x) for x in chain(*TEAM_MAP.items())])

    # ============================================================
    # 1. Load RESULTS (contains game dates and outcomes)
    # ============================================================
    print("Loading results data...")
    results = (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .csv(results_path)
    )

    results = results.withColumnRenamed("Game Id", "GameID").withColumnRenamed(
        "Ev_Team", "Ev_Team_raw"
    )

    results = (
        results.withColumn("Season", col("Season").cast(IntegerType()))
        .withColumn("GameID", col("GameID").cast(LongType()))
        .withColumn("Points", col("Points").cast(IntegerType()))
        .withColumn("Goal", col("Goal").cast(IntegerType()))
        .withColumn("Win", col("Win").cast(IntegerType()))
        .withColumn("Is_Home", col("Is_Home").cast(IntegerType()))
        .withColumn("xG", col("xG").cast(DoubleType()))
        .withColumn("Date", to_date(col("Date"), "M/d/yyyy"))
    )

    results = results.filter(col("Season") >= 20072008)
    results = results.filter(col("GameID") >= 2007020001)

    # Clean team names
    results = results.withColumn(
        "Ev_Team_clean", trim(regexp_replace(col("Ev_Team_raw"), r"\s+", " "))
    )
    results = results.withColumn(
        "TeamCode", team_map_expr.getItem(col("Ev_Team_clean"))
    )
    results = results.withColumn(
        "TeamCode",
        coalesce(
            col("TeamCode"), upper(regexp_replace(col("Ev_Team_clean"), "[^A-Z]", ""))
        ),
    )

    # ============================================================
    # 2. Load EVENTS and aggregate per game/team
    # ============================================================
    print("Loading and aggregating events data...")
    events = (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .csv(events_path)
    )

    events = (
        events.withColumn("Season", col("Season").cast(IntegerType()))
        .withColumn("GameID", col("GameID").cast(LongType()))
        .withColumn("Corsi", col("Corsi").cast(DoubleType()))
        .withColumn("Fenwick", col("Fenwick").cast(DoubleType()))
        .withColumn("Shot", col("Shot").cast(DoubleType()))
        .withColumn("Goal", col("Goal").cast(DoubleType()))
        .withColumn("ShotDistance", col("ShotDistance").cast(DoubleType()))
        .withColumn("ShotAngle", col("ShotAngle").cast(DoubleType()))
        .withColumn("xG_F", col("xG_F").cast(DoubleType()))
        .withColumn("xG_S", col("xG_S").cast(DoubleType()))
    )

    events = events.filter(col("Season") >= 20072008)
    events = events.filter(col("GameID") >= 2007020001)

    # Clean team names
    events = events.withColumn(
        "EventTeam_clean", trim(regexp_replace(col("EventTeam"), r"\s+", " "))
    )
    events = events.withColumn(
        "TeamCode", team_map_expr.getItem(col("EventTeam_clean"))
    )
    events = events.withColumn(
        "TeamCode",
        coalesce(
            col("TeamCode"), upper(regexp_replace(col("EventTeam_clean"), "[^A-Z]", ""))
        ),
    )

    # Aggregate events per game/team
    agg_events = events.groupBy("GameID", "Season", "TeamCode").agg(
        spark_sum("Corsi").alias("game_corsi"),
        spark_sum("Fenwick").alias("game_fenwick"),
        spark_sum("Shot").alias("game_shots"),
        spark_avg("ShotDistance").alias("game_avg_shot_dist"),
        spark_avg("ShotAngle").alias("game_avg_shot_angle"),
        spark_sum("xG_F").alias("game_xg"),
    )

    # ============================================================
    # 3. Join results with event aggregates
    # ============================================================
    print("Joining results with event aggregates...")
    game_data = (
        results.alias("r")
        .join(
            agg_events.alias("e"),
            (col("r.GameID") == col("e.GameID"))
            & (col("r.Season") == col("e.Season"))
            & (col("r.TeamCode") == col("e.TeamCode")),
            how="inner",
        )
        .select(
            col("r.GameID").alias("GameID"),
            col("r.Season").alias("Season"),
            col("r.Date").alias("Date"),
            col("r.TeamCode").alias("TeamCode"),
            col("r.Is_Home").alias("Is_Home"),
            col("r.Goal").alias("Goals"),
            col("r.Win").alias("Win"),
            col("r.Points").alias("Points"),
            col("r.xG").alias("xG_result"),
            "game_corsi",
            "game_fenwick",
            "game_shots",
            "game_avg_shot_dist",
            "game_avg_shot_angle",
            "game_xg",
        )
    )

    # Filter nulls
    game_data = game_data.filter(col("Points").isNotNull())
    game_data = game_data.filter(col("Date").isNotNull())

    # Cache game_data for reuse
    game_data = game_data.cache()
    print(f"Total game-team records: {game_data.count()}")

    # ============================================================
    # 4. Compute ROLLING historical stats (pre-game features)
    #    Using all PREVIOUS games in the season (excludes current game)
    # ============================================================
    print("Computing rolling historical features...")

    # Window: all previous games for this team in the season, ordered by date
    # rowsBetween: from start of partition to 1 row before current
    team_season_window = (
        Window.partitionBy("TeamCode", "Season")
        .orderBy("Date", "GameID")
        .rowsBetween(Window.unboundedPreceding, -1)
    )

    # Add game number within season for each team
    team_game_num_window = Window.partitionBy("TeamCode", "Season").orderBy(
        "Date", "GameID"
    )

    game_data = game_data.withColumn(
        "team_game_num", row_number().over(team_game_num_window)
    )

    # Compute rolling averages from PREVIOUS games only
    game_data = (
        game_data
        # Rolling averages from past games
        .withColumn("hist_goals_avg", spark_avg("Goals").over(team_season_window))
        .withColumn(
            "hist_win_pct",
            spark_avg(col("Win").cast(DoubleType())).over(team_season_window),
        )
        .withColumn(
            "hist_points_avg",
            spark_avg(col("Points").cast(DoubleType())).over(team_season_window),
        )
        .withColumn("hist_corsi_avg", spark_avg("game_corsi").over(team_season_window))
        .withColumn(
            "hist_fenwick_avg", spark_avg("game_fenwick").over(team_season_window)
        )
        .withColumn("hist_shots_avg", spark_avg("game_shots").over(team_season_window))
        .withColumn("hist_xg_avg", spark_avg("game_xg").over(team_season_window))
        .withColumn(
            "hist_shot_dist_avg",
            spark_avg("game_avg_shot_dist").over(team_season_window),
        )
        .withColumn(
            "hist_shot_angle_avg",
            spark_avg("game_avg_shot_angle").over(team_season_window),
        )
    )

    # Recent form: last 5 games (momentum)
    recent_window = (
        Window.partitionBy("TeamCode", "Season")
        .orderBy("Date", "GameID")
        .rowsBetween(-5, -1)
    )

    game_data = game_data.withColumn(
        "recent_win_pct", spark_avg(col("Win").cast(DoubleType())).over(recent_window)
    )
    game_data = game_data.withColumn(
        "recent_goals_avg", spark_avg("Goals").over(recent_window)
    )

    # Fill nulls for first games of season (no history yet)
    # Use league averages as defaults
    game_data = (
        game_data.withColumn(
            "hist_goals_avg", coalesce(col("hist_goals_avg"), lit(2.8))
        )
        .withColumn("hist_win_pct", coalesce(col("hist_win_pct"), lit(0.5)))
        .withColumn("hist_points_avg", coalesce(col("hist_points_avg"), lit(1.0)))
        .withColumn("hist_corsi_avg", coalesce(col("hist_corsi_avg"), lit(30.0)))
        .withColumn("hist_fenwick_avg", coalesce(col("hist_fenwick_avg"), lit(25.0)))
        .withColumn("hist_shots_avg", coalesce(col("hist_shots_avg"), lit(30.0)))
        .withColumn("hist_xg_avg", coalesce(col("hist_xg_avg"), lit(2.5)))
        .withColumn(
            "hist_shot_dist_avg", coalesce(col("hist_shot_dist_avg"), lit(35.0))
        )
        .withColumn(
            "hist_shot_angle_avg", coalesce(col("hist_shot_angle_avg"), lit(20.0))
        )
        .withColumn("recent_win_pct", coalesce(col("recent_win_pct"), lit(0.5)))
        .withColumn("recent_goals_avg", coalesce(col("recent_goals_avg"), lit(2.8)))
    )

    # ============================================================
    # 5. Create MATCHUP dataset (home team vs away team)
    # ============================================================
    print("Creating matchup dataset...")

    # Split into home and away
    home_teams = game_data.filter(col("Is_Home") == 1).alias("home")
    away_teams = game_data.filter(col("Is_Home") == 0).alias("away")

    # Join home and away for the same game
    matchups = home_teams.join(
        away_teams,
        (col("home.GameID") == col("away.GameID"))
        & (col("home.Season") == col("away.Season")),
        how="inner",
    ).select(
        col("home.GameID").alias("GameID"),
        col("home.Season").alias("Season"),
        col("home.Date").alias("Date"),
        col("home.TeamCode").alias("home_team"),
        col("away.TeamCode").alias("away_team"),
        # Home team historical stats
        col("home.hist_goals_avg").alias("home_goals_avg"),
        col("home.hist_win_pct").alias("home_win_pct"),
        col("home.hist_points_avg").alias("home_points_avg"),
        col("home.hist_corsi_avg").alias("home_corsi_avg"),
        col("home.hist_fenwick_avg").alias("home_fenwick_avg"),
        col("home.hist_shots_avg").alias("home_shots_avg"),
        col("home.hist_xg_avg").alias("home_xg_avg"),
        col("home.recent_win_pct").alias("home_recent_form"),
        col("home.recent_goals_avg").alias("home_recent_goals"),
        col("home.team_game_num").alias("home_games_played"),
        # Away team historical stats
        col("away.hist_goals_avg").alias("away_goals_avg"),
        col("away.hist_win_pct").alias("away_win_pct"),
        col("away.hist_points_avg").alias("away_points_avg"),
        col("away.hist_corsi_avg").alias("away_corsi_avg"),
        col("away.hist_fenwick_avg").alias("away_fenwick_avg"),
        col("away.hist_shots_avg").alias("away_shots_avg"),
        col("away.hist_xg_avg").alias("away_xg_avg"),
        col("away.recent_win_pct").alias("away_recent_form"),
        col("away.recent_goals_avg").alias("away_recent_goals"),
        col("away.team_game_num").alias("away_games_played"),
        # Label: home team points
        col("home.Points").alias("label"),
    )

    # Add differential features (home advantage perspective)
    matchups = (
        matchups.withColumn("win_pct_diff", col("home_win_pct") - col("away_win_pct"))
        .withColumn("goals_avg_diff", col("home_goals_avg") - col("away_goals_avg"))
        .withColumn("xg_diff", col("home_xg_avg") - col("away_xg_avg"))
        .withColumn("corsi_diff", col("home_corsi_avg") - col("away_corsi_avg"))
        .withColumn(
            "recent_form_diff", col("home_recent_form") - col("away_recent_form")
        )
    )

    # Drop any remaining nulls
    matchups = matchups.dropna()

    # Cache matchups for reuse
    matchups = matchups.cache()
    print(f"Total matchups: {matchups.count()}")

    # ============================================================
    # 6. Train/Test Split (temporal: latest season as test)
    # ============================================================
    max_season = matchups.agg(spark_max("Season")).collect()[0][0]

    train_df = matchups.filter(col("Season") < max_season)
    test_df = matchups.filter(col("Season") == max_season)

    # Fallback if not enough data
    if train_df.count() == 0 or test_df.count() == 0:
        train_df, test_df = matchups.randomSplit([0.8, 0.2], seed=42)
        print("Using random split (not enough seasons)")

    print(
        f"\nTrain = {train_df.count()}, Test = {test_df.count()}, Test season = {max_season}"
    )

    # ============================================================
    # 7. Feature Engineering
    # ============================================================
    feature_cols = [
        # Home team stats
        "home_goals_avg",
        "home_win_pct",
        "home_points_avg",
        "home_corsi_avg",
        "home_fenwick_avg",
        "home_shots_avg",
        "home_xg_avg",
        "home_recent_form",
        "home_recent_goals",
        "home_games_played",
        # Away team stats
        "away_goals_avg",
        "away_win_pct",
        "away_points_avg",
        "away_corsi_avg",
        "away_fenwick_avg",
        "away_shots_avg",
        "away_xg_avg",
        "away_recent_form",
        "away_recent_goals",
        "away_games_played",
        # Differentials
        "win_pct_diff",
        "goals_avg_diff",
        "xg_diff",
        "corsi_diff",
        "recent_form_diff",
    ]

    # Cast all to double
    for c in feature_cols:
        train_df = train_df.withColumn(c, col(c).cast(DoubleType()))
        test_df = test_df.withColumn(c, col(c).cast(DoubleType()))

    assembler = VectorAssembler(
        inputCols=feature_cols, outputCol="features", handleInvalid="skip"
    )

    # Dictionary to store results
    all_results = {}

    # ============================================================
    # Convert to BINARY classification for all models
    # Win (2 points) = 1, Not Win (0 or 1 points) = 0
    # ============================================================
    train_df = train_df.withColumn(
        "label_binary", when(col("label") == 2, 1.0).otherwise(0.0)
    )
    test_df = test_df.withColumn(
        "label_binary", when(col("label") == 2, 1.0).otherwise(0.0)
    )

    # Binary evaluator for all models
    binary_evaluator = BinaryClassificationEvaluator(
        labelCol="label_binary", rawPredictionCol="rawPrediction"
    )

    def evaluate_binary_model(predictions, model_name):
        """Evaluate a binary classification model."""
        auc = binary_evaluator.setMetricName("areaUnderROC").evaluate(predictions)

        # Calculate accuracy, precision, recall, F1
        tp = predictions.filter(
            (col("prediction") == 1.0) & (col("label_binary") == 1.0)
        ).count()
        tn = predictions.filter(
            (col("prediction") == 0.0) & (col("label_binary") == 0.0)
        ).count()
        fp = predictions.filter(
            (col("prediction") == 1.0) & (col("label_binary") == 0.0)
        ).count()
        fn = predictions.filter(
            (col("prediction") == 0.0) & (col("label_binary") == 1.0)
        ).count()

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        print(f"\n{'=' * 60}")
        print(f"{model_name} Results (Binary: Win vs Not Win)")
        print(f"{'=' * 60}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        print(f"\nConfusion Matrix:")
        predictions.groupBy("label_binary", "prediction").count().orderBy(
            "label_binary", "prediction"
        ).show()

        return {
            "accuracy": accuracy,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # ============================================================
    # 8a. Random Forest Classifier
    # ============================================================
    print("\n" + "=" * 60)
    print("Training Random Forest Classifier (Binary)...")
    print("=" * 60)

    rf = RandomForestClassifier(
        labelCol="label_binary",
        featuresCol="features",
        numTrees=200,
        maxDepth=10,
        seed=42,
    )

    rf_pipeline = Pipeline(stages=[assembler, rf])
    rf_model = rf_pipeline.fit(train_df)
    rf_preds = rf_model.transform(test_df)

    all_results["Random Forest"] = evaluate_binary_model(rf_preds, "Random Forest")
    print_feature_importance(rf_model, feature_cols, "Random Forest")

    # ============================================================
    # 8b. Logistic Regression
    # ============================================================
    print("\n" + "=" * 60)
    print("Training Logistic Regression (Binary)...")
    print("=" * 60)

    lr = LogisticRegression(
        labelCol="label_binary",
        featuresCol="features",
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0.8,
        family="binomial",
    )

    lr_pipeline = Pipeline(stages=[assembler, lr])
    lr_model = lr_pipeline.fit(train_df)
    lr_preds = lr_model.transform(test_df)

    all_results["Logistic Regression"] = evaluate_binary_model(
        lr_preds, "Logistic Regression"
    )

    # ============================================================
    # 8c. Gradient Boosted Trees
    # ============================================================
    print("\n" + "=" * 60)
    print("Training Gradient Boosted Trees (Binary)...")
    print("=" * 60)

    gbt = GBTClassifier(
        labelCol="label_binary",
        featuresCol="features",
        maxIter=100,
        maxDepth=8,
        seed=42,
    )

    gbt_pipeline = Pipeline(stages=[assembler, gbt])
    gbt_model = gbt_pipeline.fit(train_df)
    gbt_preds = gbt_model.transform(test_df)

    all_results["Gradient Boosted Trees"] = evaluate_binary_model(
        gbt_preds, "Gradient Boosted Trees"
    )
    print_feature_importance(gbt_model, feature_cols, "Gradient Boosted Trees")

    # ============================================================
    # 8d. Multilayer Perceptron
    # ============================================================
    print("\n" + "=" * 60)
    print("Training Multilayer Perceptron (Neural Network)...")
    print("=" * 60)

    # Network architecture: input layer (25 features) -> hidden layers -> output (2 classes)
    layers = [len(feature_cols), 64, 32, 2]

    mlp = MultilayerPerceptronClassifier(
        labelCol="label_binary",
        featuresCol="features",
        layers=layers,
        maxIter=100,
        blockSize=128,
        seed=42,
    )

    mlp_pipeline = Pipeline(stages=[assembler, mlp])
    mlp_model = mlp_pipeline.fit(train_df)
    mlp_preds = mlp_model.transform(test_df)

    all_results["Multilayer Perceptron"] = evaluate_binary_model(
        mlp_preds, "Multilayer Perceptron"
    )

    # ============================================================
    # 9. Baseline Comparisons
    # ============================================================
    print("\n" + "=" * 60)
    print("BASELINE COMPARISONS")
    print("=" * 60)

    # Calculate test set class distribution
    total_test = test_df.count()
    num_wins = test_df.filter(col("label_binary") == 1.0).count()
    num_not_wins = test_df.filter(col("label_binary") == 0.0).count()
    win_rate = num_wins / total_test
    not_win_rate = num_not_wins / total_test

    print(
        f"Test set distribution: {num_wins} wins ({win_rate:.1%}), {num_not_wins} not wins ({not_win_rate:.1%})"
    )
    print()

    # Baseline 1: Always predict majority class (Win)
    majority_class = 1.0 if num_wins >= num_not_wins else 0.0
    majority_accuracy = max(win_rate, not_win_rate)
    print(
        f"Majority Class Baseline (always predict {'Win' if majority_class == 1.0 else 'Not Win'}):"
    )
    print(f"  Accuracy: {majority_accuracy:.4f}")

    # Baseline 2: Coin flip (50/50)
    coin_flip_accuracy = 0.5
    print(f"\nCoin Flip Baseline (random 50/50):")
    print(f"  Accuracy: {coin_flip_accuracy:.4f}")

    # Baseline 3: Weighted random (matches class distribution)
    # Expected accuracy = P(predict win)*P(actual win) + P(predict not win)*P(actual not win)
    weighted_random_accuracy = win_rate * win_rate + not_win_rate * not_win_rate
    print(f"\nWeighted Random Baseline (random guess matching class distribution):")
    print(f"  Accuracy: {weighted_random_accuracy:.4f}")

    # ============================================================
    # 10. Model Comparison Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY (Binary: Win vs Not Win)")
    print("=" * 60)
    print(
        f"{'Model':<25} {'Accuracy':<10} {'AUC':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}"
    )
    print("-" * 75)

    # Add baselines to results for comparison
    print(
        f"{'Majority Class':<25} {majority_accuracy:<10.4f} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}"
    )
    print(
        f"{'Coin Flip':<25} {coin_flip_accuracy:<10.4f} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}"
    )
    print(
        f"{'Weighted Random':<25} {weighted_random_accuracy:<10.4f} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}"
    )
    print("-" * 75)

    for model_name, metrics in all_results.items():
        print(
            f"{model_name:<25} {metrics['accuracy']:<10.4f} {metrics['auc']:<10.4f} "
            f"{metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f}"
        )

    spark.stop()


if __name__ == "__main__":
    main()
