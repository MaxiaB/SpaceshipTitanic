# Spaceship Titanic Prediction Pipeline

## Overview

This repository contains our end-to-end solution for the Kaggle **Spaceship Titanic** challenge. Our goal is to predict which passengers were **Transported** to an alternate dimension, using only passenger records. Rather than chasing exotic new models, we focused on **process improvements**—structured preprocessing, robust cross-validation, ensembling, and threshold optimization—to boost accuracy from **0.8013 → 0.8052** on the public leaderboard.

## Data

* **Training set**: 8,693 rows
* **Test set**: 4,277 rows (to be predicted)
* Key features:

  * **Cabin** (Deck, Side)
  * **Name**
  * **HomePlanet**, **Destination**
  * **CryoSleep**, **VIP**, **Age**
  * Spending: **RoomService**, **FoodCourt**, **ShoppingMall**, **Spa**, **VRDeck**

## Exploratory Data Analysis (EDA)

We performed a series of visual and statistical checks to understand the data before modeling:

1. **Target Balance**: \~50/50 split between transported and non-transported, ensuring our stratified sampling maintains this ratio.
2. **Missing Value Patterns**: Heatmaps revealed that **CryoSleep** and **VIP** had scattered NaNs, while **Age** and **Cabin** were missing for certain passenger groups—informing our imputation strategy.
3. **Feature Distributions**: Histograms of **Age**, **TotalSpend**, and **GroupSize** highlighted heavy skewness and outliers, motivating log-transformations and grouping flags.
4. **Correlation Matrix**: A quick correlation check showed **TotalSpend** clusters with individual spending channels, and little correlation between **Age** and spending, suggesting we add interaction terms later.
5. **Scatterplots with Transport Rates**: Overlaying transport outcome on **Age vs. TotalSpend** and **GroupSize vs. TotalSpend** helped identify “sweet spots” and borderline cases—guiding our feature flags `Alone` and `ZeroSpender`.

## Preprocessing & Feature Engineering

1. **Data Splitting**: 5-fold **StratifiedKFold** to preserve class balance and generate out-of-fold (OOF) predictions.
2. **Feature Extraction** (`process(df)`):

   * Parse `Cabin` → `Deck`, `Side` → binary `IsPort`.
   * Extract `GroupID`, compute `GroupSize` and `Alone` flag.
   * Compute `TotalSpend` and `ZeroSpender` flag.
   * Fill missing: CryoSleep & VIP by rule-based logic; numerical by median; categorical by mode.
3. **ColumnTransformer** pipeline:

   * **Categorical**: most-frequent imputation → one-hot encoding
   * **Numerical**: median imputation → standard scaling
4. **Data Leakage Prevention**: All transforms are fit inside each CV fold to avoid peeking at validation data.

## Modeling & Ensembling

We trained four base learners through identical pipelines:

* **RandomForestClassifier**
* **XGBClassifier**
* **LGBMClassifier**
* **MLPClassifier**

### Cross-Validation Results (5 folds)

**Mean CV per ModelAccuracy:**

* RF: 0.8023
* XGB: 0.8043
* LGB: 0.8077
* MLP: 0.8033

**Ensemble weights (proportional to mean CV acc):**

| Model                                                       | Weight |
| ----------------------------------------------------------- | ------ |
| RF                                                          | 0.2493 |
| XGB                                                         | 0.2500 |
| LGB                                                         | 0.2510 |
| MLP                                                         | 0.2497 |

## Threshold Optimization

* Swept decision thresholds from 0.30 → 0.70 on OOF predictions.
* **Best threshold:** 0.51 → CV accuracy: **0.8082**
* Final public test score improved from **0.80126 → 0.80523** (+0.5 pp), correct classification for \~20 additional passengers.

## Visual Diagnostics

* **OOF Probability Distribution:** highlights confident predictions at extremes, with a “smear” of borderline cases in \[0.4–0.6], motivating threshold tuning.
* **Threshold Sweep Plot:** confirms optimal cutoff at 0.51.
* **Hyperparameter Parallel Plot:** shows best regions for `colsample_bytree`, `subsample`, `learning_rate`, `n_estimators`, and `max_depth`.

*This README summarizes our exploratory analysis, preprocessing pipeline, modeling approach, and results as presented in our deck, summarized by our dear operational assistant, ChatGPT.*
