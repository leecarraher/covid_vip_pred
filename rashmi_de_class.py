import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split # For supervised example
from sklearn.metrics import  confusion_matrix


N_LAGS = 21

# --- Custom Transformers for Time Series Features ---

class LagFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts lag features from a time series (single column).
    Fills initial NaNs resulting from lags using forward fill then backfill.
    """
    def __init__(self, n_lags=14):
        self.n_lags = n_lags

    def fit(self, X, y=None):
        # No fitting necessary for lags
        return self

    def transform(self, X, y=None):
        # Ensure X is a pandas Series or DataFrame for easy lagging
        if not isinstance(X, (pd.Series, pd.DataFrame)):
            X = pd.DataFrame(X)

        df_lags = pd.DataFrame(index=X.index)
        for i in range(1, self.n_lags + 1):
            df_lags[f'lag_{i}'] = X.shift(i)

        # Handle NaNs created by shift (important!)
        df_lags = df_lags.ffill().bfill()
        return df_lags.values # Return numpy array for sklearn compatibility

class RollingFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts rolling window features (mean, std) from a time series (single column).
    Fills initial NaNs resulting from rolling calculations using forward fill then backfill.
    """
    def __init__(self, window=21):
        self.window = window

    def fit(self, X, y=None):
        # No fitting necessary for rolling stats
        return self

    def transform(self, X, y=None):
         # Ensure X is a pandas Series or DataFrame for easy rolling
        if not isinstance(X, (pd.Series, pd.DataFrame)):
            X = pd.DataFrame(X)
        # Ensure minimum periods matches window to avoid leading NaNs in std dev if possible,
        # but we still need to handle the initial window period.
        # Using min_periods=1 avoids some NaNs but might give less stable initial stats.
        # Let's stick to default min_periods (window size) and handle NaNs after.
        df_rolling = pd.DataFrame(index=X.index)
        series = X.iloc[:, 0] # Assume single column input for rolling
        df_rolling[f'rolling_mean_{self.window}'] = series.rolling(window=self.window, closed='left').mean()
        df_rolling[f'rolling_std_{self.window}'] = series.rolling(window=self.window, closed='left').std()

        # Handle NaNs created by rolling window (important!)
        df_rolling = df_rolling.ffill().bfill()

        return df_rolling.values # Return numpy array

class RateOfChangeExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts the rate of change (difference) from the previous period.
    """
    def __init__(self):
        pass # No parameters needed

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not isinstance(X, (pd.Series, pd.DataFrame)):
            X = pd.DataFrame(X)
        series = X.iloc[:, 0]
        df_roc = pd.DataFrame(index=X.index)
        df_roc['roc_1'] = series.diff()

        # Handle the first NaN
        df_roc = df_roc.ffill().bfill()
        return df_roc.values

# Helper to pass through the original value if needed in FeatureUnion
def passthrough_func(X):
    # Sklearn expects 2D array
    if isinstance(X, pd.Series):
        return X.values.reshape(-1, 1)
    if isinstance(X, pd.DataFrame):
         return X.values
    if isinstance(X, np.ndarray) and X.ndim == 1:
        return X.reshape(-1,1)
    return X

PassthroughTransformer = FunctionTransformer(passthrough_func, validate=False)

# --- Pipeline Definition ---

# Define the feature engineering part using FeatureUnion
# Each branch of the union processes the *original* input data
feature_engineering = FeatureUnion([
    ('original_value', PassthroughTransformer), # Include the original value itself
    ('lags', LagFeatureExtractor(n_lags=N_LAGS)),  # Extract lag features (e.g., previous 5 days)
    ('rolling_stats', RollingFeatureExtractor(window=6)), # Use a 14-day rolling window
    ('rate_of_change', RateOfChangeExtractor()) # Include rate of change
    # Add more feature extractors here if needed (e.g., seasonality, Fourier)
])

# --- Option 2: Supervised Classification Pipeline (Requires Labeling) ---

# Example Labeling Strategy (apply *before* the pipeline)
def label_hotspots(series, window=7, std_dev_threshold=2.5):
    """Labels points significantly above a rolling mean + std dev."""
    rolling_mean = series.rolling(window=window, closed='left').mean()
    rolling_std = series.rolling(window=window, closed='left').std()
    threshold = rolling_mean + std_dev_threshold * rolling_std
    # Handle initial NaNs in threshold
    threshold = threshold.bfill() # Backfill to cover the start
    anomalies = (series > threshold).astype(int) # 1 if anomaly, 0 if normal
    return anomalies


features = ["reproduction_rate", "icu_patients",  "hosp_patients",  "weekly_icu_admis", "total_vaccinations", "stringency_index", "population", "population_density", "median_age", "aged_65", \
            "aged_70", "gdp_per_capita", "extreme_poverty", "cardiovasc_death_rate", "diabetes_prevalence", "female_smokers", "male_smokers", "handwashing_facilities", "hospital_beds_pths", "life_expectancy", \
            "human_development_index"]

def create_ts_for_location(df,loc):
    """
    Extracts differencing of total cases, generally the most stable available metric in the data
    """
    tmpdf = df[df.location==loc]
    x = tmpdf.total_cases.values
    diff_x = x[1:]-x[:-1]
    extra_features = (tmpdf.iloc[:-1])[features].ffill()
    extra_features = extra_features.fillna(0).values
    #standardize data
    extra_features = (extra_features-extra_features.mean(axis=0))/(extra_features.std(axis=0)+1e-6)

    return diff_x,extra_features

def predict_for_location(df,loc,inspect=False):

    data,extra_features = create_ts_for_location(df,loc)
    print(loc)
    covid_series = pd.Series(data, name='infected_rate')

    # Important: Reshape data for sklearn pipeline (needs 2D: [n_samples, n_features])
    # Here, n_samples = n_timepoints, n_features = 1 (the infection rate)
    X_input = covid_series.values.reshape(-1, 1)


    # 3. Apply Supervised Pipeline (Requires labeling first)
    # Generate labels based on our rule
    y_labels = label_hotspots(covid_series, window=13, std_dev_threshold=3)

    # Generate features separately first to allow splitting X and y consistently
    feature_pipeline = Pipeline([('features', feature_engineering)])
    X_features = feature_pipeline.fit_transform(X_input)
    y_labels_array = y_labels.values

    # Ensure X_features and y_labels_array have same length (handle potential transformer issues)
    min_len = min(len(X_features), len(y_labels_array))
    X_features = X_features[:min_len]
    y_labels_array = y_labels_array[:min_len]
    time_index_aligned = covid_series.index[:min_len] # Align index too


    # Simple split (NOT ideal for time series, use TimeSeriesSplit usually)
    X_train, X_test, y_train, y_test, index_train, index_test = train_test_split(
        np.concatenate([ X_features,extra_features],axis=1), y_labels_array, time_index_aligned, test_size=0.3, shuffle=False# shuffle=False is crucial for TS
    )
    X_train = np.nan_to_num(X_train)


    # Build the rest of the supervised pipeline (Scaler + Classifier)
    classifier_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(n_estimators=200))
    ])

    # Train the classifier pipeline on the *engineered features* and labels
    classifier_pipeline.fit(X_train, y_train)

    # Predict on the test set features
    y_pred_rf = classifier_pipeline.predict(X_test)
    is_anomaly_rf_test = y_pred_rf == 1

    if inspect:
        print(f"Defined rule labelled {y_labels.sum()} anomalies overall.")
        print(f"Random Forest predicted {sum(is_anomaly_rf_test)} anomalies on the test set.")
        # You would typically evaluate RF performance using metrics like precision, recall, F1-score on the test set.
        tn, fp, fn, tp = confusion_matrix(is_anomaly_rf_test,y_pred_rf).ravel()
        print({'true_negative':tn,'false_positve':fp,'false_negative':fn,'true_positive':tp})


        # Visualize Supervised Results (on Test Set)
        plt.figure(figsize=(15, 4))

        plt.plot(index_test, savgol_filter(covid_series.loc[index_test].values,7,3), label='Infected Rate (Test Set)')
        plt.scatter(index_test[is_anomaly_rf_test], covid_series.loc[index_test].values[is_anomaly_rf_test], color='red', label='Predicted Anomaly (RF)', s=50)
        plt.scatter(index_test[y_test == 1], covid_series.loc[index_test].values[y_test == 1], color='green', marker='x', s=60, label='True Anomaly (Rule-Based)')
        plt.title(f'Gradient Boosted Classifier Anomalies - {loc}')
        plt.legend()
        plt.grid(True)
        plt.show()
    return classifier_pipeline

#plot the US predictions
classifier = predict_for_location(df,"United States",inspect=True)

# run through all locations and store VIPs
df = pd.read_csv("OWID_COVID19_data_4_Project5-1 (1).csv")
locations = df.location.unique()
classifier = predict_for_location(df,locations[0])
vips = classifier['classifier'].feature_importances_
for loc in locations[1:]:
    try:
        classifier = predict_for_location(df,loc)
        vips+=classifier['classifier'].feature_importances_
    except:
        print("error")
    

vipdf = pd.DataFrame()
vipdf["Weights"] = vips
vipdf["Features"] = ["total_cases"]+[f"lag_{1+i}" for i in range(N_LAGS)] + ["rolling_mean", "rolling_std", "rate_of_change"] +features
vipdf.plot(x='Features', y="Weights", kind='bar')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
