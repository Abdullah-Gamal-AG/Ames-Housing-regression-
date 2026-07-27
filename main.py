import pandas
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV

plt.style.use('dark_background')

df = pandas.read_csv('data.csv')

X = df.drop(columns=["Order", "PID", "SalePrice"])
y = np.log1p(df["SalePrice"])
X["MS SubClass"] = X["MS SubClass"].astype(str)

cat_cols = [
    "MS SubClass",
    "MS Zoning",
    "Street",
    "Alley",
    "Lot Shape",
    "Land Contour",
    "Utilities",
    "Lot Config",
    "Land Slope",
    "Neighborhood",
    "Condition 1",
    "Condition 2",
    "Bldg Type",
    "House Style",
    "Roof Style",
    "Roof Matl",
    "Exterior 1st",
    "Exterior 2nd",
    "Mas Vnr Type",
    "Foundation",
    "BsmtFin Type 1",
    "BsmtFin Type 2",
    "Heating",
    "Central Air",
    "Electrical",
    "Functional",
    "Garage Type",
    "Garage Finish",
    "Paved Drive",
    "Pool QC",
    "Fence",
    "Misc Feature",
    "Sale Type",
    "Sale Condition"
]

ordinal_cols = [
    "Overall Qual",
    "Overall Cond",
    "Exter Qual",
    "Exter Cond",
    "Bsmt Qual",
    "Bsmt Cond",
    "Bsmt Exposure",
    "Heating QC",
    "Kitchen Qual",
    "Fireplace Qu",
    "Garage Qual",
    "Garage Cond"
]

num_cols = [
    "Lot Frontage",
    "Lot Area",
    "Year Built",
    "Year Remod/Add",
    "Mas Vnr Area",
    "BsmtFin SF 1",
    "BsmtFin SF 2",
    "Bsmt Unf SF",
    "Total Bsmt SF",
    "1st Flr SF",
    "2nd Flr SF",
    "Low Qual Fin SF",
    "Gr Liv Area",
    "Bsmt Full Bath",
    "Bsmt Half Bath",
    "Full Bath",
    "Half Bath",
    "Bedroom AbvGr",
    "Kitchen AbvGr",
    "TotRms AbvGrd",
    "Fireplaces",
    "Garage Yr Blt",
    "Garage Cars",
    "Garage Area",
    "Wood Deck SF",
    "Open Porch SF",
    "Enclosed Porch",
    "3Ssn Porch",
    "Screen Porch",
    "Pool Area",
    "Misc Val"
]

time_cols = [
    "Mo Sold",
    "Yr Sold"
]

drop_cols = [
    "Order",
    "PID"
]

zero_fill = [
    "Mas Vnr Area",
    "BsmtFin SF 1",
    "BsmtFin SF 2",
    "Bsmt Unf SF",
    "Total Bsmt SF",
    "Bsmt Full Bath",
    "Bsmt Half Bath",
    "Garage Cars",
    "Garage Area",
    "Pool Area",
    "Misc Val"
]

none_fill = [
    "Alley",
    "Mas Vnr Type",
    "Bsmt Qual",
    "Bsmt Cond",
    "Bsmt Exposure",
    "BsmtFin Type 1",
    "BsmtFin Type 2",
    "Fireplace Qu",
    "Garage Type",
    "Garage Finish",
    "Garage Qual",
    "Garage Cond",
    "Pool QC",
    "Fence",
    "Misc Feature"
]

median_fill = [
    "Lot Frontage"
]

mode_fill = [
    "Electrical",
    "MS Zoning",
    "Exterior 1st",
    "Exterior 2nd",
    "Kitchen Qual",
    "Sale Type"
]


def missing_values_info():
    mask = X.notna()

    plt.figure(figsize=(15, 8))
    plt.imshow(mask, aspect='auto', cmap='RdYlGn')  # أخضر = موجود، أحمر = مفقود

    plt.xlabel("Features")
    plt.ylabel("Samples")
    plt.title("Missing Values Visualization")

    plt.colorbar(label="Data Presence")
    plt.show()
    print(X.isna().sum()[X.isna().sum() > 0])


missing_percentage = X.isnull().sum().sum() / X.size * 100
print(missing_percentage)

X[zero_fill] = X[zero_fill].fillna(0)
print(X.isnull().sum()[X.isnull().sum() > 0])
X[none_fill] = X[none_fill].fillna("None")
print(X.isnull().sum()[X.isnull().sum() > 0])
X[median_fill] = X[median_fill].fillna(X[median_fill].median())
print(X.isnull().sum()[X.isnull().sum() > 0])
X[mode_fill] = X[mode_fill].fillna(X[mode_fill].mode().iloc[0])
print(X.isnull().sum()[X.isnull().sum() > 0])
X["Garage Yr Blt"] = X["Garage Yr Blt"].fillna(
    X["Year Built"]
)
print(X.isnull().sum()[X.isnull().sum() > 0])

missing_percentage = X.isnull().sum().sum() / X.size * 100
print(missing_percentage)

X["TotalSF"] = (
    X["Total Bsmt SF"] +
    X["1st Flr SF"] +
    X["2nd Flr SF"]
)

X["TotalBath"] = (
    X["Full Bath"]
    + 0.5 * X["Half Bath"]
    + X["Bsmt Full Bath"]
    + 0.5 * X["Bsmt Half Bath"]
)

X["Age"] = X["Yr Sold"] - X["Year Built"]

X["RemodAge"] = X["Yr Sold"] - X["Year Remod/Add"]

X["TotalGarage"] = (
    X["Garage Cars"] *
    X["Garage Area"]
)





def data_info(x,y):
    print("Features shape:", x.shape)
    print("Target shape:", y.shape)
    print("Features columns:", x.columns.tolist())
    print("Target column:", y.name)

def visualization_data(x, y, num_cols):
    for start in range(0, len(num_cols), 8):
        current_cols = num_cols[start:start + 8]

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for i, col in enumerate(current_cols):
            axes[i].hist(
                x[col].dropna(),
                bins=30,
                edgecolor="white",
                alpha=0.8
            )

            axes[i].set_title(col)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Count")
            axes[i].grid(alpha=0.3)

        for j in range(len(current_cols), 8):
            fig.delaxes(axes[j])

        plt.suptitle(
            f"Numerical Features Distribution ({start+1}-{start+len(current_cols)})"
        )
        plt.tight_layout()
        plt.show()

    # =============================

    for start in range(0, len(num_cols), 8):
        current_cols = num_cols[start:start + 8]

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for i, col in enumerate(current_cols):
            axes[i].boxplot(
                x[col].dropna(),
                vert=True,
                patch_artist=True
            )

            axes[i].set_title(col)
            axes[i].grid(alpha=0.3)

        for j in range(len(current_cols), 8):
            fig.delaxes(axes[j])

        plt.suptitle(
            f"Outliers Detection ({start+1}-{start+len(current_cols)})"
        )
        plt.tight_layout()
        plt.show()

    

def correlation_heatmap(X):

    corr = X.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(18, 14))

    im = ax.imshow(
        corr,
        cmap="coolwarm",
        aspect="auto",
        vmin=-1,
        vmax=1
    )

    fig.colorbar(im, ax=ax, label="Correlation")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(
        corr.columns,
        rotation=90,
        fontsize=7
    )

    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(
        corr.columns,
        fontsize=7
    )

    ax.set_title("Correlation Heatmap", fontsize=16, pad=20)

    fig.subplots_adjust(
        left=0.28,
        bottom=0.22,
        right=0.95,
        top=0.92
    )

    plt.show()

correlation_heatmap(X)


data_info(X, y)
#missing_values_info()
#visualization_data(X, y, num_cols)

missing_percentage = X.isnull().sum().sum() / X.size * 100
print(missing_percentage)
X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


onehot_encoder = OneHotEncoder(drop='first', sparse_output=False,handle_unknown='ignore')

encoded = onehot_encoder.fit_transform(X_train[cat_cols])

encoded_df = pandas.DataFrame(
    encoded,
    columns=onehot_encoder.get_feature_names_out(cat_cols),
    index=X_train.index
)

X_train = pandas.concat(
    [X_train.drop(columns=cat_cols), encoded_df],
    axis=1
)

encoded_test = onehot_encoder.transform(X_test[cat_cols])

encoded_test_df = pandas.DataFrame(
    encoded_test,
    columns=onehot_encoder.get_feature_names_out(cat_cols),
    index=X_test.index
)

X_test = pandas.concat(
    [X_test.drop(columns=cat_cols), encoded_test_df],
    axis=1
)


categories = [
    [1,2,3,4,5,6,7,8,9,10],             # Overall Qual
    [1,2,3,4,5,6,7,8,9,10],             # Overall Cond
    ["Po","Fa","TA","Gd","Ex"],         # Exter Qual
    ["Po","Fa","TA","Gd","Ex"],         # Exter Cond
    ["NA","Po","Fa","TA","Gd","Ex"],    # Bsmt Qual
    ["NA","Po","Fa","TA","Gd","Ex"],    # Bsmt Cond
    ["NA","No","Mn","Av","Gd"],         # Bsmt Exposure
    ["Po","Fa","TA","Gd","Ex"],         # Heating QC
    ["Po","Fa","TA","Gd","Ex"],         # Kitchen Qual
    ["NA","Po","Fa","TA","Gd","Ex"],    # Fireplace Qu
    ["NA","Po","Fa","TA","Gd","Ex"],    # Garage Qual
    ["NA","Po","Fa","TA","Gd","Ex"]     # Garage Cond
]

ordinal_encoder = OrdinalEncoder(
    categories=categories,
    dtype=int,
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

X_train[ordinal_cols] = ordinal_encoder.fit_transform(X_train[ordinal_cols])

X_test[ordinal_cols] = ordinal_encoder.transform(X_test[ordinal_cols])

print("X_train shape:", X_train.shape)

models = {}



random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

score = cross_val_score(random_forest_model, X_train, y_train, cv=5, scoring='r2')

y_pred_rf = random_forest_model.predict(X_test)

y_test_real = np.expm1(y_test)

y_pred_rf = np.expm1(y_pred_rf)

errors = pandas.DataFrame({
    "Actual": y_test_real,
    "Predicted": y_pred_rf
})

errors["Error"] = abs(
    errors["Actual"] - errors["Predicted"]
)

model = {"random_forest_model": {"mse": mean_squared_error(y_test_real, y_pred_rf),"rmse": root_mean_squared_error(y_test_real, y_pred_rf), "mae": mean_absolute_error(y_test_real, y_pred_rf), "r2": r2_score(y_test_real, y_pred_rf),"score": score.mean(), "errors": errors}}
models.update(model)

params = {
    "n_estimators": [300, 500],
    "learning_rate": [0.01, 0.05],
    "max_depth": [2, 3],
    "min_samples_leaf": [5, 10],
    "subsample": [0.8, 1.0]
}

gradient_boosting_model = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    params,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

gradient_boosting_model.fit(X_train, y_train)


y_pred_gb = gradient_boosting_model.predict(X_test)

y_pred_gb = np.expm1(y_pred_gb)

errors = pandas.DataFrame({
    "Actual": y_test_real,
    "Predicted": y_pred_gb
})

errors["Error"] = abs(
    errors["Actual"] - errors["Predicted"]
)

model = {"gradient_boosting_model": {"mse": mean_squared_error(y_test_real, y_pred_gb),"rmse": root_mean_squared_error(y_test_real, y_pred_gb), "mae": mean_absolute_error(y_test_real, y_pred_gb), "r2": r2_score(y_test_real, y_pred_gb),"score": gradient_boosting_model.best_score_, "errors": errors}}
models.update(model)

feature_names = X_train.columns
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred = linear_model.predict(X_test)
y_pred = np.expm1(y_pred)
score = cross_val_score(linear_model, X_train, y_train, cv=5, scoring='r2')
errors = pandas.DataFrame({
    "Actual": y_test_real,
    "Predicted": y_pred
})
errors["Error"] = abs(errors["Actual"] - errors["Predicted"])
model = {"linear_model": {"mse": mean_squared_error(y_test_real, y_pred),"rmse": root_mean_squared_error(y_test_real, y_pred), "mae": mean_absolute_error(y_test_real, y_pred), "r2": r2_score(y_test_real, y_pred),"score": score.mean(), "errors": errors}}
models.update(model)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

y_pred_ridge = ridge_model.predict(X_test)

y_pred_ridge = np.expm1(y_pred_ridge)
score = cross_val_score(ridge_model, X_train, y_train, cv=5, scoring='r2')
errors = pandas.DataFrame({
    "Actual": y_test_real,
    "Predicted": y_pred_ridge
})
errors["Error"] = abs(errors["Actual"] - errors["Predicted"])
model = {"ridge_model": {"mse": mean_squared_error(y_test_real, y_pred_ridge),"rmse": root_mean_squared_error(y_test_real, y_pred_ridge), "mae": mean_absolute_error(y_test_real, y_pred_ridge), "r2": r2_score(y_test_real, y_pred_ridge),"score": score.mean(), "errors": errors}}
models.update(model)

lasso_model = Lasso(alpha=0.0005)
lasso_model.fit(X_train, y_train)

y_pred_lasso = lasso_model.predict(X_test)
y_pred_lasso = np.expm1(y_pred_lasso)

score = cross_val_score(lasso_model, X_train, y_train, cv=5, scoring='r2')
errors = pandas.DataFrame({
    "Actual": y_test_real,
    "Predicted": y_pred_lasso
})
errors["Error"] = abs(errors["Actual"] - errors["Predicted"])
model = {"lasso_model": {"mse": mean_squared_error(y_test_real, y_pred_lasso),"rmse": root_mean_squared_error(y_test_real, y_pred_lasso), "mae": mean_absolute_error(y_test_real, y_pred_lasso), "r2": r2_score(y_test_real, y_pred_lasso),"score": score.mean(), "errors": errors}}
models.update(model)

best_gb = gradient_boosting_model.best_estimator_

importance = pandas.Series(
    best_gb.feature_importances_,
    index=feature_names
)

def visualize_results_info(models):
    for model_name, metrics in models.items():
        print(f"Model: {model_name}")
        print(f"Mean Squared Error (MSE): {metrics['mse']:.4f}")
        print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
        print(f"Mean Absolute Error (MAE): {metrics['mae']:.4f}")
        print(f"R-squared (R2): {metrics['r2']:.4f}")
        print(f"Cross-Validation R2 Score: {metrics['score']:.4f}")
        print(f"Errors DataFrame: {metrics['errors'].head()} rows")
        print("============================================================================")

print(f"Importance of features in Gradient Boosting Model:")
print(importance.sort_values(ascending=False))
visualize_results_info(models)

def visualize_results(models):

    names = [
        "Linear",
        "Ridge",
        "Lasso",
        "Random Forest",
        "Gradient Boosting"
    ]

    keys = list(models.keys())

    r2 = [models[k]["r2"] for k in keys]
    cv = [models[k]["score"] for k in keys]
    mae = [models[k]["mae"] for k in keys]
    rmse = [models[k]["rmse"] for k in keys]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 8),
        constrained_layout=True
    )

    plots = [
        ("R² Score", r2),
        ("Cross Validation", cv),
        ("MAE", mae),
        ("RMSE", rmse)
    ]

    for ax, (title, values) in zip(axes.flat, plots):

        bars = ax.bar(names, values, width=0.6)

        ax.set_title(title, fontsize=13)

        ax.tick_params(
            axis="x",
            labelrotation=15,
            labelsize=9
        )

        ax.tick_params(
            axis="y",
            labelsize=9
        )

        ax.margins(x=0.1)

        for bar in bars:
            height = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width()/2,
                height,
                f"{height:.3f}" if height < 100 else f"{height:.0f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    plt.show()


def plot_predictions(models):

    names = {
        "linear_model": "Linear",
        "ridge_model": "Ridge",
        "lasso_model": "Lasso",
        "random_forest_model": "RF",
        "gradient_boosting_model": "GB"
    }

    n = len(models)
    rows = math.ceil(n / 2)

    fig, axes = plt.subplots(
        rows,
        2,
        figsize=(12, 4 * rows),
        constrained_layout=True
    )

    axes = np.array(axes).flatten()

    for ax, (key, metrics) in zip(axes, models.items()):

        errors = metrics["errors"]

        ax.scatter(
            errors["Actual"],
            errors["Predicted"],
            s=15,
            alpha=0.6
        )

        mn = min(errors["Actual"].min(), errors["Predicted"].min())
        mx = max(errors["Actual"].max(), errors["Predicted"].max())

        ax.plot(
            [mn, mx],
            [mn, mx],
            "r--",
            linewidth=2,
            label="Perfect Prediction"
        )

        ax.set_title(names[key], fontsize=12)

        ax.set_xlabel("Actual", fontsize=10)
        ax.set_ylabel("Predicted", fontsize=10)

        ax.tick_params(axis="both", labelsize=8)

        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        ax.margins(x=0.05, y=0.05)

    for ax in axes[n:]:
        fig.delaxes(ax)

    plt.show()

def plot_feature_importance(importance, top_n=20):

    importance = importance.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10, 8))

    plt.barh(
        importance.index[::-1],
        importance.values[::-1]
    )

    plt.xlabel("Importance")
    plt.ylabel("Features")

    plt.title(f"Top {top_n} Feature Importance (Gradient Boosting)")

    plt.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()


plot_feature_importance(importance, top_n=20)
visualize_results(models)
plot_predictions(models)
