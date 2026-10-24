# =======================================================================
# Cell Growth Rate Prediction
# ============================================================================
# ============================================================================

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class CellGrowthModel:

    def __init__(self, filepath):
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"Dataset not found: {filepath}\n"
                "Check that dataset_cell_growth_4factor.csv is in the datasets folder."
            )

        self.df = pd.read_csv(filepath)

        if self.df.empty:
            raise ValueError("The dataset CSV is empty.")

        target_matches = [
            column for column in self.df.columns
            if "specific growth rate" in str(column).lower()
        ]

        if len(target_matches) != 1:
            raise ValueError(
                "The dataset must contain exactly one column with "
                "'Specific Growth Rate' in its name. "
                f"Columns found: {list(self.df.columns)}"
            )

        self.target = target_matches[0]
        self.features = [
            column for column in self.df.columns
            if column != self.target
        ]

        numeric_df = self.df[
            [self.target] + self.features
        ].apply(pd.to_numeric, errors="coerce")

        if numeric_df.isna().any().any():
            bad_columns = numeric_df.columns[
                numeric_df.isna().any()
            ].tolist()

            raise ValueError(
                f"Missing or non-numeric values found in columns: {bad_columns}"
            )

        self.df = numeric_df
        self.x = self.df[self.features].to_numpy(dtype=float)
        self.y = self.df[self.target].to_numpy(dtype=float)

        if len(self.y) < 10:
            raise ValueError(
                "At least 10 complete rows are required for this train/test and CV setup."
            )

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x,
            self.y,
            test_size = 1/6,
            random_state = 0
        )

        self.figures_dir = "figures"
        self.tables_dir = "tables"

        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.tables_dir, exist_ok=True)

        self.cv = KFold(
            n_splits=5,
            shuffle=True,
            random_state=0
        )

        self.fig_width = 10 / 2
        self.fig_height = 8 / 2
        self.fontsize_xy_label = 14
        self.fontsize_ticks = 12
        self.fontsize_legend = 8

        self.model_poly = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(include_bias=False),
            LinearRegression()
        )

        self.model_knn = make_pipeline(
            StandardScaler(),
            KNeighborsRegressor()
        )

        self.kwargs_gscv = {
            "cv": self.cv,
            "scoring": "neg_root_mean_squared_error",
            "return_train_score": True,
            "n_jobs": -1,
            "error_score": "raise"
        }

        self.param_grid_poly = {
            "polynomialfeatures__degree": [i for i in range(1, 11)]
        }

        requested_k = [1, 2, 3, 4, 6, 8, 10, 16, 32]

        smallest_cv_train_size = len(self.y_train) - int(
            np.ceil(len(self.y_train) / 5)
        )

        usable_k = [
            k for k in requested_k
            if k <= smallest_cv_train_size
        ]

        if not usable_k:
            usable_k = [1]

        self.param_grid_knn = {
            "kneighborsregressor__n_neighbors": usable_k,
            "kneighborsregressor__weights": ["uniform", "distance"]
        }

        self.results_poly = None
        self.results_knn = None
        self.model_gscv_poly = None
        self.model_gscv_knn = None

        print("Dataset loaded successfully.")
        print(f"Number of samples: {len(self.y)}")
        print(f"Training samples: {len(self.y_train)}")
        print(f"Held-out test samples: {len(self.y_test)}")
        print(f"Input columns: {self.features}")
        print(f"Target column: {self.target}")
        print(f"KNN k values tested: {usable_k}\n")

    def _save_figure(self, fig, filename):
        fig.savefig(
            os.path.join(self.figures_dir, filename),
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(fig)

    # ============================================================================
    #   Hyperparameter Optimization
    # ============================================================================
    # ============================================================================

    def generate_validation_curves(self):

        self.model_gscv_poly = GridSearchCV(
            self.model_poly,
            param_grid=self.param_grid_poly,
            **self.kwargs_gscv
        )


        self.model_gscv_knn = GridSearchCV(
            self.model_knn,
            param_grid=self.param_grid_knn,
            **self.kwargs_gscv
        )

        self.model_gscv_poly.fit(self.x_train, self.y_train)
        self.model_gscv_knn.fit(self.x_train, self.y_train)

        self.results_poly = pd.DataFrame(self.model_gscv_poly.cv_results_)
        self.results_knn = pd.DataFrame(self.model_gscv_knn.cv_results_)


        self.plot_polynomial_validation_curve(
            max_degree=10,
            filename="figure_01_validation_curve_poly_10.png"
        )

        self.plot_polynomial_validation_curve(
            max_degree=5,
            filename="figure_02_validation_curve_poly_5.png"
        )

        self.plot_knn_validation_curve()

        print("Cross-validation complete.\n")

    def plot_polynomial_validation_curve(self, max_degree, filename):
        df = self.results_poly.copy()

        df["degree"] = df[
            "param_polynomialfeatures__degree"
        ].astype(int)

        df = df[
            df["degree"] <= max_degree
        ].sort_values("degree")

        degrees = df["degree"].to_numpy()
        train_rmse = -df["mean_train_score"].to_numpy()
        valid_rmse = -df["mean_test_score"].to_numpy()

        fig, ax = plt.subplots(
            figsize=(self.fig_width, self.fig_height),
            layout="constrained"
        )

        ax.plot(
            degrees,
            train_rmse,
            marker="o",
            label="Training RMSE"
        )

        ax.plot(
            degrees,
            valid_rmse,
            marker="s",
            linestyle="--",
            label="Validation RMSE"
        )

        ax.set_xlabel(
            "Polynomial degree",
            fontsize=self.fontsize_xy_label
        )

        ax.set_ylabel(
            "RMSE [d^-1]",
            fontsize=self.fontsize_xy_label
        )

        ax.set_xticks(degrees)
        ax.tick_params(
            axis="both",
            labelsize=self.fontsize_ticks
        )

        ax.legend(fontsize=self.fontsize_legend)
        ax.set_ylim(bottom=0)

        self._save_figure(fig, filename)

    def plot_knn_validation_curve(self):
        df = self.results_knn.copy()
        df["k"] = df["param_kneighborsregressor__n_neighbors"].astype(int)
        df["weights"] = df["param_kneighborsregressor__weights"].astype(str)
        df = df.sort_values("k")

        fig, ax = plt.subplots(
            figsize=(self.fig_width, self.fig_height),
            layout="constrained"
        )

        for weights in ["uniform", "distance"]:
            subset = df[df["weights"] == weights].sort_values("k")
            k_values = subset["k"].to_numpy()
            train_rmse = -subset["mean_train_score"].to_numpy()
            valid_rmse = -subset["mean_test_score"].to_numpy()

            ax.plot(k_values, train_rmse, marker="o", label=f"Training RMSE ({weights})")
            ax.plot(k_values, valid_rmse, marker="s", linestyle="--", label=f"Validation RMSE ({weights})")

        ax.set_xlabel("Number of neighbors (k)", fontsize=self.fontsize_xy_label)
        ax.set_ylabel("RMSE [d^-1]", fontsize=self.fontsize_xy_label)
        max_k = int(df["k"].max())
        ax.set_xlim(0, max_k + 1)
        ax.set_xticks(np.arange(0, max_k + 1, 5))
        ax.tick_params(axis="both", labelsize=self.fontsize_ticks)
        ax.legend(fontsize=self.fontsize_legend)
        ax.set_ylim(bottom=0)

        self._save_figure(fig, "figure_03_validation_curve_knn.png")

    def generate_rmse_tables(self):
        """Export polynomial and KNN rankings based on validation RMSE."""

        if self.results_poly is None or self.results_knn is None:
            raise RuntimeError(
                "Run generate_validation_curves() before generating RMSE tables."
            )

        df_poly = pd.DataFrame({
            "Degree": self.results_poly[
                "param_polynomialfeatures__degree"
            ].astype(int),

            "Mean Validation RMSE": -self.results_poly[
                "mean_test_score"
            ].astype(float)
        })

        df_poly = df_poly.sort_values(
            "Mean Validation RMSE",
            ascending=True
        ).reset_index(drop=True)

        df_poly.insert(
            0,
            "Rank",
            np.arange(1, len(df_poly) + 1)
        )

        df_poly["Mean Validation RMSE"] = df_poly[
            "Mean Validation RMSE"
        ].round(6)

        df_poly.to_csv(
            os.path.join(
                self.tables_dir,
                "polynomial_rmse_ranking.csv"
            ),
            index=False
        )

        df_knn = pd.DataFrame({
            "Weighting Method": self.results_knn[
                "param_kneighborsregressor__weights"
            ].astype(str),

            "k": self.results_knn[
                "param_kneighborsregressor__n_neighbors"
            ].astype(int),

            "Mean Validation RMSE": -self.results_knn[
                "mean_test_score"
            ].astype(float)
        })

        df_knn = df_knn.sort_values(
            "Mean Validation RMSE",
            ascending=True
        ).reset_index(drop=True)

        df_knn.insert(
            0,
            "Rank",
            np.arange(1, len(df_knn) + 1)
        )

        df_knn["Mean Validation RMSE"] = df_knn[
            "Mean Validation RMSE"
        ].round(6)

        df_knn.to_csv(
            os.path.join(
                self.tables_dir,
                "knn_rmse_ranking.csv"
            ),
            index=False
        )

        print("Polynomial models ranked by mean validation RMSE:")
        print(df_poly.to_string(index=False))

        print("\nKNN models ranked by mean validation RMSE:")
        print(df_knn.to_string(index=False))
        print()

        return df_poly, df_knn

    def evaluate_top_five_test_set(self):
        """Evaluate the five best polynomial and KNN configurations."""

        if self.results_poly is None or self.results_knn is None:
            raise RuntimeError(
                "Run generate_validation_curves() before test-set evaluation."
            )

        results = []

        poly_candidates = []

        for _, row in self.results_poly.iterrows():
            degree = int(row["param_polynomialfeatures__degree"])

            poly_candidates.append({
                "Model": "Polynomial",
                "Degree / k": degree,
                "Mean Validation RMSE": -float(row["mean_test_score"]),
                "Estimator": make_pipeline(
                    StandardScaler(),
                    PolynomialFeatures(
                        degree=degree,
                        include_bias=False
                    ),
                    LinearRegression()
                )
            })

        poly_candidates.sort(
            key=lambda item: item["Mean Validation RMSE"]
        )

        knn_candidates = []

        for _, row in self.results_knn.iterrows():
            k = int(row["param_kneighborsregressor__n_neighbors"])
            weights = str(row["param_kneighborsregressor__weights"])

            knn_candidates.append({
                "Model": f"KNN({weights})",
                "Degree / k": k,
                "Mean Validation RMSE": -float(row["mean_test_score"]),
                "Estimator": make_pipeline(
                    StandardScaler(),
                    KNeighborsRegressor(
                        n_neighbors=k,
                        weights=weights
                    )
                )
            })

        knn_candidates.sort(
            key=lambda item: item["Mean Validation RMSE"]
        )

        top_candidates = (
            poly_candidates[:5] + knn_candidates[:5]
        )

        for candidate in top_candidates:
            estimator = candidate["Estimator"]

            estimator.fit(
                self.x_train,
                self.y_train
            )

            predictions = estimator.predict(self.x_test)

            results.append({
                "Model": candidate["Model"],
                "Degree / k": candidate["Degree / k"],
                "Mean Validation RMSE": candidate["Mean Validation RMSE"],
                "Test RMSE": np.sqrt(
                    mean_squared_error(
                        self.y_test,
                        predictions
                    )
                )
            })

        df_all = pd.DataFrame(results)

        df_all = df_all.sort_values(
            "Test RMSE",
            ascending=True
        ).reset_index(drop=True)

        df_all.insert(
            0,
            "Rank",
            range(1, len(df_all) + 1)
        )

        for column in [
            "Mean Validation RMSE",
            "Test RMSE"
        ]:
            df_all[column] = df_all[column].round(6)

        df_all.to_csv(
            os.path.join(
                self.tables_dir,
                "top_ten_test_results.csv"
            ),
            index=False
        )

        df_poly = df_all[
            df_all["Model"] == "Polynomial"
        ].copy()

        df_knn = df_all[
            df_all["Model"].str.startswith("KNN(")
        ].copy()

        df_poly.to_csv(
            os.path.join(
                self.tables_dir,
                "top_five_polynomial_test_results.csv"
            ),
            index=False
        )

        df_knn.to_csv(
            os.path.join(
                self.tables_dir,
                "top_five_knn_test_results.csv"
            ),
            index=False
        )

        print("\nTop 10 configurations ranked by lowest test RMSE:")
        print(df_all.to_string(index=False))

        print(
            "\nThe five polynomial and five KNN configurations were selected "
            "using cross-validation, then ranked together by test RMSE."
        )

        return df_all


    def analyze_model(self):
        """Fit a selected model and return its coefficients or parameters."""

        method = input(
            "Select model (p = Polynomial, k_d = KNN distance, k_u = KNN uniform): "
        ).strip().lower()

        while method not in ["p", "k_d", "k_u"]:
            method = input("Invalid choice. Enter p, k_d, or k_u: ").strip().lower()

        if method == "p":
            degree = int(input("Enter polynomial degree: "))

            model = make_pipeline(
                StandardScaler(),
                PolynomialFeatures(degree=degree, include_bias=False),
                LinearRegression()
            )

        else:
            k = int(input("Enter number of neighbors (k): "))
            weights = "distance" if method == "k_d" else "uniform"

            model = make_pipeline(
                StandardScaler(),
                KNeighborsRegressor(n_neighbors=k, weights=weights)
            )

        model.fit(self.x_train, self.y_train)

        if method == "p":
            polynomial = model["polynomialfeatures"]
            regression = model["linearregression"]

            coefficients = regression.coef_
            powers = polynomial.powers_
            intercept = regression.intercept_

            print("\nPolynomial Regression Equation")
            print(f"Intercept: {intercept:.6f}")
            print("\nCoefficients:")

            for coefficient, power in zip(coefficients, powers):
                terms = []

                for feature, exponent in zip(self.features, power):
                    feature_name = feature.split(" [")[0]

                    if exponent == 1:
                        terms.append(f"{feature_name}")
                    elif exponent > 1:
                        terms.append(f"{feature_name}^{exponent}")

                if terms:
                    print(f"{coefficient:+.6f} * {' * '.join(terms)}")

            return {
                "model": model,
                "intercept": intercept,
                "coefficients": coefficients,
                "powers": powers
            }

        print(f"\nKNN model: {method}")
        print(f"Number of neighbors: {k}")
        print(f"Weights: {weights}")
        print("KNN does not have a single intercept or coefficient per variable.")

        return {
            "model": model,
            "k": k,
            "weights": weights
        }

