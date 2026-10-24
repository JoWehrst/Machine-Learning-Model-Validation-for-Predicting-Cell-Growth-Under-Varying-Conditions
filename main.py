
# ============================================================================
# Cell Growth Rate Prediction - Main Program
# ============================================================================
# This program runs the cell growth prediction analysis.
#
# It:
#   1. Loads the cell growth dataset.
#   2. Generates validation curves for polynomial and KNN regression.
#   3. Exports model rankings based on validation RMSE.
#   4. Evaluates the five best configurations from each model family.
#
# Graphs are saved in the figures folder.
# Tables are saved in the tables folder.
# ============================================================================

import os

from src.classes import CellGrowthModel

filepath_dataset = os.path.join("datasets", "dataset_cell_growth_4factor.csv")


cg_model = CellGrowthModel(filepath=filepath_dataset)
cg_model.generate_validation_curves()
cg_model.generate_rmse_tables()
cg_model.evaluate_top_five_test_set()
cg_model.analyze_model()
