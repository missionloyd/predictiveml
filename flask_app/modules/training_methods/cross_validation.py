import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def perform_cross_validation(X_train, y_train, model, n_fold, scaler):
    # Perform cross-validation
    kf = KFold(n_splits=n_fold, shuffle=True)

    results = []  # Store the results for each fold

    for fold_idx, (train_index, test_index) in enumerate(kf.split(X_train)):
        # Get the training and testing data for the current fold
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

        # Create and train the model
        model.fit(X_train_fold, y_train_fold)

        # Predict on the test set
        y_pred_fold = model.predict(X_test_fold)

        # Inverse transform the predictions and actual values
        y_pred_fold = scaler.inverse_transform(y_pred_fold.reshape(-1, 1))
        y_test_fold = scaler.inverse_transform(y_test_fold.reshape(-1, 1))

        # Calculate metrics for the current fold
        rmse_fold = np.sqrt(mean_squared_error(y_test_fold, y_pred_fold))
        mae_fold = mean_absolute_error(y_test_fold, y_pred_fold)
        mape_fold = mean_absolute_percentage_error(y_test_fold, y_pred_fold)

        # Store the results for the current fold
        results.append((model, rmse_fold, mae_fold, mape_fold))

        # Print the results for the current fold
        print(f"Fold {fold_idx+1} - RMSE: {rmse_fold}, MAE: {mae_fold}, MAPE: {mape_fold}")

    # Find the best model based on MAPE
    best_result = min(results, key=lambda x: x[3])
    best_model = best_result[0]
    best_rmse = best_result[1]
    best_mae = best_result[2]
    best_mape = best_result[3]

    return best_model, best_rmse, best_mae, best_mape