import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def train_test_split_norm(path_data, drop_cols=[], train_ratio=0.8, target_col="target", train_out="data_training.csv", val_out="data_validation.csv"):
    np.random.seed(42)

    data = pd.read_csv(path_data, sep=";", index_col=0)

    if drop_cols:
        data = data.drop(drop_cols, axis=1)

    count_obj = data.shape[0]
    indx = np.random.permutation(count_obj)

    count_train = int(count_obj * train_ratio)

    train_index = indx[:count_train]
    test_index = indx[count_train:]

    X_train = data.drop(target_col, axis=1).iloc[train_index]
    y_train = data[target_col].iloc[train_index]

    X_test = data.drop(target_col, axis=1).iloc[test_index]
    y_test = data[target_col].iloc[test_index]

    train_max, train_min = X_train.max(axis=0), X_train.min(axis=0)

    diff = np.where((train_max - train_min) == 0, 10 ** (-8), train_max - train_min)

    # Нормализация
    X_train_scaled = (X_train - train_min) / diff
    X_test_scaled = (X_test - train_min) / diff

    y_train_num = y_train.apply(lambda x: 1 if x == "M" else 0)
    y_test_num = y_test.apply(lambda x: 1 if x == "M" else 0)

    df_train_to_save = X_train_scaled.copy()
    df_train_to_save[target_col] = y_train_num
    df_train_to_save.to_csv(train_out, index=False)

    df_val_to_save = X_test_scaled.copy()
    df_val_to_save[target_col] = y_test_num
    df_val_to_save.to_csv(val_out, index=False)

    print(f"Файлы успешно сохранены:")
    print(f"  -> {train_out} (размер: {X_train_scaled.shape})")
    print(f"  -> {val_out} (размер: {X_test_scaled.shape})")

    return X_train_scaled.to_numpy(), X_test_scaled.to_numpy(), y_train_num.to_numpy().reshape(-1, 1), y_test_num.to_numpy().reshape(-1, 1)


if __name__ == "__main__":
    path = "mlp_preproc_data.csv"
    train_test_split_norm(path_data=path, target_col="target")