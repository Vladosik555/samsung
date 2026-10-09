import argparse
import numpy as np
import pandas as pd

#Это
def sigmoid(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_loss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def to_one_hot(y, num_classes=2):
    y = y.astype(int).ravel()
    return np.eye(num_classes)[y]


class Predictor:

    def __init__(self, model_file):
        data = np.load(model_file, allow_pickle=True).item()
        self.layer_sizes = data["layer_sizes"]
        self.weights = data["weights"]
        self.biases = data["biases"]

    def predict_proba(self, X):
        a = X
        num_layers = len(self.weights)
        for l in range(num_layers):
            z = np.dot(a, self.weights[l]) + self.biases[l]
            a = softmax(z) if l == num_layers - 1 else sigmoid(z)
        return a

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)


def main():
    parser = argparse.ArgumentParser(description="Predict using trained MLP")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data_validation.csv",
        help="Dataset to predict",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="saved_model.npy",
        help="Path to saved model (.npy)",
    )
    args = parser.parse_args()

    predictor = Predictor(args.model)
    df = pd.read_csv(args.dataset)

    has_target = "target" in df.columns
    if has_target:
        X = df.drop("target", axis=1).to_numpy()
        y_true = to_one_hot(df["target"].to_numpy())
    else:
        X = df.to_numpy()

    probas = predictor.predict_proba(X)
    preds = np.argmax(probas, axis=1)

    print(f"--- Результаты предсказания ({len(preds)} объектов) ---")
    print(f"Первые 5 вероятностей [P(Benign), P(Malignant)]:\n{probas[:5]}")
    print(
        f"Первые 5 предсказанных классов (0 - Benign, 1 - Malignant): {preds[:5]}"
    )

    if has_target:
        loss = cross_entropy_loss(y_true, probas)
        acc = np.mean(preds == np.argmax(y_true, axis=1))
        print("\n" + "=" * 40)
        print(f"Оценка на тесте:")
        print(f"Binary Cross-Entropy Loss: {loss:.4f}")
        print(f"Accuracy:                  {acc * 100:.2f}%")
        print("=" * 40)


if __name__ == "__main__":
    main()