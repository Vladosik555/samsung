import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def sigmoid(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def sigmoid_derivative(a):
    return a * (1.0 - a)


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



class MLP:

    def __init__(self, layer_sizes):
        """layer_sizes: список размеров слоев, например [30, 24, 24, 2]"""
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            din = layer_sizes[i]
            dout = layer_sizes[i + 1]
            w = np.random.randn(din, dout) * np.sqrt(2.0 / din)
            b = np.zeros((1, dout))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        activations = [X]
        num_layers = len(self.weights)

        for l in range(num_layers):
            z = np.dot(activations[-1], self.weights[l]) + self.biases[l]
            if l == num_layers - 1:
                a = softmax(z)  
            else:
                a = sigmoid(z)  
            activations.append(a)

        return activations

    def backward(self, activations, y_true, lr):
        m = y_true.shape[0]
        num_layers = len(self.weights)

        dZ = activations[-1] - y_true

        for l in range(num_layers - 1, -1, -1):
            A_prev = activations[l]

            dW = (1.0 / m) * np.dot(A_prev.T, dZ)
            db = (1.0 / m) * np.sum(dZ, axis=0, keepdims=True)

            if l > 0:
                dA_prev = np.dot(dZ, self.weights[l].T)
                dZ = dA_prev * sigmoid_derivative(A_prev)

            # Обновление весов (Gradient Descent Step)
            self.weights[l] -= lr * dW
            self.biases[l] -= lr * db

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=84,
        batch_size=8,
        lr=0.0314,
    ):
        history = {
            "loss": [],
            "val_loss": [],
            "accuracy": [],
            "val_accuracy": [],
        }
        n_samples = X_train.shape[0]

        print(f"x_train shape : {X_train.shape}")
        print(f"x_valid shape : {X_val.shape}")

        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                xb = X_shuffled[start:end]
                yb = y_shuffled[start:end]

                activations = self.forward(xb)
                self.backward(activations, yb, lr)

            train_preds = self.forward(X_train)[-1]
            val_preds = self.forward(X_val)[-1]

            train_loss = cross_entropy_loss(y_train, train_preds)
            val_loss = cross_entropy_loss(y_val, val_preds)

            train_acc = np.mean(
                np.argmax(train_preds, axis=1) == np.argmax(y_train, axis=1)
            )
            val_acc = np.mean(
                np.argmax(val_preds, axis=1) == np.argmax(y_val, axis=1)
            )

            history["loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["accuracy"].append(train_acc)
            history["val_accuracy"].append(val_acc)

            if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
                print(
                    f"epoch {epoch:02d}/{epochs:02d} - "
                    f"loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - "
                    f"acc: {train_acc:.4f} - val_acc: {val_acc:.4f}"
                )

        return history

    def save_model(self, filepath):
        model_data = {
            "layer_sizes": self.layer_sizes,
            "weights": self.weights,
            "biases": self.biases,
        }
        np.save(filepath, model_data)
        print(f"> saving model '{filepath}' to disk...")


def main():
    parser = argparse.ArgumentParser(description="Train MLP Classifier")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data_training.csv",
        help="Path to training CSV",
    )
    parser.add_argument(
        "--val_dataset",
        type=str,
        default="data_validation.csv",
        help="Path to validation CSV",
    )
    parser.add_argument(
        "--layer",
        nargs="+",
        type=int,
        default=[24, 24],
        help="Hidden layers sizes",
    )
    parser.add_argument(
        "--epochs", type=int, default=84, help="Number of epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Mini-batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0314, help="Learning rate"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="saved_model.npy",
        help="Path to save weights",
    )
    args = parser.parse_args()

    df_train = pd.read_csv(args.dataset)
    df_val = pd.read_csv(args.val_dataset)

    X_train = df_train.drop("target", axis=1).to_numpy()
    y_train = to_one_hot(df_train["target"].to_numpy())

    X_val = df_val.drop("target", axis=1).to_numpy()
    y_val = to_one_hot(df_val["target"].to_numpy())

    architecture = [X_train.shape[1]] + args.layer + [2]
    model = MLP(architecture)

    history = model.train(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
    )

    model.save_model(args.save_path)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss", linestyle="--")
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history["accuracy"], label="Training Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy", linestyle="--")
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()