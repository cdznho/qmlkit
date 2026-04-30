"""Train an SVM on quantum-inspired features."""

from __future__ import annotations

from qmlkit import QuantumKernel


def main() -> None:
    """Run the kernel SVM example."""

    from sklearn.datasets import make_moons
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC

    X, y = make_moons(n_samples=300, noise=0.22, random_state=7)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7, stratify=y
    )

    model = Pipeline([
        ("embed", QuantumKernel(method="iqp")),
        ("clf", SVC(gamma="scale")),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"kernel_svm accuracy={accuracy_score(y_test, predictions):.3f}")


if __name__ == "__main__":
    main()
