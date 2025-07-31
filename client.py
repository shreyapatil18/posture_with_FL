import pandas as pd
from sklearn.linear_model import LogisticRegression
import flwr as fl
import numpy as np

# --- Load Data ---
def load_posture_features_data(file_path):
    df = pd.read_csv(file_path)

    if df.empty:
        print(f"[ERROR] Empty CSV: {file_path}")
        return np.array([]), np.array([])

    df['is_good_posture'] = df['OWAS_score'].apply(lambda x: 1 if x <= 2 else 0)

    features = ['neck_angle', 'shoulder_angle', 'shoulder_symmetry', 'risk_score']
    for col in features:
        if col not in df.columns:
            df[col] = 0.0

    X = df[features].values
    y = df['is_good_posture'].values

    if X.size > 0:
        max_vals = np.max(X, axis=0)
        max_vals[max_vals == 0] = 1.0
        X = X / max_vals

    print(f"[DEBUG] Loaded: X.shape = {X.shape}, y.shape = {y.shape}, y = {np.unique(y)}")
    return X, y

# --- Flower Client ---
class PostureClient(fl.client.NumPyClient):
    def __init__(self, client_id, data_path):
        self.client_id = client_id
        self.X, self.y = load_posture_features_data(data_path)
        self.num_classes = len(np.unique(self.y)) if self.y.size > 0 else 2
        self.model = LogisticRegression(max_iter=200)
        self._initialize_model_params()

    def _initialize_model_params(self):
        self.model.coef_ = np.zeros((1, 4), dtype=np.float64)
        self.model.intercept_ = np.zeros(1, dtype=np.float64)
        self.model.classes_ = np.array([0, 1])

    def get_parameters(self, config):
        self.model.coef_ = np.resize(self.model.coef_, (1, 4))
        self.model.intercept_ = np.resize(self.model.intercept_, (1,))
        return [self.model.coef_, self.model.intercept_]

    def set_parameters(self, parameters):
        self.model.coef_ = np.array(parameters[0]).reshape(1, 4)
        self.model.intercept_ = np.array(parameters[1]).reshape(1,)
        self.model.classes_ = np.array([0, 1])

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        num_examples = len(self.X)

        if num_examples > 0 and self.num_classes > 1:
            try:
                # Always reset shapes before training
                self.model.coef_ = np.zeros((1, 4), dtype=np.float64)
                self.model.intercept_ = np.zeros(1, dtype=np.float64)
                self.model.classes_ = np.array([0, 1])
                self.model.fit(self.X, self.y)
                acc = self.model.score(self.X, self.y)
                print(f"[Client {self.client_id}] Fit done. Accuracy: {acc:.4f}")
                return self.get_parameters(config), num_examples, {"accuracy": acc}
            except Exception as e:
                print(f"[Client {self.client_id}] Fit error: {e}")
                return self.get_parameters(config), 0, {"accuracy": 0.0}
        else:
            return self.get_parameters(config), 0, {"accuracy": 0.0}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        if len(self.X) > 0 and self.num_classes > 1:
            try:
                acc = self.model.score(self.X, self.y)
                return 1 - acc, len(self.X), {"accuracy": acc}
            except:
                return 1.0, 0, {"accuracy": 0.0}
        return 1.0, 0, {"accuracy": 0.0}

# --- Run ---
if __name__ == "__main__":
    client_id = "client_features_2"
    data_file = "posture_features_20250728_112508.csv"  # Update to your file
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=PostureClient(client_id, data_file).to_client()
    )
