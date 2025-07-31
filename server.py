import flwr as fl
from sklearn.linear_model import LogisticRegression
import numpy as np

# Metric aggregation
def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)} if sum(examples) > 0 else {"accuracy": 0.0}

# Strategy with global recommendations
class PostureStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.global_model_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        self.global_model_parameters = aggregated_parameters
        return aggregated_parameters, metrics

    def get_final_model_recommendations(self):
        if not self.global_model_parameters or not self.global_model_parameters.tensors:
            return ["Global model not ready or has empty parameters."]

        expected_num_features = 4
        try:
            coef = np.frombuffer(self.global_model_parameters.tensors[0], dtype=np.float64)
            intercept = np.frombuffer(self.global_model_parameters.tensors[1], dtype=np.float64)

            if coef.size != expected_num_features:
                return [f"Mismatch: Expected {expected_num_features} features, got {coef.size}"]

            coef = coef.reshape(1, expected_num_features)
            intercept = intercept.reshape(1,)
        except Exception as e:
            return [f"Error reading model parameters: {e}"]

        model = LogisticRegression()
        model.coef_ = coef
        model.intercept_ = intercept
        model.classes_ = np.array([0, 1])

        feature_names = ['neck_angle', 'shoulder_angle', 'shoulder_symmetry', 'risk_score']
        weights = model.coef_[0]
        threshold = 0.5
        recommendations = ["Posture Recommendations:"]

        for i, feature in enumerate(feature_names):
            weight = weights[i]
            if feature == 'neck_angle':
                recommendations.append("  • Good neck alignment." if weight > threshold else "  • Watch your neck posture.")
            elif feature == 'shoulder_angle':
                recommendations.append("  • Upright back posture." if weight > threshold else "  • Try to sit straighter.")
            elif feature == 'shoulder_symmetry':
                recommendations.append("  • Balanced shoulder detected." if weight > threshold else "  • Correct shoulder imbalance.")
            elif feature == 'risk_score':
                recommendations.append("  • Low risk session." if weight > threshold else "  • High ergonomic risk, take breaks.")

        return recommendations

# --- Run Server ---
if __name__ == "__main__":
    strategy = PostureStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

    print("\n" + "="*50)
    print("GLOBAL MODEL RECOMMENDATIONS AFTER TRAINING")
    print("="*50)
    for rec in strategy.get_final_model_recommendations():
        print(rec)
    print("="*50)
