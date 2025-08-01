import flwr as fl
from sklearn.linear_model import LogisticRegression
import numpy as np

from flwr.common import parameters_to_ndarrays

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
        if not self.global_model_parameters:
            return ["Global model not ready or has empty parameters. Run more rounds to train the model."]

        try:
            weights = parameters_to_ndarrays(self.global_model_parameters)
            
            # Ensure weights array has at least two elements (coef_ and intercept_)
            if len(weights) < 2:
                return ["Insufficient model parameters received from clients."]

            coef = np.array(weights[0])
            intercept = np.array(weights[1])

            print(f"[DEBUG] Received coef shape: {coef.shape}, intercept shape: {intercept.shape}")

            # Ensure coef has the expected shape (1, 4)
            if coef.shape != (1, 4):
                return [f"Mismatch: Expected (1, 4) for coef, got {coef.shape}. Check client model architecture."]

            # Create a dummy model to set coef_ and intercept_
            model = LogisticRegression()
            model.coef_ = coef
            model.intercept_ = intercept
            model.classes_ = np.array([0, 1]) # Important for predict_proba to work correctly

            feature_names = ['neck_angle', 'shoulder_angle', 'shoulder_symmetry', 'risk_score']
            weights_importance = model.coef_[0] # Get the coefficients for the features
            
            recommendations = ["Global Posture Recommendations based on aggregated model:"]

            # Interpret weights: a more negative weight means that feature strongly decreases the chance of "good posture" (OWAS <= 2)
            # A weight close to zero means it has less impact
            # A positive weight (ideally not expected for these features given their definition) would imply higher values lead to better posture.

            # We can define thresholds for "strong" or "moderate" impact based on the absolute value of weights.
            # These thresholds might need tuning based on observed weight magnitudes.
            weak_impact_threshold = 0.1
            moderate_impact_threshold = 0.5

            for i, feature in enumerate(feature_names):
                weight = weights_importance[i]
                
                if feature == 'neck_angle':
                    if weight < -moderate_impact_threshold:
                        recommendations.append("  • Significant neck angle correction needed for better posture.")
                    elif weight < -weak_impact_threshold:
                        recommendations.append("  • Minor adjustments to neck posture can improve overall alignment.")
                    else: # Includes positive or very small negative weights
                        recommendations.append("  • Maintain good neck alignment.")
                elif feature == 'shoulder_angle':
                    if weight < -moderate_impact_threshold:
                        recommendations.append("  • Pay close attention to your trunk lean; significant straightening is advised.")
                    elif weight < -weak_impact_threshold:
                        recommendations.append("  • Slight trunk lean adjustments could benefit your posture.")
                    else:
                        recommendations.append("  • Your trunk posture appears stable.")
                elif feature == 'shoulder_symmetry':
                    if weight < -moderate_impact_threshold:
                        recommendations.append("  • Address significant shoulder height differences for better symmetry.")
                    elif weight < -weak_impact_threshold:
                        recommendations.append("  • Work on minor shoulder height adjustments for improved balance.")
                    else:
                        recommendations.append("  • Good shoulder symmetry observed.")
                elif feature == 'risk_score':
                    # For risk_score, a negative weight means higher risk score (bad) makes good posture less likely.
                    # This is consistent. We want to recommend action if risk_score has a strong negative impact.
                    if weight < -moderate_impact_threshold:
                        recommendations.append("  • The overall session risk is high; prioritize frequent breaks and posture changes.")
                    elif weight < -weak_impact_threshold:
                        recommendations.append("  • Monitor your session risk; consider short, regular breaks.")
                    else:
                        recommendations.append("  • Your session risk appears to be well-managed.")
            
            # Add a general recommendation based on the intercept, which represents the baseline log-odds of good posture
            # Higher (less negative) intercept indicates a higher baseline probability of good posture.
            if intercept[0] > 0: # If log-odds are positive, baseline probability > 0.5
                 recommendations.append("  • Overall, a good foundation for healthy posture has been established.")
            else:
                 recommendations.append("  • Continue consistent posture monitoring for sustained improvement.")


            return recommendations
        except Exception as e:
            # Log the full traceback for debugging
            import traceback
            traceback.print_exc()
            return [f"Error processing model parameters for recommendations: {e}"]

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
