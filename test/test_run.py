import numpy as np
from inconst_joint.experiment import compare_disparate_impact_with_reconstructed_joints
from inconst_joint.distributions import generate_random_joint

# Test with random joint and classifier probabilities
if __name__ == "__main__":
    joint = generate_random_joint()
    classifier_probs = {
        (0, 0): 0.7, (0, 1): 0.4,
        (1, 0): 0.3, (1, 1): 0.6
    }

    # Run experiment
    results = compare_disparate_impact_with_reconstructed_joints(
        joint=joint,
        classifier_probs=classifier_probs,
        min_value=0.0,
        max_value=1.0,
        num=50
    )

    # Print the results
    print(f"True metric: {results['true_metric']}")
    print(f"Min candidate metric: {results['min_candidate_metric']}")
    print(f"Max candidate metric: {results['max_candidate_metric']}")
    print(f"Number of valid reconstructions: {results['num_candidates']}")
