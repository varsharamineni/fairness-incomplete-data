import itertools
import numpy as np
from inconst_joint.distributions import generate_random_joint, generate_random_classifier_probs, generate_systematic_joint, generate_systematic_classifier_probs, compute_kl_y_given_e
from inconst_joint.metrics import compute_disparate_impact_from_joint, compute_demographic_disparity_from_joint
from inconst_joint.break_consistency import break_o_consistency, check_marginal_consistency_kl
from inconst_joint.valid_joint import solve_joint_distribution_iterate_cond
import pandas as pd


from data.marginals import compute_joint_indep_given_overlap, compute_joint_marginal_preservation

def obtain_plausible_metrics(joint, classifier_probs, metric_fns, alpha=1.0, beta=1.0, min_value=0.00, max_value=1.0, num=100):
    
        """
        Compare true vs. estimated fairness metrics over reconstructed joints.

        Args:
            joint (dict): True joint distribution over (s, o, e)
            classifier_probs (dict): {(s, o): P(ŷ=1 | s, o)}
            metric_fns (dict): {metric_name: function(joint, classifier_probs)}
            alpha, beta (float): distortion parameters
            min_value, max_value, num: candidate joint generation parameters

        Returns:
            dict: metric comparison statistics for each fairness metric
        """
        results = {}
        pa, pb = {}, {}

        for key, value in joint.items():
            s, o, e = int(key[1]), int(key[2]), int(key[3])
            pa[(o, e)] = pa.get((o, e), 0.0) + value
            pb[(s, o)] = pb.get((s, o), 0.0) + value

        pb = break_o_consistency(pb, alpha=alpha, beta=beta)
        const = check_marginal_consistency_kl(pa, pb)

        candidate_joints = solve_joint_distribution_iterate_cond(
            pa=pa, pb=pb, min_value=min_value, max_value=max_value, num=num
        )
        
        indep_given_overlap_joint = compute_joint_indep_given_overlap(pa, pb)
        marginal_preservation_joint = compute_joint_marginal_preservation(pa, pb)


        results["const"] = const
        results["alpha"] = alpha
        results["beta"] = beta
        results["num_candidates"] = len(candidate_joints)

        for name, fn in metric_fns.items():
            true_val = fn(joint, classifier_probs)
            indep_given_overlap_val = fn(indep_given_overlap_joint, classifier_probs)
            marginal_preservation_val = fn(marginal_preservation_joint, classifier_probs)
            
            
            candidate_vals = [fn(c, classifier_probs) for c in candidate_joints]
            kl_true_joints = [compute_kl_y_given_e(joint, c, classifier_probs) for c in candidate_joints]
            
            results['avg_kl_y_given_e'] = np.mean(kl_true_joints)

            if candidate_vals:
                results[f"true_{name}"] = true_val
                results[f"mean_{name}"] = np.mean(candidate_vals)
                results[f"true_mean_diff_{name}"] = true_val - np.mean(candidate_vals)
                results[f"min_{name}"] = min(candidate_vals)
                results[f"max_{name}"] = max(candidate_vals)
                results[f"range_{name}"] = max(candidate_vals) - min(candidate_vals)
                results[f"avg_diff_{name}"] = np.mean([abs(v - true_val) for v in candidate_vals])
                results[f'plausible_metrics_{name}'] = candidate_vals
                
                
                results[f"indep_given_overlap_{name}"] = indep_given_overlap_val
                results[f"marginal_preservation_{name}"] = marginal_preservation_val
            else:
                for k in ["true", "mean", "true_mean_diff", "min", "max", "range", "avg_diff"]:
                    results[f"{k}_{name}"] = np.nan

        return results



from data.marginals import compute_distributions


def run_full_experiment_real_data(
    data_folder = 'real_data/processed/german_clean.csv',
    s_var='employment-since',
    o_var='housing',
    e_var='sex',
    alphas=[1.0, 1.5, 2.0],
    betas=[1.0, 0.8, 0.5],
    num_candidates=100,
    metric_fns= {
    "di": compute_disparate_impact_from_joint,
    "dd": compute_demographic_disparity_from_joint},
    classifier_probs=None,
    seed=100
):
    results = []
    
    pa, pb, joint = compute_distributions(data_folder, s_var, o_var, e_var)

    if classifier_probs is None:
        classifier_probs = generate_random_classifier_probs(seed=seed)

    for alpha, beta in itertools.product(alphas, betas):
        res = obtain_plausible_metrics(
            joint, classifier_probs, metric_fns,
            alpha=alpha, beta=beta,
            num=num_candidates
        )
        results.append(res)

    return pd.DataFrame(results)



