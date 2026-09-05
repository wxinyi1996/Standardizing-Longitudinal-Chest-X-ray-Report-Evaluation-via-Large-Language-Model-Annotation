import numpy as np
import pandas as pd
from itertools import combinations

def compute_confidence_intervals(
        boot_matrix,
        original_scores,
        metric_names,
        model_names,
        save_path):

    ci_rows = []

    for metric_idx, metric in enumerate(metric_names):

        print(f"\nMetric : {metric}")

        for model_idx, model in enumerate(model_names):

            values = boot_matrix[:, model_idx, metric_idx]

            score = original_scores[
                model_idx,
                metric_idx
            ]

            lower = np.percentile(values, 2.5)
            upper = np.percentile(values, 97.5)


            ci_rows.append([
                metric,
                model,
                score,
                lower,
                upper
            ])

            print(
                f"{model:<25}"
                f"{score:.4f} "
                f"[{lower:.4f}, {upper:.4f}]"
            )


    ci_df = pd.DataFrame(
        ci_rows,
        columns=[
            "Metric",
            "Model",
            "Score",
            "CI_lower",
            "CI_upper"
        ]
    )


    ci_df.to_csv(
        save_path,
        index=False
    )

    return ci_df

def paired_bootstrap_test(
        boot_matrix,
        metric_names,
        model_names,
        save_dir):


    pair_rows=[]


    for metric_idx, metric in enumerate(metric_names):

        print(f"\nMetric : {metric}")


        for i,j in combinations(
                range(len(model_names)),2):


            score1 = boot_matrix[:,i,metric_idx]

            score2 = boot_matrix[:,j,metric_idx]


            diff = score1-score2


            mean_diff=np.mean(diff)

            prob=np.mean(diff>0)


            lower=np.percentile(
                diff,2.5
            )

            upper=np.percentile(
                diff,97.5
            )


            pair_rows.append([
                metric,
                model_names[i],
                model_names[j],
                mean_diff,
                prob,
                lower,
                upper
            ])



            print(
                f"{model_names[i]} vs "
                f"{model_names[j]} "
                f"Diff={mean_diff:.4f} "
                f"P={prob:.3f}"
            )


    pair_df=pd.DataFrame(
        pair_rows,
        columns=[
            "Metric",
            "Model1",
            "Model2",
            "MeanDiff",
            "Prob(Model1>Model2)",
            "Diff_CI_lower",
            "Diff_CI_upper"
        ]
    )


    pair_df.to_csv(
        f"{save_dir}/paired_bootstrap_test.csv",
        index=False
    )


    return pair_df

def ranking_stability_analysis(
        boot_matrix,
        metric_names,
        model_names,
        save_dir):


    num_boots = boot_matrix.shape[0]
    num_models = len(model_names)


    results={}


    for metric_idx, metric in enumerate(metric_names):


        rank_count=np.zeros(
            (num_models,num_models)
        )


        rank_matrix=np.zeros(
            (num_boots,num_models)
        )


        for b in range(num_boots):


            scores = boot_matrix[
                b,:,metric_idx
            ]


            ranking=np.argsort(
                -scores
            )


            for rank,model in enumerate(ranking):

                rank_count[
                    model,
                    rank
                ]+=1



            rank=np.empty(num_models)


            rank[ranking]=np.arange(
                1,
                num_models+1
            )


            rank_matrix[b]=rank



        rank_prob=(
            rank_count/
            num_boots
        )


        rank_df=pd.DataFrame(
            rank_prob,
            index=model_names,
            columns=[
                f"Rank{i}"
                for i in range(1,num_models+1)
            ]
        )


        rank_df.to_csv(
            f"{save_dir}/ranking_probability_{metric}.csv"
        )


        avg_rank=[]


        for i,model in enumerate(model_names):


            avg=np.mean(
                rank_matrix[:,i]
            )

            lower=np.percentile(
                rank_matrix[:,i],
                2.5
            )

            upper=np.percentile(
                rank_matrix[:,i],
                97.5
            )


            avg_rank.append([
                model,
                avg,
                lower,
                upper
            ])



        avg_rank_df=pd.DataFrame(
            avg_rank,
            columns=[
                "Model",
                "MeanRank",
                "RankCI_lower",
                "RankCI_upper"
            ]
        )


        avg_rank_df.to_csv(
            f"{save_dir}/average_rank_{metric}.csv",
            index=False
        )


        results[metric]=avg_rank_df


    return results

def bootstrap(boot_matrix, original_scores, metric_names, model_names):

    save_dir="./bootstrap_evaluation"
    boot_matrix = np.asarray(boot_matrix)
    original_scores = np.asarray(original_scores)

    ci_df = compute_confidence_intervals(
        boot_matrix,
        original_scores,
        metric_names,
        model_names,
        f"{save_dir}/bootstrap_confidence_interval.csv"
    )


    pair_df = paired_bootstrap_test(
        boot_matrix,
        metric_names,
        model_names,
        save_dir
    )


    ranking_results = ranking_stability_analysis(
        boot_matrix,
        metric_names,
        model_names,
        save_dir
    )


    np.save(
        f"{save_dir}/bootstrap_matrix.npy",
        boot_matrix
    )


    print("\nAll Done.")


    return {
        "confidence_intervals": ci_df,
        "paired_tests": pair_df,
        "ranking_stability": ranking_results
    }