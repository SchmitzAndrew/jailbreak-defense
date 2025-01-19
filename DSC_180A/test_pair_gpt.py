from pair import pair_algorithm
import pandas as pd
import json
import os

results = []
benchmark_df = pd.read_csv('behaviors_benchmark.csv')
complete_objectives = set()
if os.path.isfile("results/gpt_results.json"):
    with open("results/gpt_results.json", "r") as json_file:
        data = json.load(json_file)
        for result in data:
            complete_objectives.add(result['Objective'])
            results.append(result)

# Record the results of running PAIR into a dictionary for each objective
for _, row in benchmark_df.iterrows():
    if row['Goal'] in complete_objectives:
        continue
    retries = 0
    while True:
        try:
            row_dict = {
                "Objective": row["Goal"],
                "Category": row["Category"],
                "Output": pair_algorithm(num_iterations=4, attack_objective=row["Goal"], start_string=row["Target"])
            }
            # Append the dictionary to the result list
            results.append(row_dict)
            with open("results/gpt_results.json", "w") as json_file:
                json.dump(results, json_file, indent=4)
            break
        except json.JSONDecodeError as e:
            retries += 1
            print(f"Retrying: {retries}")

print("Results saved to results/gpt_results.json")