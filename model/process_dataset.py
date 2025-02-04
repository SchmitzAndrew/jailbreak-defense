# Load required libraries
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd
import numpy as np
from tqdm import tqdm
from rapidfuzz import fuzz


# Load and process jailbreak dataset
jailbreak_dataset = load_dataset('TrustAIRLab/in-the-wild-jailbreak-prompts', 'jailbreak_2023_05_07')['train']
jailbreak_df = pd.DataFrame(jailbreak_dataset)

print("\nJailbreak Dataset Info:")
print(f"Total rows: {len(jailbreak_df)}")
print(f"Unique prompts: {jailbreak_df['prompt'].nunique()}")
print("\nSample of jailbreak prompts:")
print(jailbreak_df[['prompt']].head(3))

# Remove exact duplicates from jailbreak dataset
jailbreak_df_unique = jailbreak_df.drop_duplicates(subset=['prompt'])
print(f"\nShape after exact duplicates: {len(jailbreak_df_unique)} rows")

def find_similar_prompts(prompts, threshold=90):
    """
    Find groups of similar prompts using fuzzy matching.
    Args:
        prompts: List of prompts to compare
        threshold: Similarity score (0-100) above which prompts are considered duplicates
    Returns:
        Dictionary mapping original prompt indices to groups of similar prompt indices
    """
    similar_groups = {}
    processed = set()
    
    print("Finding similar prompts...")
    for i in tqdm(range(len(prompts))):
        if i in processed:
            continue
            
        current_prompt = prompts[i]
        group = [i]
        
        for j in range(i + 1, len(prompts)):
            if j in processed:
                continue
                
            if fuzz.ratio(current_prompt, prompts[j]) >= threshold:
                group.append(j)
                processed.add(j)
                
        if len(group) > 1:
            similar_groups[i] = group
        processed.add(i)
    
    return similar_groups

# Find and remove fuzzy duplicates from jailbreak dataset
jailbreak_prompts = jailbreak_df['prompt'].tolist()
similar_groups = find_similar_prompts(jailbreak_prompts, threshold=90)

indices_to_keep = set(range(len(jailbreak_prompts))) - {
    idx for group in similar_groups.values() 
    for idx in group[1:]
}
jailbreak_df_fuzzy_unique = jailbreak_df.iloc[list(indices_to_keep)]

# Load regular prompts dataset
regular_dataset = load_dataset('TrustAIRLab/in-the-wild-jailbreak-prompts', 'regular_2023_05_07')['train']
regular_df = pd.DataFrame(regular_dataset)

# Sample equal number of regular prompts
num_jailbreak = len(jailbreak_df_fuzzy_unique)
regular_df_sampled = regular_df.sample(n=num_jailbreak, random_state=42)

# Add labels to both datasets
jailbreak_df_fuzzy_unique['label'] = "jailbreak"
regular_df_sampled['label'] = "safe"

# Combine datasets
combined_df = pd.concat([jailbreak_df_fuzzy_unique, regular_df_sampled]).sample(frac=1, random_state=42)

# Convert to HuggingFace Dataset format
combined_dataset = Dataset.from_pandas(combined_df)

# Define the prompt template
instruction_prompt = """Below is an instruction or prompt. Determine if it is a safe, regular prompt to a large language model or a jailbreak attempt.

### Instruction:
{}

### Input:


### Response:
{}"""

EOS_TOKEN = "<|endoftext|>"  # temporary solution since we're not in the jupyter notebook

def formatting_prompts_func(examples):
    prompts = examples["prompt"]
    # Create empty inputs since we don't have input context
    inputs = [""] * len(prompts)
    # Use detailed responses based on the label
    responses = []
    for label in examples["label"]:
        if label == "jailbreak":
            responses.append("jailbreak")
        else:
            responses.append("safe")
    texts = []
    for prompt, input_text, response in zip(prompts, inputs, responses):
        text = instruction_prompt.format(prompt, response) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

# Format the dataset
formatted_dataset = combined_dataset.map(formatting_prompts_func, batched=True)

# Print final statistics
print("\nFinal Dataset Statistics:")
print(f"Total examples: {len(formatted_dataset)}")
print(f"Jailbreak examples: {len(jailbreak_df_fuzzy_unique)}")
print(f"Regular examples: {len(regular_df_sampled)}")
print("\nLabel distribution:")
print(pd.Series(combined_dataset['label']).value_counts())

# Display samples from the final formatted dataset
print("\nSample from formatted dataset:")
print(formatted_dataset[:3]['text']) 