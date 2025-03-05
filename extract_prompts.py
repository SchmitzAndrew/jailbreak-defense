import json
import os

def extract_prompts_with_rating_10(json_file_path):
    """
    Extract prompts with a rating of 10 from the JSON file.
    Only include the first prompt that scored a 10 in each set.
    
    Args:
        json_file_path (str): Path to the JSON file
        
    Returns:
        list: List of dictionaries containing objective and prompt
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except json.JSONDecodeError:
        print(f"Error: {json_file_path} is not a valid JSON file.")
        return []
    except FileNotFoundError:
        print(f"Error: {json_file_path} not found.")
        return []
    
    results = []
    
    # Check if data is a list (array of objects)
    if isinstance(data, list):
        for item in data:
            # Extract objective and category
            objective = item.get('Objective', 'No objective provided')
            category = item.get('Category', 'No category provided')
            
            # Process output data
            output = item.get('Output', {})
            
            # Check different sections for prompts with rating 10
            for section_name in output:
                section = output[section_name]
                if isinstance(section, list):
                    # Look for the first prompt with rating 10 in this section
                    for prompt_data in section:
                        if isinstance(prompt_data, dict) and 'Rating' in prompt_data and prompt_data['Rating'] == 10:
                            results.append({
                                'Objective': objective,
                                'Category': category,
                                'Section': section_name,
                                'Prompt': prompt_data.get('Prompt', 'No prompt provided')
                            })
                            # Only include the first prompt with rating 10
                            break
    
    return results

def main():
    # Path to the JSON file
    json_file_path = 'results/new_open_source_results.json'
    
    # Extract prompts with rating 10
    prompts = extract_prompts_with_rating_10(json_file_path)
    
    # Print the results
    print(f"Found {len(prompts)} prompts with rating 10.")
    for i, prompt_data in enumerate(prompts, 1):
        print(f"\n{i}. Objective: {prompt_data['Objective']}")
        print(f"   Category: {prompt_data['Category']}")
        print(f"   Section: {prompt_data['Section']}")
        print(f"   Prompt: {prompt_data['Prompt'][:200]}...")  # Print first 200 chars
    
    # Save the results to a new JSON file
    output_file = 'results/rating_10_prompts.json'
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(prompts, file, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main() 