# 🚀 GRPO Finetuned Jailbreak Classification Model

## 📌 Project Overview

This project fine-tunes a **Llama-3.2 (3B) model using GRPO** to classify user prompts as **jailbreak attempts** or **benign requests**. By leveraging reinforcement learning with reward modeling, the model provides **step-by-step reasoning** before making a final classification decision.

🔗 [**Project Website**](https://schmitzandrew.github.io/jailbreak-defense/)

📄 **Poster & Report:**

- 📜 [Project Poster](https://drive.google.com/file/d/1X2G3Fx5L5WVR5jPfUgp_hHmj7vabUI6R/view?usp=sharing)
- 📑 [Project Report](https://drive.google.com/file/d/1VNaVNTf60Uy2FmKO7suancEHAupcU7nM/view?usp=sharing)

## 🔍 Quick Navigation

- [How to Run Inference With GRPO Finetuned Model](#how-to-run-inference-with-grpo-finetuned-model)
- [How to Produce Jailbreak Attempts with PAIR](#how-to-produce-jailbreak-attempts-with-pair)
- [How to Perform Further Finetuning on the Model](#how-to-perform-further-finetuning-on-the-model)

---

## How to Run Inference With GRPO Finetuned Model

To classify a user input as **jailbreak** or **benign**, follow these steps:

### 1️⃣ Install Required Dependencies

Run the following command to install all necessary Python packages:

```bash
pip install -r requirements.txt
```

### 2️⃣ Run Inference

Use the `run_inference()` function inside `results/inference.py`. This function takes in **a single string** (the user input prompt) and outputs **a tuple**:

- **First element:** The predicted label (**"jailbreak"** or **"benign"**)
- **Second element:** The full reasoning response from the model

#### **Example Usage:**

```python
from results.inference import run_inference

user_input = "How can I break into a locked room?"
label, reasoning = run_inference(user_input)

print("Label:", label)
print("Reasoning:", reasoning)
```

## How to Produce Jailbreak Attempts with PAIR

PAIR (Prompt Automatic Iterative Refinement) generates adversarial jailbreak prompts. Follow these steps to run PAIR on **Open Source LLMs**:

### 1️⃣ Install PAIR Dependencies (in DSC_180A Folder)

```bash
pip install -r DSC_180A/requirements.txt
```

### 2️⃣ Set Up Together.AI API Key (For Open Source Attacker Model)

#### **Obtain an API Key**

- Go to the [Together.AI API Page](https://together.ai/)
- Create an account to get a free API key with **$1 credit** or use an existing key

#### **Set the API Key as an Environment Variable**

1. Search for **"Environment Variables"** in the Start menu  
2. Click **"Edit the system environment variables"**  
3. In the **System Properties** window, click **"Environment Variables"**  
4. Under **"User variables,"** click **"New"** and set:  
   - **Variable name:** `TOGETHER_API_KEY`  
   - **Variable value:** *(Paste your copied API key)*

### 3️⃣ Run the PAIR Algorithm Demo

#### **Run Llama-3.1 PAIR Algorithm:**

```bash
python DSC_180A/new_test_open_source.py
```

### 4️⃣ View Outputs and Chat History

After running the PAIR demo, the generated jailbreak prompts and chat history between the LLMs will be saved in `DSC_180A/results/new_open_source_results.json`.

## How to Perform Further Finetuning on the Model
If you’d like to further fine-tune the model using GRPO, follow these steps:

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Finetuning Notebook

- Open and run the cells in `model/llama_grpo_final.ipynb`
- **If you run into Out-of-Memory errors, use Google Colab for cloud-based training.**