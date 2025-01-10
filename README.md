# PAIR Algorithm Implementation

This repository contains a demonstration of the PAIR algorithm. The PAIR algorithm is a state-of-the-art jailbreaking algorithm to get objectionable and morally questionable content from a target LLM. It works by pitting another attacker LLM against the target, improving its own jailbreak approach with each attempt.

Follow the instructions below to test the algorithm on your own.

---

## Prerequisites

- **Python Version**: Ensure that you have Python 3.10 or newer installed. Check your Python version by running:

  ```bash
  python --version

---

## Installation Instructions

1. Clone the Repository
  ```bash
  git clone https://github.com/jeffmeredith/pair-implement.git
  ```

2. Navigate to the Project Directory
  ```bash
  cd pair-implement
  ```

3. Create a Virtual Environment
  ```bash
  python -m venv venv
  ```
Next, activate this environment by running:
  ```bash
  .\venv\Scripts\activate
  ```

4. Install Dependencies
  ```bash
  pip install -r requirements.txt
  ```

---

## Set Up OpenAI API Key (For GPT-3.5 Attacker Model)

5. Obtain an API Key
    1. Go to the [OpenAI API Key page](https://platform.openai.com/api-keys).
    2. Generate a new API key or use an existing one to copy to your clipboard.

6. Set the API Key as an Environment Variable
To allow the script to use your API key, set it as an environment variable.
- Search for "Environment Variables" in the Start menu
- Click "Edit the system environment variables"
- In the System Properties window, click "Environment Variables"
- Under "User variables," click "New" and set:
    - Variable name: ```OPENAI_API_KEY```
    - Variable value: Paste your copied API key

---

## Set Up Together.AI API Key (For Open Source Attacker Model)

5. Obtain an API Key
    1. Go to the [together.ai API Page](https://api.together.xyz/).
    2. Create an account to get a free API key with $1 credit or use an existing key.

6. Set the API Key as an Environment Variable
To allow the script to use your API key, set it as an environment variable.
- Search for "Environment Variables" in the Start menu
- Click "Edit the system environment variables"
- In the System Properties window, click "Environment Variables"
- Under "User variables," click "New" and set:
    - Variable name: ```TOGETHER_API_KEY```
    - Variable value: Paste your copied API key

---

## Running the PAIR Algorithm Demo

7. Run the Script <br><br>
To run the GPT-3.5 implementation of PAIR algorithm on the benchmark, type the following into command line:
  ```bash
  python test_pair_gpt.py
  ```

To run the Mixtral-8x7B implementation of PAIR on the benchmark, type:
  ```bash
  python test_pair_open_source.py
  ```

---

## View Outputs and Chat History

After running the PAIR demo, you can view the resulting jailbreak prompt and chat history between the LLMs in the json files inside the ```results``` folder.