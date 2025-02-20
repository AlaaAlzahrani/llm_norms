

<h1 align="center">Obtain LLM-generated norms </h1>

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" alt="Python">
</p>

This is a python script that allows you to generate LLM estimates for one or more psycholinguistic dimensions (concreteness, familiarity, Valence, arousal...etc) derived from  OpenAI (GPT-4) or Anthropic (Claude) models using your preferred prompt(s).


## How to use this script

1. **Install Python**

2. **Clone this repository**

3. **Prepare your word list**
   - Create a text file named `words.txt` in the `data/input` folder. Add your words, one per line.

4. **Create and activate a new virtual environment**
   ```bash
   python -m venv env   
   env\Scripts\activate # for windows
   source env/bin/activate # for Mac/Linux
   ```

5. **Install Dependencies**
   ```bash
   pip install numpy   
   pip install -r requirements.txt
   ```

5. **Set Up your API Keys**
   - Create a new file named `.env`. Copy the content from `.env.example` to your `.env` file. Then add your API keys to `.env`. 

6.  **Obtain LLM-generated norm ratings**

Use the command line-interface to enter your command. In this command, each option starts with -- and is followed by your choice:

```bash
python src/main.py --dimensions valence --prompt-types detailed --model gpt-4o
```


Available Options:
```bash
--dimensions     Choose one or more dimensions: concreteness, arousal, valence
                 Example: --dimensions concreteness
                 For multiple: --dimensions concreteness arousal

--prompt-types   Choose one or more prompts: detailed, standard, short
                 Example: --prompt-types detailed
                 For multiple: --prompt-types detailed standard short

--model          Choose one or more models: gpt-4, gpt-4o, claude-3-5-sonnet-20241022
                 Example: --model gpt-4o 
                 For multiple: --model gpt-4o gpt-4 claude-3-5-sonnet-20241022

--batch-size     Choose how many words to process at once
                 Example: --batch-size 100
```

Full Example:
```bash
python src/main.py --dimensions concreteness valence --prompt-types detailed standard short --model gpt-4o gpt-4 --batch-size 100
```

7.   **Results**

Results are saved in `data/output` as CSV files.

Each file contains:
- The word
- Rating 
- Model used
- Prompt type


## Development

Please feel free to change the script to add new:
-  dimensions
- prompts
- models

## License
This work is licensed under an [MIT](https://github.com/AlaaAlzahrani/ARSA/blob/master/LICENSE) license.


## Citation

If you use this script, please cite the following paper to support the authors.

```bibtex
@misc{Alzahrani:2025:kalimah,
    title = "{Kalimah Norms: Ratings for 2,467 Modern Standard Arabic words on two scales}",
    author = {Alzahrani, Alaa and Aljuaythin, Wafa and Alshumrani, Hassan and Saleh, Alaa Mamoun and Mostafa, Mohamed M.},
    year = "2025",
    note = {Submitted manuscript}
}
```