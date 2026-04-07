# How to run

## Install instructions
1. You need to install Docker and set up a devcontainer. VSCode has great support for this.
2. You need to install piston (https://github.com/huggingface/open-r1/blob/main/slurm/piston/README.md) and
install the package for Codeforces.
3. From Codeforces-R1 (https://huggingface.co/datasets/open-r1/codeforces), you need to download the dataset with all the problems and download the generated tests.
4. In the .devcontainer folder, you need to create a .env file and setup `OPENAI_ENDPOINT`, `OPENAI_TOKEN`. You have .dev_env as an example.

## How to run
There is a `parameters.yaml` file that allows you to specify various parameters for each task. The following tasks are supported:
- create_dataset: Creates a .csv based on multiple .json files extracted via the Codeforces API with various submissions from different users. (skip if you already have the csv)
- preprocess_dataset: Preprocesses the generated .csv. Also splits the dataset into train, validation and test folds. (skip if you already have the preprocessed csv)
- predict: Using the preprocessed dataset, run various predicts with different methods. All the methods should use the same model.
- evaluate: Create multiple evaluation .csv files based on the predicted results.
- fit: Do prompt optimization via GEPA.

If you run via VSCode, there are multiple tasks defined in .vscode. If not, you just do `python main.py [step]`. E.g. `python main.py --predict`.

## Data
In the `/data` folder you can find the following:
- dataset.csv - The original dataset
- dataset_preprocessed.csv - The preprocessed dataset
- dataset_train.csv - Train dataset used for GEPA
- dataset_val.csv - Validation dataset used for GEPA
- dataset_test.csv - Test dataset used for prediction and evaluation.
## Models
In the `data/models` you can find the prompt after GEPA optimization.
