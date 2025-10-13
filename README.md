# Benchmarking Large Language Models and Clinicians Using Locally Generated Primary Healthcare Vignettes in Kenya 

## 1. Running LLMs

The script `run_llms.py` is used to generate responses from multiple large language models (LLMs) for a set of clinical scenarios. It leverages the [LangChain](https://python.langchain.com/) framework to interface with different LLM providers, including OpenAI, Google, HuggingFace, and Ollama. The script reads the scenarios from a CSV file, sends each scenario to the selected LLMs, and stores the responses in a SurrealDB database as well as a CSV file. The models and system prompts can be configured within the script. This process enables standardised benchmarking of LLMs on the same set of clinical queries.

## 2. Descriptive and Ordinal Logistic Regression

The data for these analyses are available in the *datasets* folder. 'Combined review data.csv' or 'Combined review data.parquet' contain the expert panel ratings for the 507 vignettes, while 'Prompt responses.xlsx' contains 
the formatted responses from the clinicians and LLM models.

The R scripts `descriptives.R` and `analyses/models.R` provide the statistical analysis pipeline:

- **Descriptive Statistics (`descriptives.R`):**  
  This script summarizes the LLM responses by calculating means and standard deviations of scores across different clinical domains and models. It produces summary tables and radar plots to visualize model performance on various dimensions.

- **Ordinal Logistic Regression (`analyses/models.R`):**  
  This script fits Bayesian ordinal logistic regression models using the `brms` package. The models compare LLMs on 5-point Likert outcomes across multiple domains, accounting for hierarchical structure (e.g., random effects for panel ). The script includes:
    - Model fitting with informative and non-informative priors.
    - Extraction and formatting of fixed effects as odds ratios.
    - Posterior predictive checks and visualizations.
    - Pairwise comparisons of models within each domain, reporting differences in log-odds and odds ratios.
    - Visualization of predicted probabilities and model contrasts.

All outputs (tables, plots, and model summaries) are saved to the specified output directory for reporting and further interpretation.
