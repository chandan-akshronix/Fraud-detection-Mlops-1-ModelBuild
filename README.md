# Fraud-detection-Mlops-1-ModelBuild

This repository constitutes the model-building phase of a fraud detection MLOps pipeline, utilizing Amazon SageMaker to preprocess data, train an XGBoost model, evaluate its performance, and register it for deployment. The workflow is orchestrated via SageMaker Pipelines, ensuring scalability, reproducibility, and adaptability for detecting fraudulent transactions.

## Features

- **Custom Data Preprocessing**: Implements feature engineering tailored for fraud detection, such as frequency encoding and time-based features.
- **XGBoost Model Training**: Trains an XGBoost model with hyperparameter optimization to maximize fraud detection accuracy.
- **Model Evaluation**: Assesses model performance using precision, recall, F1-score, and AUC-ROC metrics after batch transformation.
- **Pipeline Orchestration**: Leverages SageMaker Pipelines to automate preprocessing, training, evaluation, and model registration.
- **Bias and Quality Checks**: Integrates SageMaker Clarify and Model Monitor for data quality, model quality, bias, and explainability analysis.
- **Testing**: Includes test scripts to ensure pipeline reliability.

## File Structure

The repository is structured as follows:

```
Fraud-detection-Mlops-1-ModelBuild/
├── tests/
│   └── test_pipelines.py           # Unit tests for pipeline validation
├── pipelines/
│   ├── Fraud/
│   │   ├── __init__.py             # Module initialization
│   │   ├── custom_transformers.py  # Custom preprocessing logic
│   │   ├── evaluate_post_transform.py # Post-transform evaluation script
│   │   ├── pipeline.py             # SageMaker pipeline definition
│   │   ├── preprocess.py           # Data preprocessing and splitting
│   │   └── xgboost_inference.py    # XGBoost inference logic
│   ├── __init__.py                 # Pipelines module initialization
│   ├── __version__.py              # Version metadata
│   ├── _utils.py                   # Shared utility functions
│   ├── get_pipeline_definition.py  # Pipeline definition retrieval script
│   └── run_pipeline.py             # Pipeline execution script
├── .coveragerc                     # Code coverage configuration
├── .gitignore                      # Git ignore rules
├── .pydocstylerc                   # Documentation style configuration
├── codebuild-buildspec.yml         # AWS CodeBuild build specification
├── CONTRIBUTING.md                 # Contribution guidelines
├── LICENSE                         # License terms
├── README.md                       # Project documentation (this file)
├── sagemaker-pipelines-project.ipynb # Example notebook
├── setup.cfg                       # Setup tools configuration
├── setup.py                        # Package installation script
└── tox.ini                         # Testing environment configuration
```

## File Descriptions

### `tests/test_pipelines.py`
- **Purpose**: Validates the pipeline's functionality through unit tests.
- **How It Works**: Executes test cases using mock data or predefined inputs to ensure each pipeline step performs as expected.
- **Input**: Test data or mocked pipeline inputs.
- **Output**: Test results (pass/fail) logged to the console or a report.

### `pipelines/Fraud/__init__.py`
- **Purpose**: Initializes the `Fraud` module for importability.
- **How It Works**: Acts as a Python package marker, enabling imports of submodules.
- **Input**: None.
- **Output**: None (module initialization only).

### `pipelines/Fraud/custom_transformers.py`
- **Purpose**: Contains custom preprocessing transformers for fraud detection.
- **How It Works**: 
  - `FrequencyEncoder`: Encodes categorical variables based on their frequency.
  - `FeatureEngineeringTransformer`: Generates fraud-specific features (e.g., transaction frequency, time deltas).
- **Input**: Raw dataset with categorical and numerical features.
- **Output**: Transformed dataset with encoded and engineered features.

### `pipelines/Fraud/evaluate_post_transform.py`
- **Purpose**: Evaluates model performance post-batch transform.
- **How It Works**: Reads predictions, compares them to ground truth, and computes metrics like precision, recall, F1-score, and AUC-ROC.
- **Input**: Prediction file from batch transform (e.g., CSV or JSON).
- **Output**: JSON file containing evaluation metrics.

### `pipelines/Fraud/pipeline.py`
- **Purpose**: Defines the end-to-end SageMaker pipeline.
- **How It Works**: Orchestrates steps including preprocessing, training, batch transform, evaluation, and model registration using SageMaker SDK.
- **Input**: Pipeline parameters (e.g., S3 data path, instance types, hyperparameters).
- **Output**: A configured SageMaker pipeline instance ready for execution.

### `pipelines/Fraud/preprocess.py`
- **Purpose**: Preprocesses raw data and splits it into train, validation, and test sets.
- **How It Works**: Applies transformations from `custom_transformers.py`, handles missing values, and performs dataset splitting.
- **Input**: Raw data from an S3 bucket.
- **Output**: Preprocessed train, validation, and test datasets; a preprocessing pipeline artifact.

### `pipelines/Fraud/xgboost_inference.py`
- **Purpose**: Defines inference logic for the XGBoost model.
- **How It Works**: Loads the trained model and generates predictions (probabilities or binary labels) on input data.
- **Input**: Preprocessed data in a compatible format (e.g., CSV).
- **Output**: Predictions (e.g., fraud probability scores).

### `pipelines/__init__.py`
- **Purpose**: Initializes the `pipelines` module.
- **How It Works**: Enables imports of submodules like `Fraud`.

### `pipelines/__version__.py`
- **Purpose**: Stores version information for the pipeline.
- **How It Works**: Provides a single source of truth for versioning, accessible via imports.
- **Input**: None.
- **Output**: Version string (e.g., "1.0.0").

### `pipelines/_utils.py`
- **Purpose**: Contains shared utility functions.
- **How It Works**: Offers helper methods (e.g., S3 file operations, logging) used across pipeline scripts.
- **Input**: Varies by function (e.g., S3 paths, dataframes).
- **Output**: Varies by function (e.g., downloaded files, formatted data).

### `pipelines/get_pipeline_definition.py`
- **Purpose**: Retrieves the pipeline definition for inspection or deployment.
- **How It Works**: Queries SageMaker to export the pipeline configuration.
- **Input**: Pipeline name or ARN.
- **Output**: JSON file or string containing the pipeline definition.

### `pipelines/run_pipeline.py`
- **Purpose**: Executes the SageMaker pipeline.
- **How It Works**: Parses command-line arguments and triggers pipeline execution via the SageMaker SDK.
- **Input**: Command-line arguments (e.g., AWS region, role ARN, S3 paths).
- **Output**: Pipeline execution status (e.g., "Running", "Completed").

### `.coveragerc`
- **Purpose**: Configures code coverage reporting.
- **How It Works**: Specifies which files to include/exclude in coverage analysis during testing.

### `.gitignore`
- **Purpose**: Defines files and directories to exclude from version control.
- **How It Works**: Prevents temporary files, logs, or sensitive data from being committed.

### `.pydocstylerc`
- **Purpose**: Enforces documentation style standards.
- **How It Works**: Configures `pydocstyle` to check for consistent docstrings.

### `codebuild-buildspec.yml`
- **Purpose**: Specifies the AWS CodeBuild build process.
- **How It Works**: Defines build phases (e.g., install, test, deploy) and artifacts for CI/CD.
- **Input**: Source code from the repository.
- **Output**: Build artifacts (e.g., test reports, packaged code).

### `sagemaker-pipelines-project.ipynb`
- **Purpose**: Demonstrates pipeline usage in a Jupyter notebook.
- **How It Works**: Provides executable code to set up and run the pipeline interactively.
- **Input**: User inputs (e.g., S3 paths, parameters).
- **Output**: Pipeline execution results and visualizations.

### `setup.cfg`
- **Purpose**: Configures setup tools for the package.
- **How It Works**: Defines metadata, dependencies, and build options.
- **Input**: None (configuration file).
- **Output**: None (used by `setup.py`).

### `setup.py`
- **Purpose**: Installs the package and its dependencies.
- **How It Works**: Runs `pip install` to set up the environment based on `setup.cfg`.
- **Input**: None (executed directly).
- **Output**: Installed Python package.

### `tox.ini`
- **Purpose**: Configures testing environments with Tox.
- **How It Works**: Runs tests across multiple Python versions and dependency sets.
- **Input**: Test commands.
- **Output**: Test results per environment.

## Getting Started

### Prerequisites

- **AWS Account**: Access to SageMaker and S3 with appropriate IAM permissions.
- **Python**: Version 3.6+.
- **Dependencies**:
  - `sagemaker>=2.243.1`
  - `xgboost==1.7.1`
  - Other dependencies in `setup.py`.
- **AWS CLI**: Configured with credentials.
- **Git**: For cloning the repository.

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/chandan-akshronix/Fraud-detection-Mlops-1-ModelBuild.git
   cd Fraud-detection-Mlops-1-ModelBuild
   ```

2. **Install Dependencies**:
   ```bash
   pip install -e .
   ```

### Usage

Run the pipeline with:
```bash
python pipelines/run_pipeline.py --region <aws-region> --role <sagemaker-role-arn>
```

For a step-by-step example, see `sagemaker-pipelines-project.ipynb`.

## Pipeline Workflow

1. **Preprocessing**:
   - **Input**: Raw data from S3.
   - **Process**: Applies transformations and splits data.
   - **Output**: Train, validation, test datasets; preprocessing artifact.

2. **Training**:
   - **Input**: Preprocessed train and validation data.
   - **Process**: Trains an XGBoost model with hyperparameter tuning.
   - **Output**: Trained model artifact.

3. **Batch Transform**:
   - **Input**: Trained model and test data.
   - **Process**: Generates predictions.
   - **Output**: Prediction file.

4. **Evaluation**:
   - **Input**: Predictions and ground truth.
   - **Process**: Computes metrics.
   - **Output**: Evaluation JSON report.

5. **Quality & Bias Checks**:
   - **Input**: Data and model artifacts.
   - **Process**: Runs SageMaker Clarify and Model Monitor checks.
   - **Output**: Quality and bias reports.

6. **Model Registration**:
   - **Input**: Model and evaluation metrics.
   - **Process**: Registers the model if metrics meet thresholds.
   - **Output**: Model package in SageMaker Model Registry.

## Contributing

See `CONTRIBUTING.md` for guidelines on contributing, including pull request processes and coding standards.

## License

This project is licensed under the terms in `LICENSE`.
