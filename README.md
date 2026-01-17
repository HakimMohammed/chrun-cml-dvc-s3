# 🏦 Customer Churn Prediction ML Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![DVC](https://img.shields.io/badge/DVC-3.0+-orange.svg)](https://dvc.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-green.svg)](https://scikit-learn.org/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF.svg)](https://github.com/features/actions)
[![CML](https://img.shields.io/badge/CML-Continuous%20ML-blueviolet.svg)](https://cml.dev/)

A comprehensive Machine Learning project for predicting customer churn using bank customer data, with integrated data version control, automated CI/CD pipeline, and advanced techniques for handling imbalanced datasets.

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Data Processing Pipeline](#-data-processing-pipeline)
- [Model Training](#-model-training)
- [DVC Configuration](#-dvc-configuration)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Setup](#local-setup)
  - [GitHub Secrets Setup](#github-secrets-setup)
- [Model Outputs](#-model-outputs)
- [Workflow](#-workflow)
- [License](#-license)
- [Contributing](#-contributing)

## 🎯 Project Overview

This project implements a customer churn prediction system for a bank, utilizing machine learning to identify customers likely to exit. The project demonstrates best practices in ML operations, including:

- **ML Framework**: scikit-learn with RandomForestClassifier
- **Data Version Control**: DVC with AWS S3 backend for data and model versioning
- **CI/CD**: GitHub Actions with CML (Continuous Machine Learning) for automated training and reporting
- **Data Challenge**: Handling imbalanced datasets using three different approaches:
  1. Baseline (without considering imbalance)
  2. Using class weights
  3. Using SMOTE oversampling (sampling_strategy=0.7)

The project uses bank customer data to predict customer churn (binary classification on the `Exited` target variable), comparing three different strategies to handle the inherent class imbalance in the dataset.

## ✨ Features

- ✅ **Data version control** with DVC and S3
- ✅ **Automated ML pipeline** with GitHub Actions
- ✅ **CML integration** for model reporting and metrics visualization
- ✅ **Imbalanced data handling** with 3 different strategies
- ✅ **Comprehensive preprocessing pipeline** with numerical and categorical feature handling
- ✅ **Model artifact versioning** for reproducibility
- ✅ **Automated metrics tracking** with F1-score evaluation
- ✅ **Visual performance reports** with confusion matrices

## 🛠 Tech Stack

### Core Dependencies

Based on `requirements.txt`:

```
scikit-learn==1.3.2          # Machine learning algorithms
sklearn_features             # Custom transformers for pipeline
matplotlib>=3.7.0            # Plotting and visualization
seaborn>=0.12.0             # Statistical data visualization
pandas>=2.0.0               # Data manipulation and analysis
numpy>=1.24.0               # Numerical computing
dvc[s3]>=3.0.0              # Data version control with S3 support
joblib>=1.3.0               # Model serialization
imbalanced-learn>=0.11.0    # SMOTE and imbalanced data techniques
Pillow>=10.0.0              # Image processing
```

### Infrastructure

- **Version Control**: Git & GitHub
- **Data Versioning**: DVC
- **Cloud Storage**: AWS S3
- **CI/CD**: GitHub Actions
- **ML Reporting**: CML (Continuous Machine Learning)

## 📁 Project Structure

```
.
├── .dvc/                    # DVC configuration
│   └── config              # DVC remote config (S3: s3://hakim-dvc-bucket)
├── .github/
│   └── workflows/
│       └── s3-dvc.yml      # CI/CD pipeline with CML
├── data/                    # Data directory (tracked by DVC)
│   └── dataset.csv         # Customer churn dataset
├── models/                  # Model artifacts (tracked by DVC)
│   └── RandomForestClassifier.pkl  # Trained model
├── data.dvc                # DVC file tracking data directory
├── models.dvc              # DVC file tracking models directory
├── script.py               # Main ML training script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 🔄 Data Processing Pipeline

The data preprocessing pipeline in `script.py` includes the following steps:

### 1. Data Loading and Cleaning
- Loads data from `data/dataset.csv`
- **Drops irrelevant columns**: `RowNumber`, `CustomerId`, `Surname`
- **Filters outliers**: Removes records where `Age > 80`

### 2. Feature Engineering

The pipeline processes three types of features:

#### Numerical Features
- **Features**: `Age`, `CreditScore`, `Balance`, `EstimatedSalary`
- **Preprocessing**:
  - Median imputation for missing values
  - StandardScaler for normalization

#### Categorical Features
- **Features**: `Gender`, `Geography`
- **Preprocessing**:
  - Most frequent imputation for missing values
  - OneHotEncoder (drop='first') for encoding

#### Ready Features
- **Features**: All remaining features (e.g., `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `Tenure`)
- **Preprocessing**:
  - Most frequent imputation for missing values
  - No additional transformation needed

### 3. Train/Test Split
- **Split ratio**: 80/20
- **Random state**: 45
- **Stratification**: Applied on target variable to maintain class distribution
- **Target variable**: `Exited` (binary classification: 0 = stayed, 1 = churned)

## 🤖 Model Training

### Algorithm Configuration
- **Model**: RandomForestClassifier
- **Parameters**:
  - `n_estimators=500`: Number of trees in the forest
  - `max_depth=10`: Maximum depth of trees
  - `random_state=45`: For reproducibility

### Evaluation Metric
- **Primary metric**: F1-score (harmonic mean of precision and recall)
- Suitable for imbalanced datasets as it balances both false positives and false negatives

### Three Training Variants

The project compares three approaches to handle class imbalance:

#### 1. Baseline (without-imbalance)
```python
# No special handling for imbalance
train_model(X_train=X_train_final, y_train=y_train, 
            plot_name='without-imbalance', class_weight=None)
```

#### 2. Class Weights (with-class-weights)
```python
# Automatically adjust weights inversely proportional to class frequencies
train_model(X_train=X_train_final, y_train=y_train, 
            plot_name='with-class-weights', class_weight=dict_weights)
```
- Calculates class weights based on class distribution
- Normalizes weights to emphasize minority class

#### 3. SMOTE Oversampling (with-SMOTE)
```python
# Synthetic Minority Over-sampling Technique
over = SMOTE(sampling_strategy=0.7)
X_train_resampled, y_train_resampled = over.fit_resample(X_train_final, y_train)
train_model(X_train=X_train_resampled, y_train=y_train_resampled, 
            plot_name='with-SMOTE', class_weight=None)
```
- Creates synthetic samples for minority class
- `sampling_strategy=0.7`: Minority class will be 70% of majority class after resampling

### Model Artifacts
- Models are saved as `.pkl` files using joblib
- Confusion matrix visualizations generated for each approach
- Combined visualization comparing all three approaches

## 📊 DVC Configuration

### Remote Storage
- **Storage**: AWS S3 bucket
- **Bucket**: `s3://hakim-dvc-bucket`
- **Configuration**: Defined in `.dvc/config`

### Tracked Artifacts
1. **Data directory** (`data/`)
   - Tracked by `data.dvc`
   - Contains customer dataset
   
2. **Models directory** (`models/`)
   - Tracked by `models.dvc`
   - Contains trained model files

### Required AWS Credentials

These credentials should be configured for DVC to access S3:
- `ACCESS_KEY_ID`: AWS access key
- `SECRET_ACCESS_KEY`: AWS secret key
- `REGION`: AWS region (e.g., us-east-1)

### DVC Commands
```bash
# Pull data and models from S3
dvc pull

# Track new changes
dvc add data/ models/

# Push changes to S3
dvc push
```

## 🚀 CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/s3-dvc.yml`) automates the entire ML pipeline:

### Trigger
- Runs on every `push` to the repository

### Workflow Steps

1. **Checkout Repository**
   - Uses `actions/checkout@v4`

2. **Setup Python Environment**
   - Installs Python 3.11
   - Uses `actions/setup-python@v5`

3. **Install CML Utility**
   - Sets up CML for ML reporting
   - Uses `iterative/setup-cml@v2`

4. **Install Dependencies**
   - Updates pip
   - Installs packages from `requirements.txt`

5. **Configure DVC Remote**
   - Configures AWS S3 credentials from GitHub secrets
   - Sets up DVC remote access

6. **Pull Data & Models**
   - Downloads latest data and models from S3
   - Command: `dvc pull -v`

7. **Train/Evaluate Model**
   - Executes training script
   - Command: `python script.py`

8. **Track New Artifacts**
   - Adds new/updated artifacts to DVC
   - Commits DVC metadata files
   - Uses `[Skip CI]` to prevent recursive triggers

9. **Push to S3**
   - Uploads new artifacts to S3
   - Command: `dvc push -v`

10. **Publish CML Report**
    - Generates markdown report with:
      - F1-scores for all three approaches
      - Confusion matrix visualizations
    - Posts report as GitHub comment

### Required GitHub Secrets

Configure these in your repository settings (Settings → Secrets and variables → Actions):

| Secret Name | Description |
|------------|-------------|
| `ACCESS_KEY_ID` | AWS access key ID |
| `SECRET_ACCESS_KEY` | AWS secret access key |
| `REGION` | AWS region (e.g., us-east-1) |

## 🚦 Getting Started

### Prerequisites

- **Python**: Version 3.11 or higher
- **AWS Account**: With S3 bucket access
- **Git**: For version control
- **DVC**: Installed (included in requirements.txt)
- **AWS Credentials**: Access key and secret key with S3 permissions

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/HakimMohammed/chrun-cml-dvc-s3.git
   cd chrun-cml-dvc-s3
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure DVC remote**
   
   Replace the placeholders with your actual AWS credentials:
   ```bash
   dvc remote modify --local s3remote access_key_id YOUR_ACCESS_KEY
   dvc remote modify --local s3remote secret_access_key YOUR_SECRET_KEY
   dvc remote modify --local s3remote region YOUR_REGION
   ```

5. **Pull data and models from S3**
   ```bash
   dvc pull
   ```

6. **Run the training script**
   ```bash
   python script.py
   ```

   This will:
   - Load and preprocess the data
   - Train three models with different imbalance handling strategies
   - Generate metrics in `metrics.txt`
   - Create confusion matrix visualizations
   - Save trained models to `models/` directory

### GitHub Secrets Setup

For the CI/CD pipeline to work automatically:

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add the following secrets:

   - **Name**: `ACCESS_KEY_ID`  
     **Value**: Your AWS access key ID

   - **Name**: `SECRET_ACCESS_KEY`  
     **Value**: Your AWS secret access key

   - **Name**: `REGION`  
     **Value**: Your AWS region (e.g., `us-east-1`)

## 📈 Model Outputs

After running the training script, the following outputs are generated:

### Trained Models
- **Location**: `models/RandomForestClassifier.pkl`
- **Format**: Joblib pickle file
- **Note**: All three approaches overwrite the same file (last one is SMOTE-based)

### Metrics File
- **Location**: `metrics.txt`
- **Content**: F1-scores for training and validation sets for all three approaches
- **Format**:
  ```
  RandomForestClassifier without-imbalance
  F1-score of Training is: XX.XX %
  F1-Score of Validation is: XX.XX %
  ----------------------------------------
  RandomForestClassifier with-class-weights
  F1-score of Training is: XX.XX %
  F1-Score of Validation is: XX.XX %
  ----------------------------------------
  RandomForestClassifier with-SMOTE
  F1-score of Training is: XX.XX %
  F1-Score of Validation is: XX.XX %
  ----------------------------------------
  ```

### Visualizations
- **Combined confusion matrix**: `conf_matrix.png`
  - Side-by-side comparison of all three approaches
  - Shows prediction performance on test set
  - Saved at 300 DPI for high quality
  
- **Individual confusion matrices** (temporary):
  - `without-imbalance.png`
  - `with-class-weights.png`
  - `with-SMOTE.png`
  - These are deleted after being combined into `conf_matrix.png`

## 🔄 Workflow

The complete ML workflow operates as follows:

1. **Data Storage & Versioning**
   - Raw data is stored in the `data/` directory
   - DVC tracks the data and stores it in S3
   - Data versions are managed through DVC metadata files

2. **Code Push Trigger**
   - Developer pushes code changes to GitHub
   - GitHub Actions workflow is automatically triggered

3. **Environment Setup**
   - GitHub Actions sets up Python environment
   - Installs all required dependencies
   - Configures DVC with AWS credentials

4. **Data & Model Retrieval**
   - DVC pulls the latest data and models from S3
   - Ensures training uses the most recent versions

5. **Model Training**
   - Training script (`script.py`) executes
   - Three models are trained with different imbalance handling strategies:
     - Baseline without imbalance consideration
     - Class weights approach
     - SMOTE oversampling approach

6. **Evaluation & Comparison**
   - F1-scores calculated for all approaches
   - Confusion matrices generated for visual analysis
   - Metrics saved to `metrics.txt`

7. **Artifact Management**
   - New models are saved to `models/` directory
   - DVC tracks the changes
   - Artifacts are pushed to S3 for versioning

8. **Reporting**
   - CML generates a comprehensive report
   - Report includes F1-scores and confusion matrices
   - Posted as a comment on the commit/PR for easy review

9. **Version Control**
   - DVC metadata files (`.dvc` files) are committed to Git
   - Actual data/model files remain in S3
   - Full reproducibility of any version

## 📚 Additional Resources

- [DVC Documentation](https://dvc.org/doc) - Data Version Control guide
- [CML Documentation](https://cml.dev/) - Continuous Machine Learning
- [scikit-learn Documentation](https://scikit-learn.org/stable/) - Machine learning in Python
- [imbalanced-learn Documentation](https://imbalanced-learn.org/) - SMOTE and imbalanced datasets
- [GitHub Actions Documentation](https://docs.github.com/en/actions) - CI/CD automation

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Commit your changes**
   ```bash
   git commit -m "Add your commit message"
   ```
5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Open a Pull Request**

### Contribution Guidelines

- Follow the existing code style and structure
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR
- Write clear commit messages

---

**Made with ❤️ for ML Operations**
