# Ml_pipeline_scoring

Scoring model for credit ratings classification (and more)

## 📁  Detailed Project Structure

MLpipeline_scoring/ \
├── [.venv/](.venv/)                 # Python Virtual Environment \
├── [backend/](backend/)               # Backend Application \
│   ├── [data_explorer.py](backend/data_explorer.py) # Data Analysis    \
│   ├── [main.py](backend/main.py)      # Main Script   \
│   ├── [metrics.py](backend/metrics.py) # Evaluation Metrics   \
│   ├── [split_data.py](backend/split_data.py) # Data Splitting \
│   ├── [train_model.py](backend/train_model.py) # Model Training   \
│   ├── [data/](backend/data/)          # Datasets Folder   \
│   ├── [data_yaml/](backend/data_yaml/) # Data Configuration Files(Optional)  \
│   ├── [mlruns/](backend/mlruns/)      # MLflow Runs Storage   \
│   ├── [models/](backend/models/)      # Models Storage    \
│   ├── [split_yaml/](backend/split_yaml/) # Split Configuration Files  \
│   ├── [yaml_runs/](backend/yaml_runs/) # Training Run Configuration Files \
│   └── [.env](backend/.env)           # Environment Variables  \
├── [.env](.env)                       # Root Directory Environment Variables   \
├── [.gitignore](.gitignore)    \
├── [deployment.txt](deployment.txt)   # Docker Compose Launch Example  \
├── [docker-compose.yml](docker-compose.yml) # Docker Compose Configuration \
├── [Dockerfile](Dockerfile)           # Main Dockerfile    \
├── [Dockerfile.backend](Dockerfile.backend) # Backend Dockerfile   \
├── [Dockerfile.niflow](Dockerfile.niflow) # Workflow Dockerfile    \
├── [README.md](README.md)  \
└── [requirements.txt](requirements.txt) # Python Dependencies  \

## 🚀 Project Launch

### 1.  Launch via Docker Compose (Recommended)

```bash
# Clone and setup
git clone <repository-url>
cd ML_pipeline_scoring

# Build all services
docker compose build 
# Start server 
docker compose up -d 
# Optional (push to local server)
docker push {name_tag_image}
# Optional (pull on local server)
docker pull {name_tag_image}
```

### 2.Services Available (Default):
🎯 MLflow UI: http://localhost:5000

📊 Streamlit: http://localhost:8501

### 2.2.Services Available (Via .env):
🎯 MLflow UI: http://{DEPLOY_HOST}:{DEPLOY_PORT}

📊 Streamlit: http://{DEPLOY_HOST_STREAMLIT}:{DEPLOY_PORT_STREAMLIT}

### 3.1.Local Launch Example (MLFLOW)
```bash
# List experiments
mlflow experiments list

# View specific run
mlflow runs list --experiment-id 0

# Compare runs
mlflow ui --port 5000
```
### 3.2.Local Launch Example (STREAMLIT)
```bash
# Launch with specific settings
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# View logs
streamlit run app.py --logger.level debug
```

### 4. Example Configuration File (run_yaml_test.yaml) for Startup
<details> <summary>📁 Click to view run_yaml_test.yaml configuration</summary>

```
run:
  type_data: test
  type_class: base
  split_type: base #todo adapt to train
  type_class_model: classification

full_data:
  data:
    path: ""     # Path to Excel file
    sheet_name: "rate"                 # Worksheet name (if None - first sheet is taken)
    target: "target"             # Target variable (if exists)
    skiprows: 7
    dropna: False                       # Remove rows with missing values
    skip: "skip_data_quality"
    date_column: "report_date"
    filter_columns:
      - "report_date"
      - "global_ogrn"
      
    mapper:
          aaa: aaa
          aa+: aa
          aa: aa
          aa-: aa
          a+: a
          a: a
          a-: a
          bbb+: bbb
          bbb: bbb
          bbb-: bbb
          bb+: bb
          bb: bb
          bb-: bb
          b+: b
          b: b
          b-: b
          ccc: ccc

  features:
    numeric:
      - "lg(FFO/TT)_value"
      - "lg(TR/TT)_value"
      - "Key_assets_share_indicator"
      - "CapEx_revenue_indicator"
      - "Debt_burden_OIBDA_indicator"
      - "Debt_burden_FFO_indicator"
      - "Debt_service_OIBDA_indicator"
      - "Absolute_liquidity_indicator"
      - "Current_liquidity_indicator"
      - "OIBDA_profitability_indicator"
      - "Autonomy_coefficient_indicator"
      - "CapEx_revenue_indicator"

    categorical:
        - "Company_public_nonpublic_indicator"
        - "OKVED_section_actual"
        - "EMISS_industry"
        - "Market_concentration_level_jan25_GLOB"
        - "GLOB_Market_entry_barriers_significance"
        - "NAT_Market_entry_barriers_significance"
        - "Market_concentration_level_jan25_LOC"
        - "LOC_Market_entry_barriers_significance"
        - "Economic_sectors_share_consumers_goods"
        - "Assortment_diversity"
        - "Number_key_objects_80percent_revenue"
        - "Key_objects_risk_exposure"

model:
  model_name: credit_risk_test_4_base
  framework: sklearn
  ensemble:
    type: base         # possible right now: voting | stacking | bagging | bases
    voting: soft          # for VotingClassifier
    final_estimator: null # for StackingClassifier, example LogisticRegression
    estimators:
      - name: rf
        type: RandomForestClassifier
        params:
          n_estimators: 200
          max_depth: 5

metrics:
    custom:
          ar:
            use_transform: direct
            min_benchmark: 1
            max_benchmark: 7

          psi:
    base:
      - auc
      - f1
      - accuracy
      - mse
      - rmse
      - r2
```

### 5. Example Configuration File (.env in root of directory) for Startup
<details> <summary>📁 Click to view .env configuration</summary>

```
EXT_PORT_WEB=8301
INT_PORT_WEB=8300
EXT_PORT_FLOW=5101
INT_PORT_FLOW=5100
host_server=0.0.0.0

user_name=user
user_pass=pass
host=localhost
port=5432
data_base=database
schema=mlflow_pipeline

```

### 6. Example Configuration File (backend/.env in root of directory) for Startup
<details> <summary>📁 Click to view backend/.env configuration</summary>

```
user_name=user
user_pass=pass
host=localhost
port=5432
data_base=database
schema=mlflow_pipeline
```