# NVDA Stock Trading Model - Knowledge Disc. & Data Mining

## Project Overview

This is the repository for an ML Trading Agent project for the Knowledge Discovery & Data Mining course. We apply machine learning techniques to predict Nvidia's future stock price, and submit trade orders for a real-time trading simulation environment.

Our implementation consists of two Python notebooks run in series:

1. **DataBook_CEFLANN.ipynb**: Handles data extraction, preprocessing, and feature engineering. 
2. **Model_CEFLANN_new.ipynb**: Implements the CEFLANN model, training, evaluation, and trading simulation. 

Running both notebooks in the correct order will show the training results and NVIDIA price predictions.

## Data Preparation (DataBook_CEFLANN.ipynb)

This notebook handles all aspects of data preparation for the CEFLANN model:

### Key Components:
- **Data Acquisition**: Downloads stock data for NVIDIA, Taiwan Semiconductors, and Invesco QQQ from 2020 to 2025 using the Yahoo Finance API
- **Technical Indicators**: Calculates six technical indicators used in the CEFLANN model:
  - Simple Moving Average (MA15)
  - Moving Average Convergence Divergence (MACD26)
  - Stochastic Oscillator K14
  - Stochastic Oscillator D3
  - Relative Strength Index (RSI14)
  - Larry Williams R% (WR14)
- **Data Normalization**: Normalizes indicators using Min-Max normalization
- **Trend Classification**: Identifies uptrend and downtrend sections based on price-MA15 relationship
- **Training Signal Generation**: Creates trading signals in range 0-1 based on trend
- **Data Splitting**: Splits data into training, testing, and evaluation sets by trading week

### Output Files:
- `data/nvidia_technical_data.csv`: Raw technical indicators
- `data/nvidia_normalized_data.csv`: Normalized indicator values
- `data/nvidia_ceflann_data.csv`: Data with trading signals for model training
- Various NumPy files for training, testing, and evaluation datasets, stored in the `data/numpy/` folder

## Model Implementation (Model_CEFLANN_new.ipynb)

This notebook implements the CEFLANN model as described in the research paper "A hybrid stock trading framework integrating technical analysis with machine learning" (Dash & Dash, 2016):

### Key Components:

#### CEFLANN Model
- **Functional Expansion Block**: Transforms input features using tanh activation
- **Training Algorithm**: Regularized least squares (Ridge Regression)
- **Hyperparameter Tuning**: Optimization for expansion order and regularization parameters
- **Cross-Validation**: Time series cross-validation for model evaluation

#### Trading Systems
- **Trading Decision System**: Converts model predictions to BUY/SELL/HOLD decisions
- **Order Generator**: Determines quantity of shares to buy or sell based on signal strength
- **Trading Simulation**: Simulates trading with initial capital and transaction costs

### Workflow Functions:
- `run_trading_pipeline()`: Complete prediction and trading process
- `find_best_hyperparameters()`: Optimizes model parameters
- `train_final_model()`: Trains model with optimal parameters
- `ceflann_workflow()`: High-level workflow with different operation modes:
  - `quick_eval`: Single week evaluation
  - `optimize`: Train and save optimal model
  - `comprehensive`: Complete evaluation on multiple weeks
  - `predict`: Future week prediction

### Visualization Functions:
- `plot_predictions_vs_actual()`: Compare predictions with actual values
- `plot_portfolio_performance()`: Visualize trading performance
- `plot_error_histogram()`: Analyze prediction errors
- `plot_learning_curve()`: Visualize training vs. validation performance

## Usage Steps 

1. **Data Preparation**:
   - Run `DataBook_CEFLANN.ipynb` to download and prepare data
   - Review output files to ensure proper data generation

2. **Model Training and Evaluation**:
   - Run `Model_CEFLANN_new.ipynb` to train and evaluate the model
   - Choose appropriate workflow operation:
     - `quick_eval` for quick evaluation on a recent week
     - `optimize` to find optimal hyperparameters
     - `comprehensive` for thorough evaluation across multiple weeks
     - `predict` to forecast the next trading week (March 24-28, 2025)

3. **Interpreting Results**:
   - Review model accuracy metrics (MSE, RMSE, R²)
   - Analyze trading performance (profit, number of trades)
   - Examine price forecasts and confidence intervals


## Citation

R. Dash and P. K. Dash, “A hybrid stock trading framework integrating technical analysis with machine learning techniques,” The Journal of Finance and Data Science, vol. 2, no. 1, pp. 42–57, Mar. 2016, doi: 10.1016/j.jfds.2016.03.002.
  
