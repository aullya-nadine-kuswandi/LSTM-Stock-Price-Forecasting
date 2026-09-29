# Stock Price Forecasting with LSTM (FB & IBM)

One-step-ahead forecasting of daily closing prices for **Facebook (FB)** and **IBM** using LSTM networks. The project builds a baseline LSTM, improves it with more capacity and a non-linear Dense head, then runs hyperparameter search with Keras Tuner. Models are evaluated on the last year of data with MAPE, RMSE, and MAE.

**Result:** the modified LSTM (85 units + Dense head) performed best, cutting FB MAPE from **2.32% to 1.66%** (about 28% relative) and slightly improving IBM (1.34% to 1.33%).

## Key Objectives

- **EDA:** Inspect date ranges, descriptive statistics, missing values, duplicates, price distributions, and 30/90-day moving averages for both stocks.
- **Chronological Splitting:** Use the last year of data as the test set, and the final 10% of the training windows (in time order) as validation. No random shuffling, to avoid leaking future information.
- **Preprocessing:** Use only the `Close` price, scale with `MinMaxScaler` fitted on training data only, and build sliding windows (5 days as input, next day as target).
- **Baseline LSTM:** A single LSTM layer followed by a Dense output.
- **Modified LSTM:** More LSTM units and an extra Dense layer for non-linear transformation.
- **Hyperparameter Tuning:** Random search over LSTM units, Dense units, and activation function, selected by validation loss (not test metrics).
- **Evaluation:** Compare all models on the test set with MAPE, RMSE, and MAE, in original price scale.

## Dataset

Daily historical prices (Date, Open, High, Low, Close, Adj Close, Volume). Only `Close` is used.

| Stock | Period | Rows |
|-------|--------|------|
| FB | 2012-05-18 to 2020-04-01 | 1,980 |
| IBM | 1962-01-02 to 2020-04-01 | 14,663 |

**Split (cutoff 2019-04-01)**

| | FB windows | IBM windows |
|---|-----------|-------------|
| Train | 1,549 | 12,964 |
| Validation | 173 | 1,441 |
| Test (last 1 year) | 248 | 248 |

**Windowing:** window size = 5 days (Monday to Friday), horizon = 1 day ahead.

## Models

### 1. Baseline
- LSTM (50 units, ReLU) → Dense 1
- 10,451 parameters

### 2. Modified
- LSTM (85 units, ReLU) → Dense 8 (ReLU) → Dense 1

### 3. Tuned (Keras Tuner, Random Search, 18 trials)
Search space: LSTM units {64, 96, 128}, LSTM activation {ReLU, tanh}, Dense units {8, 16, 32}.

| Stock | Best configuration |
|-------|--------------------|
| FB | LSTM 64, tanh, Dense 32 |
| IBM | LSTM 128, tanh, Dense 16 |

## Training Setup

- Loss: MSE, optimizer: Adam, batch size 32
- Early stopping on validation loss with best-weights restore
  - Baseline: up to 100 epochs, patience 10
  - Modified: up to 150 epochs, patience 8
  - Tuned: learning rate 1e-3, up to 100 epochs, patience 10

## Results

Test set metrics (original price scale, USD):

**Facebook (FB)**

| Model | MAPE (%) | RMSE | MAE |
|-------|----------|------|-----|
| Baseline | 2.3184 | 5.5445 | 4.3408 |
| **Modified** | **1.6625** | **4.1831** | **3.0947** |
| Tuned | 2.8881 | 6.5223 | 5.4858 |

**IBM**

| Model | MAPE (%) | RMSE | MAE |
|-------|----------|------|-----|
| Baseline | 1.3377 | 2.6901 | 1.7379 |
| **Modified** | **1.3272** | **2.6659** | **1.7199** |
| Tuned | 2.2022 | 4.1274 | 2.8904 |

## Key Findings

- **Baseline:** loss dropped very quickly, which suggests the model mostly learned *persistence* (tomorrow's price is close to today's) rather than richer temporal patterns.
- **Modified model:** added capacity and the Dense head gave a clear improvement on FB (MAPE down about 28% relative, RMSE down 1.36, MAE down 1.25) with a small gap between train and validation loss. On IBM the gain was tiny, because the baseline was already strong with much more training data.
- **Tuned model:** the configuration with the best validation loss did **not** generalize best. It was worse than the baseline on both stocks, showing that the best validation score does not always give the best test performance in time series, where behavior differs across periods.
- **FB vs IBM:** IBM's lower MAPE does not mean the model is better on IBM. IBM's price moved more steadily during the test period, so relative errors are naturally smaller. RMSE and MAE are also not comparable across stocks because their price ranges differ.

## Conclusion

A modestly larger LSTM with a small non-linear Dense head was the best choice here, giving a meaningful improvement on the more volatile FB series and a marginal one on IBM. Larger architecture and validation-based tuning did not automatically translate to better test performance, so the simpler modified model was selected as the final model for both stocks.

## Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras, Keras Tuner
- **Metrics & Preprocessing:** scikit-learn (MinMaxScaler, MAPE, RMSE, MAE)
- **Data Handling & Visualization:** Pandas, NumPy, Matplotlib, Seaborn
- **Environment:** Kaggle Notebooks
