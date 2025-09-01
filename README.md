# 📘 Gold Price Forecasting: ARIMA vs. Stacked LSTM vs. LSTM + Attention  

## 🔎 Project Purpose  
The purpose of this project was to rigorously evaluate how different modeling approaches perform in forecasting gold prices. Gold was chosen as a test case because it is both economically significant and statistically challenging: like many financial assets, it exhibits **near-random-walk behavior**. This means tomorrow’s price is best predicted by today’s price plus random noise, leaving little room for exploitable patterns.  

This property makes gold an excellent benchmark for comparing:  
1. **Classical statistical models (ARIMA)** – simple, interpretable, and often surprisingly hard to beat.  
2. **Deep learning architectures (Stacked LSTM, LSTM with Attention)** – powerful but more complex, with the potential to capture nonlinear dependencies.  

The central research question was: **do advanced neural networks meaningfully outperform statistical baselines when forecasting noisy financial series like gold?**  

---

## 📊 Methodology  

### Dataset  
- **14 years of daily gold prices (~3,500 data points).**  
- Preprocessing included scaling and reshaping into sequences for LSTM input.  
- Data split chronologically into **80% training, 20% testing**, simulating real-world forecasting where models are trained on the past and evaluated on unseen future data.  

### Models Compared  
1. **ARIMA(0,1,0): Random Walk Baseline**  
   - Equivalent to saying: *“Tomorrow’s price = Today’s price + error.”*  
   - Provides a rigorous null hypothesis. If a model cannot outperform this, it has little practical value.  
   - Implemented in **static mode**: trained on the training set, then used to forecast the full test horizon.  

2. **Stacked LSTM**  
   - Multi-layer recurrent neural network designed to capture nonlinear temporal dependencies.  
   - Configured with `hidden_size=100`, `num_layers=3`, `epochs=100`.  
   - Well-suited for capturing **short-to-medium lag dependencies**.  

3. **LSTM with Attention**  
   - Extends the LSTM by applying attention weights over past hidden states, allowing the model to “focus” on the most relevant time steps.  
   - Originally underperformed due to poor hyperparameter alignment. Improvements made:  
     - Dropout removed from attention weights.  
     - ReLU activation added before final output.  
   - Same base hyperparameters as stacked LSTM for comparability, though in practice attention often requires separate tuning.  

### Evaluation Metrics  
- **R² (Coefficient of Determination):** How much variance is explained by the model.  
- **RMSE (Root Mean Squared Error):** Penalizes large errors.  
- **MAPE (Mean Absolute Percentage Error):** Percentage-based error measure, useful for interpretability.  
- Metrics are reported only on the **test set (last 20%)** to ensure no leakage from training data.  

---

## 🧠 Why LSTMs and Attention?  

**LSTMs** (Long Short-Term Memory networks) are designed to overcome the vanishing gradient problem of vanilla RNNs. They use input, output, and forget gates to regulate information flow, allowing them to capture patterns across longer horizons.  

- A **stacked LSTM** deepens this capability by layering multiple LSTMs, enabling richer feature extraction at different temporal scales.  
- An **LSTM with Attention** goes further: instead of compressing all history into a single hidden state, it learns to assign weights to each time step. This is especially valuable in contexts where **only certain lags matter** (e.g., seasonality, recurring events, or long-term dependencies).  

In fields like NLP, attention has been transformative. But in financial time series like gold, where signal-to-noise ratio is low and dependencies are mostly local, its value is less certain. Testing attention here directly probes that uncertainty.  

---

## 📈 Results  

| Metric   | ARIMA(0,1,0) | Stacked LSTM | LSTM + Attention |
|----------|--------------|--------------|------------------|
| **R²**   | -0.60        | 0.89         | 0.23             |
| **RMSE** | 124.01       | 0.0319       | 0.0831           |
| **MAPE** | 5.19         | 3.28         | 8.16             |  

### Interpretation  
- **ARIMA(0,1,0):** As expected, produced a flat forecast anchored at the last training value. Negative R² confirmed it explains less variance than simply predicting the mean.  
- **Stacked LSTM:** Achieved strong performance, with R² close to 0.9, capturing short-term fluctuations effectively. Its depth allowed it to model local dependencies without overfitting.  
- **LSTM + Attention:** Initially performed poorly (negative R²), but improved after architectural fixes. Still underperformed the stacked LSTM, likely due to:  
  1. **Gold’s near-random-walk nature:** little long-term structure for attention to exploit.  
  2. **Dataset size (~3,500 points):** relatively small for attention-heavy models, which typically thrive on larger datasets.  
  3. **Hyperparameter tuning:** reusing LSTM settings for attention likely under-optimized it.  

---

## 📝 Lessons Learned  

1. **Statistical Rigor Matters**  
   - Using ARIMA as a baseline ensured the evaluation was grounded.  
   - Any neural network must meaningfully outperform ARIMA to demonstrate value.  

2. **Complexity Isn’t Always Better**  
   - Despite theoretical advantages, attention underperformed stacked LSTM here.  
   - This highlights the importance of aligning model complexity with data complexity.  

3. **Hyperparameters Are Architecture-Specific**  
   - The same settings that worked for stacked LSTM did not translate to the attention model.  
   - Tuning must be tailored to each architecture.  

4. **Interpretability Is Valuable Even Without Accuracy Gains**  
   - Attention weights can still be visualized, offering insights into which time steps the model emphasizes.  
   - This is useful for explainability in domains like finance, even if predictive performance is weaker.  

---

## 🎯 Conclusion  

This project demonstrates that rigorous benchmarking is as important as model building. By comparing ARIMA, stacked LSTM, and LSTM with attention, we showed:  

- **ARIMA(0,1,0)** provides a realistic null hypothesis but cannot capture nonlinear dependencies.  
- **Stacked LSTM** effectively models short-run dynamics, achieving the best performance.  
- **Attention-enhanced LSTM** improved after fixes but still underperformed, underscoring that more complex models are not inherently better.  

---
