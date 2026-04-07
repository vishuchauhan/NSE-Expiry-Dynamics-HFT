# NSE-Expiry-Dynamics-HFT

An autonomous, high-frequency quantitative algorithmic trading stack designed to exploit market microstructure and institutional delta-hedging on the National Stock Exchange (NSE) of India. 

Instead of relying on lagging technical indicators or black-box deep learning, this engine models the market as a **Stochastic Potential Field**. It calculates the "Attractor" ($S^*$)—the mathematical equilibrium point where institutional hedging pressure is minimized—and dynamically executes trades based on the velocity of this shifting gravity well.

## The Alpha (Out-Of-Sample Performance)
Tested blindly on a 3.2 Million row Out-Of-Sample (OOS) high-frequency tick dataset across multiple trading sessions, incorporating strict execution constraints and heavy friction penalties.

* **OOS Win Rate:** `74.19%`
* **Net Return on Capital:** `+12.97%` (3 Trading Days)
* **Execution Model:** 30-Minute Chronological Cooldown (Preventing HFT overfitting)
* **Risk Management:** Dynamic Lot-Sizing via Liquidity-Capped Kelly Criterion

---

## System Architecture (The 3 Pillars)

### 1. The Physics Engine (`preprocess_data.py`)
Extracts raw Option Chain data and calculates the fundamental physical forces of the market:
* **The Attractor ($S^*$):** Computes the true NIFTY index equilibrium using a rolling weighted average of open interest strikes. Includes an automated "NIFTY Isolation Filter" to purge corporate stock anomalies.
* **Attractor Velocity ($v^*$):** Calculates the first derivative of the Attractor to measure how fast market makers are rolling their hedges.
* **Implied Fear Index (Skew):** A volatility proxy indicating retail panic versus institutional control.

### 2. The AI Cortex (`hmm_brain.py`)
An unsupervised **Hidden Markov Model (HMM)** designed for continuous regime detection. It does not predict price; it predicts the *environment*.
* Ingests physical Velocity and Skew.
* Autonomously classifies the market into Hidden States (e.g., Stationary Trap, Institutional Squeeze, Retail Shock).

### 3. The Risk Manager (`risk_manager.py`)
A mathematical safeguard preventing risk of ruin. 
* Utilizes a localized **Kelly Criterion** formula.
* **Dynamic Throttling:** Automatically scales down position sizes (lots) during modeled drawdowns.
* **Liquidity Wall:** Hard-capped at 20 lots to respect real-world NSE order-book depth.
