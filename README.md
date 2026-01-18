# Macro-Economic Intelligence Dashboard (Winter Analytics)

## Project Overview
This project is a **DSGE-inspired Economic Analytics Engine** designed to analyze how productivity changes and macroeconomic shocks affect industrial sectors over time. 

Built on **Django**, it serves as a "Hybrid" forecasting tool that combines classical econometrics (ARIMA, Linear Trends) with modern Machine Learning (Random Forest, XGBoost) to identify **Economic Shocks**, calculate **Sector Resilience**, and forecast future output.

---

## Objectives
Based on the Winter Analytics Problem Statement, this tool aims to:
1.  **Identify Economic Shocks:** Detect significant deviations (Booms/Busts) in productivity using statistical volatility thresholds (DSGE intuition).
2.  **Analyze Sectoral Resilience:** Determine which sectors (e.g., Manufacturing vs. Services) recover fastest from shocks like the 2008 Crisis.
3.  **Forecast Trends:** Compare simple linear benchmarks against complex AI models to find the most "economically interpreted" forecast.
4.  **Distinguish Trend vs. Cycle:** Separate long-run growth trends from short-term cyclical fluctuations.

---

## Features & Capabilities

This application is built with a **Hybrid Architecture**. It detects its environment and adjusts its capabilities accordingly.

### Core Features (Works on Website & Laptop)
These features are optimized for speed and work on the deployed Render website as well as locally.
* **Smart Data Ingestion:** Auto-detects "Country", "Year", and "Sector" columns from messy CSV/Excel files (ILO, Penn World Table, etc.).
* **Global vs. Local Analysis:**
    * *All Countries:* Ranks nations against each other.
    * *Single Country:* Ranks internal sectors (e.g., Agriculture vs. Tech).
* **DSGE Shock Detection:** Automatically flags years of "Contraction", "Expansion", and "Pandemic Shocks" (2020).
* **Sector Leaderboard:** Ranks sectors by **Growth vs. Volatility** (Risk-Reward profile).
* **Base Forecasting Arena:**
    * Naive Persistence (Benchmark).
    * Linear Regression (Macro-Augmented).
    * ARIMA (Python Statsmodels).
    * Random Forest Regressor.
* **Visualizations:** Dynamic Interactive Charts, Pie Charts (Top 10 Sectors), and Distribution Histograms.
* **CSV Export:** Download model performance metrics.

### Laptop/Local Exclusive Features (Power Mode)
These features require heavy computational resources or specific system libraries (like R). They automatically activate when running on a local machine but are disabled on the cloud to prevent timeouts.
* **R-Bridge (Auto.Arima):** Connects Python to **R's `forecast` package** to run superior statistical ARIMA models.
* **Facebook Prophet:** Runs the advanced Prophet additive regression model for seasonality detection.
* **XGBoost (SOTA):** Runs full Gradient Boosting analysis (often restricted on free cloud tiers).

---

## Installation Guide (How to Run Locally)

If you are pulling this code to a new laptop, follow these steps to set up the "Power Mode" environment.

### 1. Prerequisites (Software to Install)
* **Python 3.10+**: [Download Here](https://www.python.org/downloads/)
* **Git**: [Download Here](https://git-scm.com/downloads)
* **(Optional) R Language**: Required only if you want the "R-Bridge" feature. [Download Here](https://cran.r-project.org/). 
    * *Note:* After installing R, open R studio or terminal and run: `install.packages("forecast")`

### 2. Clone and Setup
Open your terminal (Command Prompt or PowerShell) and run:

```bash
# 1. Clone the repository
git clone <YOUR_REPO_URL_HERE>
cd winter_project

# 2. Create a Virtual Environment (Recommended)
python -m venv venv

# 3. Activate the Virtual Environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install Python Dependencies
pip install -r requirements.txt
