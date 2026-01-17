import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Prevent GUI errors
import matplotlib.pyplot as plt
import seaborn as sns
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from io import BytesIO
import base64
import warnings
warnings.filterwarnings("ignore")

# --- MODEL IMPORTS WITH SAFETY NETS ---
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

# Try importing XGBoost
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Try importing Prophet
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

# --- KNOWLEDGE BASE ---
SHOCK_EVENTS = {
    2020: "COVID-19 Pandemic (Global Supply Shock)",
    2021: "Post-Pandemic Volatility",
    2022: "Inflation & Geopolitical Conflict",
    2008: "Global Financial Crisis",
    2009: "Great Recession",
    1991: "Balance of Payments Crisis (India)",
    1997: "Asian Financial Crisis",
    2001: "Dot-com Bubble Burst",
    2016: "Demonetization Policy (India)"
}

def get_shock_context(year):
    return SHOCK_EVENTS.get(year, "Structural Deviation")

# --- THE MODEL ARENA ---
class ModelArena:
    def __init__(self, X_train, y_train, X_test, y_test, use_log=False):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.use_log = use_log
        self.results = {}
        self.predictions = {}
        
    def _inverse(self, data):
        return np.exp(data) if self.use_log else data

    def run_native(self):
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        self._store_result("Native (Linear)", preds)

    def run_ai(self):
        # Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        self._store_result("AI (Random Forest)", preds)

    def run_arima(self):
        try:
            # ARIMA requires 1D array
            model = ARIMA(self.y_train, order=(1,1,1))
            model_fit = model.fit()
            preds = model_fit.forecast(steps=len(self.y_test))
            self._store_result("ARIMA (Econometrics)", preds)
        except:
            pass # Skip if convergence fails

    def run_xgboost(self):
        if HAS_XGB:
            model = XGBRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
            model.fit(self.X_train, self.y_train)
            preds = model.predict(self.X_test)
            self._store_result("XGBoost (SOTA)", preds)

    def run_neural_net(self):
        # MLPRegressor as a proxy for Transformers/Deep Learning
        model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        self._store_result("Neural Net (Deep Learning)", preds)

    def run_prophet(self, df_train_prophet, future_periods):
        if HAS_PROPHET:
            try:
                # Prophet needs specific 'ds' and 'y' columns
                m = Prophet(daily_seasonality=False, weekly_seasonality=False)
                m.fit(df_train_prophet)
                forecast = m.predict(future_periods)
                # Extract only the test period predictions
                preds = forecast['yhat'].values[-len(self.y_test):]
                self._store_result("Prophet (Meta)", preds)
            except:
                pass

    def _store_result(self, name, preds):
        # Inverse transform if log logic was applied
        real_preds = self._inverse(preds)
        real_actuals = self._inverse(self.y_test)
        
        # Calculate MAPE
        try:
            mape = mean_absolute_percentage_error(real_actuals, real_preds) * 100
        except:
            mape = 999.9
            
        self.results[name] = round(mape, 2)
        self.predictions[name] = real_preds

# --- MAIN VIEW ---
def dashboard(request):
    context = {}
    
    if request.method == 'POST' and 'dataset' in request.FILES:
        # Phase 1: Upload
        myfile = request.FILES['dataset']
        fs = FileSystemStorage()
        if fs.exists(myfile.name): fs.delete(myfile.name)
        filename = fs.save(myfile.name, myfile)
        file_path = fs.path(filename)
        
        try:
            df = load_smart_data(file_path, filename)
            country_col = next((c for c in df.columns if 'country' in c.lower()), None)
            countries = df[country_col].unique().tolist() if country_col else ['Global/No Country']
            numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if 'year' not in c.lower()]
            cat_cols = [c for c in df.select_dtypes(include=['object']).columns if c != country_col]
            context = {'step': 'configure', 'filename': filename, 'countries': countries, 'variables': numeric_cols, 'sector_cols': cat_cols}
            return render(request, 'core/dashboard.html', context)
        except Exception as e:
            context['error'] = f"Scan Error: {str(e)}"

    elif request.method == 'POST' and 'filename' in request.POST:
        # Phase 2: Analysis
        filename = request.POST['filename']
        selected_country = request.POST.get('country')
        target_var = request.POST.get('target_var')
        sector_col = request.POST.get('sector_col')
        
        fs = FileSystemStorage()
        file_path = fs.path(filename)
        
        try:
            df = load_smart_data(file_path, filename)
            
            # --- DATA PREP ---
            country_col = next((c for c in df.columns if 'country' in c.lower()), None)
            if country_col and selected_country != 'Global/No Country':
                df = df[df[country_col] == selected_country]
            
            year_col = next((c for c in df.columns if 'year' in c.lower()), None)
            df = df[[year_col, target_var] + ([sector_col] if sector_col else [])].dropna()
            df[year_col] = pd.to_numeric(df[year_col])
            df = df.sort_values(year_col)

            if len(df) < 5: raise ValueError("Not enough data points.")

            # --- LOG LOGIC (ROBUSTNESS) ---
            y_raw = df[target_var].values
            X = df[[year_col]].values
            
            # Auto-detect if Log is needed (if mean > 1000 or high variance)
            use_log = (np.mean(y_raw) > 1000) or (np.std(y_raw) > np.mean(y_raw))
            y = np.log(y_raw) if use_log else y_raw

            # Split Data (80/20)
            cutoff = int(len(y) * 0.8)
            X_train, X_test = X[:cutoff], X[cutoff:]
            y_train, y_test = y[:cutoff], y[cutoff:]

            # --- INITIALIZE ARENA ---
            arena = ModelArena(X_train, y_train, X_test, y_test, use_log)
            
            # 1. Run All Models
            arena.run_native()
            arena.run_ai()
            arena.run_arima()
            arena.run_xgboost()
            arena.run_neural_net()
            
            # Prophet Prep (Special formatting)
            if HAS_PROPHET:
                prophet_df = pd.DataFrame({'ds': pd.to_datetime(df[year_col][:cutoff], format='%Y'), 'y': y_train})
                future = pd.DataFrame({'ds': pd.to_datetime(df[year_col][cutoff:], format='%Y')})
                arena.run_prophet(prophet_df, future)

            # --- RESULTS PROCESSING ---
            results = arena.results
            sorted_results = dict(sorted(results.items(), key=lambda item: item[1]))
            champion_model = list(sorted_results.keys())[0]
            champion_score = list(sorted_results.values())[0]

            # --- FORECAST (Using Champion Logic - fallback to robust XGB/RF) ---
            # We refit on FULL data for projection
            full_X = X
            full_y = y
            last_year = int(X[-1][0])
            future_years = np.array([[last_year + i] for i in range(1, 6)])
            
            # Use XGBoost if available, else RF for reliable projection
            final_model = XGBRegressor(n_estimators=500) if HAS_XGB else RandomForestRegressor(n_estimators=100)
            final_model.fit(full_X, full_y)
            forecast_log = final_model.predict(future_years)
            forecast_vals = np.exp(forecast_log) if use_log else forecast_log

            # --- SHOCK DETECTION (Using Trend Line) ---
            lin = LinearRegression()
            lin.fit(full_X, full_y)
            trend_line = lin.predict(full_X)
            residuals = full_y - trend_line
            std_dev = np.std(residuals)
            shock_idx = np.where(np.abs(residuals) > 1.0 * std_dev)[0]
            shocks = [{'year': int(X[i][0]), 'val': y_raw[i], 'reason': get_shock_context(int(X[i][0]))} for i in shock_idx]

            # --- PLOTTING ---
            # 1. Battle of Models Plot
            plt.figure(figsize=(10, 6))
            # Plot Actual Data (Test Set Only for Clarity)
            plt.plot(df[year_col][cutoff:], y_raw[cutoff:], 'k-', linewidth=3, label='Actual Data')
            
            colors = ['#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899']
            for i, (name, preds) in enumerate(arena.predictions.items()):
                # Only plot if we have predictions
                if len(preds) == len(y_test):
                    plt.plot(df[year_col][cutoff:], preds, '--', label=f"{name}", color=colors[i % len(colors)])
            
            plt.title(f"Model Battle: {target_var} ({selected_country})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            battle_plot = get_image_base64(plt)

            # 2. Bar Chart of Errors
            plt.figure(figsize=(8, 4))
            sns.barplot(x=list(results.values()), y=list(results.keys()), palette='viridis')
            plt.xlabel("MAPE Error % (Lower is Better)")
            mape_plot = get_image_base64(plt)

            # 3. Pie Chart (if Sector)
            pie_plot = None
            if sector_col and sector_col != country_col:
                latest_yr = df[year_col].max()
                full_df = load_smart_data(file_path, filename)
                if country_col: full_df = full_df[full_df[country_col] == selected_country]
                sector_data = full_df[full_df[year_col] == latest_yr].groupby(sector_col)[target_var].sum()
                if not sector_data.empty:
                    plt.figure(figsize=(6, 6))
                    plt.pie(sector_data, labels=sector_data.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
                    plt.title(f"Sector Split ({latest_yr})")
                    pie_plot = get_image_base64(plt)
            
            # 4. Histogram
            plt.figure(figsize=(6, 4))
            sns.histplot(y_raw, kde=True, color='#0f172a')
            plt.title("Distribution")
            hist_plot = get_image_base64(plt)

            context = {
                'step': 'results',
                'selected_country': selected_country, 'target_var': target_var,
                'results': sorted_results, 'champion': champion_model, 'champion_score': champion_score,
                'shocks': shocks, 'forecast': zip(future_years.flatten(), forecast_vals),
                'battle_plot': battle_plot, 'mape_plot': mape_plot, 'pie_plot': pie_plot, 'hist_plot': hist_plot
            }

        except Exception as e:
            context['error'] = f"Engine Error: {str(e)}"

    return render(request, 'core/dashboard.html', context)

def load_smart_data(path, filename):
    ext = filename.split('.')[-1].lower()
    if ext == 'csv': return pd.read_csv(path)
    elif ext in ['xlsx', 'xls']: return pd.read_excel(path)
    return pd.DataFrame()

def get_image_base64(plt_obj):
    buffer = BytesIO()
    plt_obj.savefig(buffer, format='png', bbox_inches='tight')
    plt_obj.close()
    return base64.b64encode(buffer.getvalue()).decode('utf-8')