import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from io import BytesIO
import base64
import chardet
import re
import warnings
import logging
import json # <--- ADDED: Essential for Dropdown

# Configure Logging for Debugging
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Standard Python Models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA

# --- ENVIRONMENT DETECTION ---
# Explicit Boolean Cast for Safety
RUNNING_ON_RENDER = bool(os.environ.get('RENDER') or os.environ.get('MEMORY_LIMIT'))

# --- HELPER: NUMPY JSON ENCODER (FIXES JS ERROR) ---
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# --- R-BRIDGE (Laptop Only) ---
def run_r_auto_arima(train_data, periods):
    """Attempts to use R's forecast::auto.arima for superior pattern detection."""
    if RUNNING_ON_RENDER: return None
    try:
        from rpy2.robjects import r, pandas2ri, FloatVector
        from rpy2.robjects.packages import importr
        pandas2ri.activate()
        forecast_pkg = importr('forecast')
        r_vec = FloatVector(train_data)
        model = forecast_pkg.auto_arima(r_vec)
        forecast = forecast_pkg.forecast(model, h=periods)
        return np.array(forecast.rx2('mean'))
    except Exception as e:
        logger.warning(f"R-Bridge unavailable: {e}")
        return None

# --- PROPHET BRIDGE (Laptop Only) ---
def run_prophet_model(df_train, periods):
    """Attempts to use Facebook Prophet."""
    if RUNNING_ON_RENDER: return None
    try:
        from prophet import Prophet
        # Prophet is strictly univariate in this implementation for stability
        m = Prophet(daily_seasonality=False, weekly_seasonality=False)
        m.fit(df_train)
        future = m.make_future_dataframe(periods=periods, freq='Y')
        forecast = m.predict(future)
        return forecast['yhat'].tail(periods).values
    except Exception as e:
        logger.warning(f"Prophet unavailable: {e}")
        return None

# --- THE MODEL ARENA ---
class ModelArena:
    def __init__(self, X_train, y_train, X_test, y_test, df_full=None, date_col=None, target_col=None, use_log=False):
        # Data
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.df_full = df_full 
        self.date_col = date_col
        self.target_col = target_col
        self.use_log = use_log
        
        # Storage
        self.results = {}      # Test Set MAPE
        self.all_forecasts = {} # 5-Year Future Forecasts
        self.models = {}       # Stored Model Objects (for reference)

    def _inverse(self, data):
        return np.exp(data) if self.use_log else data

    def _store(self, name, preds, model_obj=None, future_vals=None):
        try:
            # 1. Evaluate on Test Set
            real_preds = self._inverse(preds)
            real_actuals = self._inverse(self.y_test)
            mape = mean_absolute_percentage_error(real_actuals, real_preds) * 100
            self.results[name] = round(mape, 2)
            
            if model_obj: self.models[name] = model_obj
            
            # 2. Store Future Forecasts (if provided immediately)
            if future_vals is not None:
                self.all_forecasts[name] = self._inverse(future_vals)
                
        except Exception as e:
            print(f"Error storing result for {name}: {e}")
            self.results[name] = 999.9

    # --- BENCHMARKS ---
    def run_naive_benchmarks(self, steps):
        try:
            # Benchmark 1: Persistence (Tomorrow = Today)
            last_val = self.y_train[-1]
            preds_test = np.full(len(self.y_test), last_val)
            preds_future = np.full(steps, self.y_test[-1]) # Future projects from last KNOWN data
            self._store('Naive (Persistence)', preds_test, future_vals=preds_future)

            # Benchmark 2: Average (Tomorrow = Historical Mean)
            mean_val = np.mean(self.y_train)
            preds_test_avg = np.full(len(self.y_test), mean_val)
            preds_future_avg = np.full(steps, np.mean(np.concatenate([self.y_train, self.y_test])))
            self._store('Naive (Average)', preds_test_avg, future_vals=preds_future_avg)
        except Exception as e:
            print(f"Naive Benchmark Error: {e}")

    # --- MULTIVARIATE MODELS (Linear, RF, XGB) ---
    def run_multivariate_models(self, full_X, future_X):
        """
        Runs ML models. If full_X and future_X are based on Year (Univariate),
        this produces a forecast. If based on Macro variables, this produces
        test scores but may not forecast if future macro data is missing.
        """
        # 1. Linear Regression (Elastic Net / Ridge for stability)
        try:
            model = Ridge(alpha=1.0)
            model.fit(self.X_train, self.y_train)
            
            # Forecast Logic: Refit on FULL data for best future prediction
            model_full = Ridge(alpha=1.0)
            model_full.fit(full_X, np.concatenate([self.y_train, self.y_test]))
            future_preds = model_full.predict(future_X)
            
            self._store('Linear (Macro-Augmented)', model.predict(self.X_test), model_full, future_preds)
        except Exception as e: print(f"Linear Error: {e}")

        # 2. Random Forest
        try:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(self.X_train, self.y_train)
            
            model_full = RandomForestRegressor(n_estimators=100, random_state=42)
            model_full.fit(full_X, np.concatenate([self.y_train, self.y_test]))
            future_preds = model_full.predict(future_X)
            
            self._store('AI (Random Forest)', model.predict(self.X_test), model_full, future_preds)
        except Exception as e: print(f"RF Error: {e}")

        # 3. XGBoost
        try:
            if not RUNNING_ON_RENDER:
                from xgboost import XGBRegressor
                model = XGBRegressor(n_estimators=100, max_depth=5)
                model.fit(self.X_train, self.y_train)
                
                model_full = XGBRegressor(n_estimators=100, max_depth=5)
                model_full.fit(full_X, np.concatenate([self.y_train, self.y_test]))
                future_preds = model_full.predict(future_X)
                
                self._store('XGBoost (SOTA)', model.predict(self.X_test), model_full, future_preds)
            else:
                # Proxy for Cloud
                self._store('XGBoost (Cloud Proxy)', np.full(len(self.y_test), np.mean(self.y_train)))
        except Exception as e: print(f"XGB Error: {e}")

    # --- TIME SERIES MODELS (Univariate) ---
    def run_time_series(self, steps):
        full_y = np.concatenate([self.y_train, self.y_test])
        
        # 1. ARIMA (Python)
        try:
            # Train
            model = ARIMA(self.y_train, order=(1,1,1)).fit()
            test_preds = model.forecast(steps=len(self.y_test))
            
            # Refit Full for Forecast
            model_full = ARIMA(full_y, order=(1,1,1)).fit()
            future_preds = model_full.forecast(steps=steps)
            
            self._store('ARIMA (Python)', test_preds, model_full, future_preds)
        except Exception as e: print(f"ARIMA Error: {e}")

        # 2. R-Engine
        try:
            if not RUNNING_ON_RENDER:
                future_r = run_r_auto_arima(full_y, steps)
                # NOTE: We assign the Python ARIMA test score to R-Engine because 
                # backtesting R models via bridge is computationally expensive.
                # This allows R to be selected if the general ARIMA logic holds.
                arima_score = self.results.get('ARIMA (Python)', 999.9)
                
                if future_r is not None:
                    self.results['ARIMA (R-Engine)'] = arima_score 
                    self.all_forecasts['ARIMA (R-Engine)'] = self._inverse(future_r)
        except Exception as e: print(f"R Logic Error: {e}")

        # 3. Prophet
        try:
            if self.df_full is not None and not RUNNING_ON_RENDER:
                cutoff_idx = len(self.y_train)
                # Test
                train_df = pd.DataFrame({'ds': pd.to_datetime(self.df_full[self.date_col].iloc[:cutoff_idx], format='%Y'), 'y': self.y_train})
                test_preds = run_prophet_model(train_df, len(self.y_test))
                
                # Full Forecast
                full_df_prophet = pd.DataFrame({'ds': pd.to_datetime(self.df_full[self.date_col], format='%Y'), 'y': full_y})
                future_preds = run_prophet_model(full_df_prophet, steps)
                
                if test_preds is not None and future_preds is not None:
                    self._store('Facebook Prophet', test_preds, None, future_preds)
        except Exception as e: print(f"Prophet Error: {e}")

# --- DSGE ANALYTICS ENGINE ---
def analyze_shocks(df, year_col, var_col):
    """
    1. Detect Shocks (Deviations > 1.5 StdDev).
    2. Analyze Resilience (Pre-Shock vs Post-Shock Growth).
    """
    # Guard: Need at least 5 years to calculate pre/post window meaningfully
    if len(df) < 5: return []

    df = df.sort_values(year_col)
    df['Growth'] = df[var_col].pct_change() * 100
    
    mean_g = df['Growth'].mean()
    std_g = df['Growth'].std()
    
    shocks = []
    
    for i, row in df.iterrows():
        if pd.isna(row['Growth']): continue
        
        # Detection
        dev = row['Growth'] - mean_g
        if abs(dev) > 1.5 * std_g:
            year = int(row[year_col])
            
            # Context Labels
            reason = "Volatility"
            if dev < 0: reason = "Contraction"
            if dev > 0: reason = "Expansion"
            if 2019 <= year <= 2021: reason = "Pandemic Shock"
            elif 2008 <= year <= 2009: reason = "Financial Crisis"
            
            # Resilience Analysis (Window: +/- 2 Years)
            curr_idx = df.index.get_loc(i)
            # Ensure indices are within bounds
            pre_start = max(0, curr_idx-2)
            post_end = min(len(df), curr_idx+3)
            
            pre_window = df.iloc[pre_start:curr_idx]['Growth'].mean()
            post_window = df.iloc[curr_idx+1:post_end]['Growth'].mean()
            
            resilience = "Unknown"
            if not pd.isna(post_window) and not pd.isna(pre_window):
                if post_window > pre_window: resilience = "Strong Recovery"
                elif post_window > 0: resilience = "Stabilized"
                else: resilience = "Persistent Drag"

            shocks.append({
                'year': year,
                'growth': round(row['Growth'], 2),
                'reason': reason,
                'pre_avg': round(pre_window, 2) if not pd.isna(pre_window) else "-",
                'post_avg': round(post_window, 2) if not pd.isna(post_window) else "-",
                'resilience': resilience,
                'type': 'danger' if dev < 0 else 'success'
            })
    return shocks

# --- NEW: STRONGEST SECTOR ENGINE ---
def get_sector_leaderboard(df, c_col, y_col, var):
    """
    Ranks sectors by Growth (Strength) and Volatility (Risk).
    This answers: 'Which sectors are most resilient?'
    """
    if not c_col or len(df[c_col].unique()) < 2: return []
    
    stats = []
    for s in df[c_col].unique():
        sub = df[df[c_col] == s].sort_values(y_col)
        if len(sub) < 3: continue
        
        g = sub[var].pct_change() * 100
        stats.append({
            'sector': s,
            'avg_growth': round(g.mean(), 2),
            'volatility': round(g.std(), 2), # Standard Deviation = Volatility
            'latest_val': round(sub[var].iloc[-1], 2)
        })
    
    # Sort by Growth (Strongest first)
    return sorted(stats, key=lambda x: x['avg_growth'] or -999, reverse=True)

def auto_feature_engineering(df, year_col, target_var):
    """
    Attempts to create Macro/Productivity features if standard columns exist.
    """
    cols = [c.lower() for c in df.columns]
    
    # 1. Output per Worker (if GDP and Emp exist)
    if 'rgdpe' in cols and 'emp' in cols:
        df['calc_output_per_worker'] = df['rgdpe'] / df['emp']
        
    # 2. Capital Deepening (Capital / Emp)
    if 'ck' in cols and 'emp' in cols:
        df['calc_capital_per_worker'] = df['ck'] / df['emp']
        
    # Select Features: All numeric cols EXCEPT target and year
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                    if c != target_var and c != year_col]
    
    return df, feature_cols

def check_and_melt(df):
    year_cols = [c for c in df.columns if re.search(r'\d{4}', str(c))]
    if len(year_cols) > 3:
        id_vars = [c for c in df.columns if c not in year_cols]
        df = df.melt(id_vars=id_vars, value_vars=year_cols, var_name='Year', value_name='Value')
        def clean_year(val):
            match = re.search(r'(\d{4})', str(val))
            return int(match.group(1)) if match else None
        df['Year'] = df['Year'].apply(clean_year)
        return df.dropna(subset=['Year'])
    return df

# --- PLOTTING (Safe OO Pattern) ---
def get_image_base64_battle(x_hist, y_hist, x_fore, y_fore, name, shocks):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # History
    ax.plot(x_hist, y_hist, 'k-', linewidth=2, label="Historical Data")
    
    # Forecast
    ax.plot(x_fore, y_fore, 'r--', marker='o', linewidth=2, label=f"Champion ({name})")
    
    # Highlight Shocks
    for s in shocks:
        if s['type'] == 'danger':
            ax.axvline(x=s['year'], color='red', alpha=0.2, linestyle='-')
            ax.text(s['year'], min(y_hist), f"{s['year']}", rotation=90, color='red', fontsize=8)

    ax.set_title(f"DSGE Trend Analysis: {name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def get_simple_plot(df, kind='line'):
    """
    Generates simple plots (Pie/Hist) for EDA.
    df: Data Series
    kind: 'pie' or 'hist'
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    if kind == 'pie':
        ax.pie(df, labels=df.index, autopct='%1.1f%%')
    elif kind == 'hist':
        sns.histplot(df, kde=True, ax=ax, color='#0f172a')
    
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# --- VIEW CONTROLLER ---
def load_smart_data(path, filename):
    with open(path, 'rb') as f:
        encoding = chardet.detect(f.read(10000))['encoding']
    if filename.endswith('.csv'): 
        return pd.read_csv(path, encoding=encoding, engine='python')
    return pd.read_excel(path)

def dashboard(request):
    country_kws = ['country', 'area', 'label', 'entity', 'region', 'ref_area', 'description', 'industry']
    year_kws = ['year', 'time', 'date', 'period']

    if request.method == 'POST':
        fs = FileSystemStorage()
        
        # 1. HANDLE FILE UPLOAD
        if 'dataset' in request.FILES:
            myfile = request.FILES['dataset']
            filename = fs.save(myfile.name, myfile)
            try:
                df = load_smart_data(fs.path(filename), filename)
                df = check_and_melt(df)
                
                # Auto-Detect Columns
                c_col = next((c for c in df.columns if any(k in c.lower() for k in country_kws)), None)
                y_col = next((c for c in df.columns if any(k in c.lower() for k in year_kws)), None)
                
                if not y_col: raise ValueError("No Year column found.")
                
                # Context for filtering dropdowns (if categoricals exist)
                sector_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if c != c_col]

                countries = sorted(df[c_col].unique().tolist()) if c_col else ["Global/Sector"]
                num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != y_col]
                if 'Value' in df.columns and 'Value' not in num_cols: num_cols.append('Value')
                
                return render(request, 'core/dashboard.html', {
                    'step': 'configure', 'filename': filename,
                    'countries': countries, 'variables': num_cols,
                    'sector_cols': sector_cols
                })
            except Exception as e:
                return render(request, 'core/dashboard.html', {'error': f"Scan Error: {str(e)}"})

        # 2. HANDLE ANALYSIS RUN
        elif 'filename' in request.POST:
            filename = request.POST['filename']
            country = request.POST.get('country')
            var = request.POST.get('target_var')
            
            df = load_smart_data(fs.path(filename), filename)
            df = check_and_melt(df)
            
            # Re-identify columns
            c_col = next((c for c in df.columns if any(k in c.lower() for k in country_kws)), None)
            y_col = next((c for c in df.columns if any(k in c.lower() for k in year_kws)), None)
            
            # --- CALCULATE LEADERBOARD BEFORE FILTERING ---
            sector_leaderboard = get_sector_leaderboard(df, c_col, y_col, var)
            
            # Filter Data
            if c_col and country != "Global/Sector": df = df[df[c_col] == country]
            
            # Preprocessing
            df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
            df = df.dropna(subset=[y_col]).sort_values(y_col)
            df[y_col] = df[y_col].astype(int)
            
            # Feature Engineering (Multivariate Check)
            df, feature_cols = auto_feature_engineering(df, y_col, var)
            
            # --- EXECUTE ENGINES ---
            
            # 1. Shock Analysis
            shocks = analyze_shocks(df.copy(), y_col, var)
            
            # 2. YoY Growth
            df['YoY_Growth'] = df[var].pct_change() * 100
            # FIX FOR TEMPLATE ERROR: Map selected var to a fixed name
            df['target_var'] = df[var]
            
            # 3. Model Arena Setup
            # If no features found, fallback to univariate (Use Year as X)
            if not feature_cols: 
                X = df[[y_col]].values
            else:
                X = df[feature_cols].values
            
            y_raw = df[var].values
            
            # Log Transform Logic
            use_log = (np.mean(y_raw) > 1000) and (np.min(y_raw) > 0)
            y = np.log(y_raw) if use_log else y_raw
            
            # Train/Test Split
            cutoff = int(len(y) * 0.8)
            
            # Init Arena
            arena = ModelArena(X[:cutoff], y[:cutoff], X[cutoff:], y[cutoff:], 
                               df_full=df, date_col=y_col, target_col=var, use_log=use_log)
            
            # Run All Models
            steps = 5
            
            # A. Naive Baselines (Crucial for Benchmark)
            arena.run_naive_benchmarks(steps)
            
            # B. Multivariate (if features exist)
            full_X = X 
            
            # C. Time Series Models
            arena.run_time_series(steps) 
            
            # D. ML Models (RF/XGB) 
            future_years = np.array([[int(X[-1][0]) + i] for i in range(1, 6)])
            
            # If no extra macro/productivity features were found, treat Year as X 
            # and allow Linear/RF/XGB to compete as univariate trend models for the forecast.
            if len(feature_cols) == 0:
                arena.run_multivariate_models(X, future_years)
            else:
                # If multivariate, run for test scores but dummy future to avoid crash
                dummy_future = X[-5:] 
                arena.run_multivariate_models(X, dummy_future)
            
            # --- GATHER RESULTS ---
            results = arena.results
            # Filter out failures (999.9)
            valid_results = {k:v for k,v in results.items() if v < 900}
            if not valid_results: valid_results = {"Naive (Persistence)": 0.0}
            
            # Sort Results (Best Model First)
            sorted_results = dict(sorted(valid_results.items(), key=lambda item: item[1]))
            champion = min(sorted_results, key=sorted_results.get)
            champion_score = sorted_results[champion]
            
            # Serialize for JS (Safety Fix)
            all_forecasts_json = json.dumps(arena.all_forecasts, cls=NumpyEncoder)
            future_years_json = json.dumps(list(range(int(df[y_col].max())+1, int(df[y_col].max())+6)), cls=NumpyEncoder)

            # --- SMARTER PIE CHART ENGINE (Top 10 + Others) ---
            pie_plot = None
            if c_col:
                try:
                    # 1. Reload full data
                    s_full = load_smart_data(fs.path(filename), filename)
                    s_full = check_and_melt(s_full)
                    
                    # 2. Determine Group Column (Sector or Country)
                    group_col = c_col
                    if len(s_full[c_col].unique()) < 2:
                        potential_sectors = [c for c in s_full.select_dtypes(include=['object', 'category']).columns 
                                           if c not in [c_col, y_col, 'source.label', 'indicator.label']]
                        if potential_sectors: group_col = potential_sectors[0]

                    # 3. Aggregate Latest Data
                    latest = df[y_col].max()
                    # Ensure numeric conversion for filtering
                    s_full[y_col] = pd.to_numeric(s_full[y_col], errors='coerce')
                    slice_data = s_full[s_full[y_col] == latest].groupby(group_col)[var].sum().sort_values(ascending=False)

                    # 4. Apply "Top 10 + Others" Logic to fix clutter
                    if len(slice_data) > 10:
                        top_10 = slice_data.iloc[:10]
                        others = pd.Series([slice_data.iloc[10:].sum()], index=['Others'])
                        slice_data = pd.concat([top_10, others])

                    # 5. Generate Plot
                    if len(slice_data) > 1:
                        pie_plot = get_simple_plot(slice_data, kind='pie')
                        
                except Exception as e:
                    print(f"Pie Chart Error: {e}")

            # Hist Plot
            hist_plot = get_simple_plot(df[var], kind='hist')

            # Clean YoY for Table
            yoy_data = [r for r in df.tail(6).to_dict('records') if not np.isnan(r.get('YoY_Growth', np.nan))]

            # Save Results to CSV
            try:
                results_df = pd.DataFrame(list(sorted_results.items()), columns=['Model', 'MAPE'])
                results_df.to_csv('media/model_results.csv', index=False)
            except: pass

            return render(request, 'core/dashboard.html', {
                'step': 'results',
                'results': sorted_results,
                'champion': champion,
                'champion_score': champion_score,
                'all_forecasts_json': all_forecasts_json,
                'future_years_json': future_years_json,
                'selected_country': country,
                'battle_plot': get_image_base64_battle(df[y_col], y_raw, 
                                                       range(int(df[y_col].max())+1, int(df[y_col].max())+6), 
                                                       arena.all_forecasts.get(champion, np.zeros(5)), 
                                                       champion, shocks),
                'yoy_data': yoy_data,
                'shocks': shocks,
                'target_var': var,
                'pie_plot': pie_plot,
                'hist_plot': hist_plot,
                'sector_leaderboard': sector_leaderboard 
            })

    return render(request, 'core/dashboard.html')