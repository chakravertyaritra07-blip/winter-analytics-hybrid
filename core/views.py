import os
import json
import subprocess
import pandas as pd
import numpy as np
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

def dashboard(request):
    context = {}
    
    if request.method == 'POST' and request.FILES.get('dataset'):
        print("--- DEBUG: RECEIVED FILE ---") # Debug 1
        
        # 1. SAVE FILE
        myfile = request.FILES['dataset']
        fs = FileSystemStorage()
        # Force delete old file to ensure we get the new one
        if fs.exists(myfile.name):
            fs.delete(myfile.name)
        filename = fs.save(myfile.name, myfile)
        file_path = fs.path(filename)
        
        print(f"--- DEBUG: SAVED AT {file_path} ---") # Debug 2
        
        try:
            # 2. READ DATA WITH ROBUST CLEANING
            # encoding='utf-8-sig' removes hidden Excel characters
            df = pd.read_csv(file_path, encoding='utf-8-sig') 
            
            print("--- DEBUG: RAW DATA HEAD ---")
            print(df.head()) # Show us what it read
            
            # NUCLEAR CLEANING: Force everything to strings, then to numbers
            # This handles "Year" headers that appear in the middle of data
            df = df.astype(str) 
            df = df.apply(pd.to_numeric, errors='coerce')
            
            # Drop Empty Rows
            df.dropna(inplace=True)
            
            print("--- DEBUG: CLEANED DATA HEAD ---")
            print(df.head()) # Show us the clean version

            if len(df) < 5:
                raise ValueError(f"Dataset too small. Only found {len(df)} valid numeric rows.")

            # 3. PYTHON AI MODELING
            # Column 0 = X, Column 1 = y
            X = df.iloc[:, 0].values.reshape(-1, 1)
            y = df.iloc[:, 1].values 
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            rf = RandomForestRegressor(n_estimators=100)
            rf.fit(X_train, y_train)
            py_preds = rf.predict(X_test)
            py_mape = mean_absolute_percentage_error(y_test, py_preds) * 100
            
            # 4. CALL R BACKEND
            r_script = os.path.join(settings.BASE_DIR, 'backend_r', 'script_main.R')
            static_dir = os.path.join(settings.BASE_DIR, 'static')
            r_exe = r"C:\Program Files\R\R-4.5.2\bin\Rscript.exe"
            
            cmd = [r_exe, r_script, file_path, static_dir]
            process = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            r_data = json.loads(process.stdout)
            
            context = {
                'success': True,
                'py_mape': round(py_mape, 2),
                'r_mape': r_data['r_mape'],
                'comparison': "Python AI Wins" if py_mape < r_data['r_mape'] else "R Statistics Wins",
                'plot_url': r_data['plot_url'],
                'forecast': r_data['forecast']
            }
            
        except Exception as e:
            print("--- FATAL ERROR ---")
            print(e) # This will show the exact error in terminal
            context['error'] = f"System Error: {str(e)}"

    return render(request, 'core/dashboard.html', context)