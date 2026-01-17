# backend_r/script_main.R
args <- commandArgs(trailingOnly = TRUE)
file_path <- args[1]
output_dir <- args[2]

# Load libraries silently
suppressPackageStartupMessages({
  library(ggplot2)
  library(forecast)
  library(jsonlite)
  library(readr)
})

# 1. READ DATA
df <- read_csv(file_path, show_col_types = FALSE)

# 2. GENERATE TREND PLOT
plot_filename <- "trend_analysis.png"
plot_path <- file.path(output_dir, plot_filename)

png(plot_path, width=800, height=400)

# Plot 2nd col vs 1st col
p <- ggplot(df, aes(x=df[[1]], y=df[[2]])) +
  geom_line(color="#2c3e50", linewidth=1) +
  geom_smooth(method="loess", color="#e74c3c", fill="#e74c3c", alpha=0.2) +
  labs(title="Sectoral Output Trend & Shock Detection", 
       x=names(df)[1], y=names(df)[2]) +
  theme_minimal()

print(p)

# --- THE FIX IS HERE ---
# We assign the result to a dummy variable 'x' so it doesn't print to console
x <- dev.off() 

# 3. STATISTICAL MODELING (ARIMA)
ts_data <- ts(df[[2]]) 
model <- auto.arima(ts_data)
preds <- forecast(model, h=5)

# Calculate Error (MAPE)
actuals <- tail(ts_data, 5)
fitted <- tail(fitted(model), 5)
mape <- mean(abs((actuals - fitted) / actuals)) * 100

# 4. SEND JSON BACK TO PYTHON
results <- list(
  r_mape = round(mape, 2),
  forecast = as.numeric(preds$mean),
  plot_url = paste0("/static/", plot_filename)
)

# Print ONLY the JSON
cat(toJSON(results, auto_unbox = TRUE))