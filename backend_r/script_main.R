args <- commandArgs(trailingOnly = TRUE)
file_path <- args[1]
output_dir <- args[2]

suppressPackageStartupMessages({
  library(ggplot2)
  library(forecast)
  library(jsonlite)
  library(readr)
})

# 1. READ PROCESSED DATA
df <- read_csv(file_path, show_col_types = FALSE)
colnames(df) <- c("Time", "Value") # Standardize

# 2. SHOCK DETECTION (Econometric Deviation)
# We use LOESS to estimate the "Long Run Trend"
trend_model <- loess(Value ~ Time, data=df, span=0.5)
df$Trend <- predict(trend_model)
df$Cycle <- df$Value - df$Trend # This is the "Shock" component

# Identify Shock Years (Deviation > 1.5 Std Dev)
std_dev <- sd(df$Cycle)
df$IsShock <- abs(df$Cycle) > (1.5 * std_dev)

# 3. VISUALIZATION (Trend vs. Shock)
plot_filename <- "trend_shock_analysis.png"
plot_path <- file.path(output_dir, plot_filename)

png(plot_path, width=900, height=500)

p <- ggplot(df, aes(x=Time)) +
  # A. The Long Run Trend (Blue)
  geom_line(aes(y=Trend, color="Long-Run Trend"), linewidth=1.2) +
  # B. The Actual Data (Grey)
  geom_line(aes(y=Value, color="Actual Output"), linewidth=0.8, alpha=0.6) +
  # C. Highlight Shocks (Red Points)
  geom_point(data=subset(df, IsShock==TRUE), aes(y=Value), color="red", size=3) +
  # D. Forecast Area
  labs(title="Structural Break & Shock Analysis (DSGE)", 
       subtitle="Red dots indicate economic shocks (deviations > 1.5 SD)",
       y="Productivity / Output", x="Year") +
  theme_minimal() +
  scale_color_manual(values=c("Long-Run Trend"="#2563eb", "Actual Output"="#475569"))

print(p)
garbage <- dev.off() # Silence R

# 4. FORECASTING (ARIMA)
ts_data <- ts(df$Value, start=min(df$Time))
model <- auto.arima(ts_data)
preds <- forecast(model, h=5)

# Calculate MAPE
actuals <- tail(ts_data, 5)
fitted <- tail(fitted(model), 5)
mape <- mean(abs((actuals - fitted) / actuals)) * 100

results <- list(
  r_mape = round(mape, 2),
  forecast = as.numeric(preds$mean),
  plot_url = paste0("/static/", plot_filename)
)

cat(toJSON(results, auto_unbox = TRUE))