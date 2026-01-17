# Start with a Python system
FROM python:3.10-slim

# 1. Install R and system tools
RUN apt-get update && apt-get install -y \
    r-base \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python Libraries
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3. Install R Libraries (ggplot2, forecast)
# We use a simple R script to install them during the build
RUN Rscript -e "install.packages(c('ggplot2', 'forecast', 'jsonlite', 'readr'), repos='http://cran.rstudio.com/')"

# 4. Copy your code
COPY . .

# 5. Run the Server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]