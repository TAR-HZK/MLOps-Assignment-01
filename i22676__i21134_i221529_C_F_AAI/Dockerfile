FROM python:3.9-slim

WORKDIR /app

# copy only requirements first to leverage cache
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# copy entire project
COPY . .

# expose port your Flask app uses
EXPOSE 5000

# run your Flask app; if your file is app/app.py using "app" callable
CMD ["python", "app/app.py"]
