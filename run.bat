@echo off
echo Starting Linear Regression Learning System (Python)...
python -m pip install -r requirements.txt --quiet
python -m streamlit run app.py
pause
