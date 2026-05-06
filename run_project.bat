@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -m venv .venv
)

echo Activating virtual environment...
call ".venv\Scripts\activate.bat"

echo Installing/updating dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

if not exist "artifacts\fruit_stage_cnn.keras" (
    echo Stage model missing. Training stage model...
    python -m src.train
)

if not exist "artifacts\fruit_type_cnn.keras" (
    echo Type model missing. Training type model...
    python -m src.train_type
)

echo Starting Streamlit dashboard...
python -m streamlit run app.py

endlocal
