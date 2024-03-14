@echo off

REM Name of the virtual environment
SET VENV_NAME=venv

REM Remove the existing virtual environment if it exists
IF EXIST "%VENV_NAME%" (
    echo Removing existing virtual environment %VENV_NAME%...
    rmdir /s /q %VENV_NAME%
)

REM Create the virtual environment
python -m venv %VENV_NAME%
echo Virtual environment %VENV_NAME% created.

REM Activate the virtual environment
CALL %VENV_NAME%\Scripts\activate

REM Upgrade pip using python -m pip to avoid the error
python -m pip install --upgrade pip

REM Install TensorFlow
pip install tensorflow

REM Install NumPy
pip install numpy

REM Install Pandas
pip install pandas

REM Install google-api-python-client
pip install google-api-python-client

REM Install google-auth-oauthlib
pip install google-auth-oauthlib

REM Install google-auth-httplib2
pip install google-auth-httplib2

REM Install matplotlib
pip install matplotlib

REM Install scikit-learn
pip install scikit-learn

echo Dependencies installed in virtual environment %VENV_NAME%.

REM Deactivate the virtual environment
deactivate