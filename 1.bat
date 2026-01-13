@echo off

cd /d C:\Quant

python RO.py

if %errorlevel% neq 0 (

    echo.

    echo [오류 발생] 위 메시지를 확인하세요.

    pause

)

cmd /k