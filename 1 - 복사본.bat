@echo off
cd /d C:\Quant

echo ==========================
echo 1. ROre.py 실행 중...
echo ==========================
python ROre.py
if %errorlevel% neq 0 (
    echo.
    echo [오류 발생] ROre.py 실행 실패. 위 내용을 확인하세요.
    pause
    exit
)
if %errorlevel% neq 0 (
    echo.
    echo [오류 발생] Order.py 실행 실패. 위 내용을 확인하세요.
    pause
    exit
)

echo.