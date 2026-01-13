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

echo.
echo ==========================
echo 2. ro.py 실행 중...
echo ==========================
python ro.py
if %errorlevel% neq 0 (
    echo.
    echo [오류 발생] ro.py 실행 실패. 위 내용을 확인하세요.
    pause
    exit
)

echo.
echo ==========================
echo 30초 대기 중... (주문 준비)
echo ==========================
timeout /t 30

echo.
echo ==========================
echo 3. Order.py 실행 중...
echo ==========================
python Order.py
if %errorlevel% neq 0 (
    echo.
    echo [오류 발생] Order.py 실행 실패. 위 내용을 확인하세요.
    pause
    exit
)

echo.
echo 모든 작업 완료.
cmd /k