@echo off
echo Testing Railway Deployment...
echo.
echo Opening signal generation endpoint in browser...
start https://web-production-41882.up.railway.app/api/signals/generate
echo.
echo Opening main dashboard...
start https://web-production-41882.up.railway.app/
echo.
echo Check the browser windows that just opened!
pause
