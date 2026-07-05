@echo off
REM --- Auto push for SimpliSql ---
cd /d "C:\Users\Chandana\SimpliSql"
git add .
git commit -m "Auto commit and push"
git push origin main

REM --- Auto push for Desktop_agent (SIMPLE_AI) ---
cd /d "C:\Users\Chandana\Desktop_agent"
git add .
git commit -m "Auto commit and push"
git push origin main

echo All changes pushed for both projects.
pause
