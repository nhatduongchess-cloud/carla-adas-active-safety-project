@echo off
setlocal
REM ============================================================
REM  Khoi dong CARLA server toi uu FPS cho demo ADAS
REM  Cach dung (double-click hoac chay trong terminal):
REM    launch_carla.bat          -> Low quality, co cua so 3D
REM    launch_carla.bat off      -> RenderOffScreen: FPS cao nhat, khong cua so
REM                                 (camera van hoat dong, ban xem qua dashboard)
REM  Neu cai CARLA cho khac, sua CARLA_ROOT ben duoi.
REM ============================================================
set "CARLA_ROOT=C:\Users\Admin\OneDrive\Desktop\Carla Simulator"
set "PORT=2000"

if not exist "%CARLA_ROOT%\CarlaUE4.exe" (
  echo [LOI] Khong tim thay "%CARLA_ROOT%\CarlaUE4.exe"
  echo       Hay sua bien CARLA_ROOT trong file nay.
  pause
  exit /b 1
)

if /I "%~1"=="off" (
  echo [CARLA] RenderOffScreen - FPS cao nhat, khong cua so 3D.
  start "" "%CARLA_ROOT%\CarlaUE4.exe" -RenderOffScreen -quality-level=Low -carla-rpc-port=%PORT%
) else (
  echo [CARLA] Low-quality windowed 800x600 - nhuong GPU cho YOLO.
  start "" "%CARLA_ROOT%\CarlaUE4.exe" -quality-level=Low -windowed -ResX=800 -ResY=600 -carla-rpc-port=%PORT%
)

echo.
echo [OK] Da khoi dong CARLA server (port %PORT%).
echo      Cho ~10-20s cho server san sang, roi chay:
echo         python chinh.py --clean --vehicles 15 --town Town02
endlocal
