@echo off
setlocal
REM ============================================================
REM  Khoi dong CARLA server toi uu FPS cho demo ADAS
REM  Cach dung (double-click hoac chay trong terminal):
REM    launch_carla.bat          -> RHI mac dinh (DX12 trong build nay), co cua so 3D
REM    launch_carla.bat off      -> RHI mac dinh RenderOffScreen, khong cua so
REM                                 (camera van hoat dong, ban xem qua dashboard)
REM    launch_carla.bat default-rhi       -> RHI mac dinh cua CARLA (DX12 trong build nay), co cua so
REM    launch_carla.bat default-rhi off   -> RHI mac dinh + RenderOffScreen
REM    launch_carla.bat dx11              -> DX11 Low quality, co cua so 3D
REM    launch_carla.bat dx11 off          -> DX11 RenderOffScreen
REM  Neu cai CARLA cho khac, sua CARLA_ROOT ben duoi.
REM ============================================================
set "CARLA_ROOT=C:\Users\Admin\OneDrive\Desktop\Carla Simulator"
set "PROJECT_ROOT=%~dp0"
set "PROJECT_PYTHON=%PROJECT_ROOT%.venvCarLa\Scripts\python.exe"
set "PORT=2000"
set "RHI_MODE=default-rhi"
set "RHI_ARGS="
set "RENDER_MODE=windowed"
set "ARG_ERROR="

REM Parse optional renderer/RHI tokens without changing legacy defaults.
if /I "%~1"=="off" set "RENDER_MODE=offscreen"
if /I "%~1"=="windowed" set "RENDER_MODE=windowed"
if /I "%~1"=="dx11" (
  set "RHI_MODE=dx11"
  set "RHI_ARGS=-d3d11"
)
if /I "%~1"=="default-rhi" (
  set "RHI_MODE=default-rhi"
  set "RHI_ARGS="
)
if /I "%~1"=="" goto :rhi_args_second
if /I "%~1"=="off" goto :rhi_args_second
if /I "%~1"=="windowed" goto :rhi_args_second
if /I "%~1"=="dx11" goto :rhi_args_second
if /I "%~1"=="default-rhi" goto :rhi_args_second
set "ARG_ERROR=%~1"

:rhi_args_second
if /I "%~2"=="off" set "RENDER_MODE=offscreen"
if /I "%~2"=="windowed" set "RENDER_MODE=windowed"
if not "%~2"=="" if /I not "%~2"=="off" if /I not "%~2"=="windowed" set "ARG_ERROR=%~2"
if defined ARG_ERROR (
  echo [LOI] Tham so khong hop le: %ARG_ERROR%
  echo [LOI] Dung: launch_carla.bat [dx11^|default-rhi] [off^|windowed]
  exit /b 5
)

if not exist "%CARLA_ROOT%\CarlaUE4.exe" (
  echo [LOI] Khong tim thay "%CARLA_ROOT%\CarlaUE4.exe"
  echo       Hay sua bien CARLA_ROOT trong file nay.
  pause
  exit /b 1
)

powershell -NoProfile -Command "if (Get-Process -Name 'CarlaUE4-Win64-Shipping' -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
if not errorlevel 1 (
  echo [LOI] CARLA dang chay hoac dang treo. Hay dong tien trinh cu truoc khi launch lai.
  exit /b 2
)

if /I "%RENDER_MODE%"=="offscreen" (
  echo [CARLA] %RHI_MODE% RenderOffScreen Low - dashboard se la cua so quan sat.
  start "" "%CARLA_ROOT%\CarlaUE4.exe" %RHI_ARGS% -RenderOffScreen -quality-level=Low -nosound -carla-rpc-port=%PORT%
) else (
  echo [CARLA] %RHI_MODE% Low-quality windowed 640x360 - VSync giu render queue on dinh.
  start "" "%CARLA_ROOT%\CarlaUE4.exe" %RHI_ARGS% -quality-level=Low -windowed -ResX=640 -ResY=360 -nosound -carla-rpc-port=%PORT%
)

echo.
if not exist "%PROJECT_PYTHON%" (
  echo [LOI] Khong tim thay "%PROJECT_PYTHON%" de probe CARLA API.
  exit /b 3
)

echo [CARLA] Dang cho world API va frame async san sang, timeout 120s...
"%PROJECT_PYTHON%" "%PROJECT_ROOT%wait_for_carla.py" --host 127.0.0.1 --port %PORT% --timeout 120
if errorlevel 1 (
  echo [LOI] Port co the mo nhung CARLA world khong san sang. Khong chay chinh.py.
  exit /b 4
)

echo [OK] CARLA server va world API da san sang tren port %PORT%.
echo      Chay pipeline bang moi truong dung:
echo      "%PROJECT_PYTHON%" "%PROJECT_ROOT%chinh.py" --vehicles 0 --town Town02 --performance-profile low-memory --runtime-mode async-stable --inference-device auto
endlocal
