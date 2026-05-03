param([string]$PythonExe = "D:/software/Anaconda_envs/envs/myEVS/python.exe")
powershell -ExecutionPolicy Bypass -File "$PSScriptRoot/run_led_scene_full.ps1" -Scene "scene_1034" -PythonExe $PythonExe -EvflowLite
