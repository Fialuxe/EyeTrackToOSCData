@echo off

:: 1. Create virtual environment if it doesn't exist
:: (Windows uses 'python', not 'python3' usually, unless you have a specific alias)
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

:: 2. Activate the virtual environment
:: On Windows, the activation script is in Scripts\activate
call .\venv\Scripts\activate

:: 3. Install requirements
pip install -r requirements.txt

:: 4. Set the environment variable (Windows uses 'set' instead of 'export')
set LSL_FORCE_DEFAULT_MULTICAST=0

:: 5. Suppress the specific Protobuf UserWarning
:: We tell Python to ignore warnings that match this specific text
set PYTHONWARNINGS=ignore:SymbolDatabase.GetPrototype() is deprecated

:: 6. Run the script
:: Change --camera 0 to --camera 1 if it does not work with your default camera(like OBS virtual camera)
echo Starting OSC Server...
python osc_server.py --camera 0 --filter kalman

:: Pause so you can see errors if the script crashes immediately
pause