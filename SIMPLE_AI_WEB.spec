# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

# NOTE: Do not bundle local model files in web build.
datas = [('web', 'web'), ('instruction_templates.json', '.')]
binaries = []
hiddenimports = ['webview', 'requests', 'bs4', 'pandas', 'openpyxl', 'reportlab', 'docx', 'fitz', 'pytesseract', 'PIL', 'PIL.Image', 'duckdb', 'sklearn', 'joblib', 'numpy', 'scipy', 'scipy.sparse', 'pyttsx3', 'pyttsx3.drivers', 'pyttsx3.drivers.sapi5', 'pyttsx3.voice', 'comtypes', 'comtypes.client', 'win32com', 'win32com.client']
tmp_ret = collect_all('llama_cpp')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('sklearn')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['agent_web.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SIMPLE_AI_WEB',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SIMPLE_AI_WEB',
)
