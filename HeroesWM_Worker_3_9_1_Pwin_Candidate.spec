# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

source = r'C:/Users/mrstr/Documents/Codex/2026-08-09/apier-apier-python-apier-windows-x64/work/production_3_9/source'
datas = [
    (source + '/cards_catalog.json', '.'),
    (source + '/policy_models.json', '.'),
    (source + '/opponent_policy.json', '.'),
]
binaries = []
hiddenimports = ['onnxruntime.capi._pybind_state']
tmp_ret = collect_all('ddddocr')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('onnxruntime')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

a = Analysis(
    [source + '/main.py'],
    pathex=[source],
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
    a.binaries,
    a.datas,
    [],
    name='HeroesWM_Worker_3_9_1_Pwin_Candidate',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
