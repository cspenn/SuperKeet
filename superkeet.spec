# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['superkeet/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('assets', 'assets'),
        ('config.yml', '.'),
        ('credentials.yml.dist', '.'),
    ],
    hiddenimports=[
        'parakeet_mlx',
        'pynput',
        'sounddevice',
        'PySide6',
        'numpy',
        'superkeet.config.validators',
        'superkeet.ui.tray_app',
        'superkeet.ui.main_window',
        'superkeet.ui.settings_dialog',
        'superkeet.ui.first_run_dialog',
        'superkeet.ui.waveform_widget',
        'superkeet.ui.audio_animation_widget',
        'superkeet.ui.drop_zone_widget',
        'superkeet.ui.batch_progress_dialog',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SuperKeet',
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
    icon='assets/SuperKeet.icns',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SuperKeet',
)
app = BUNDLE(
    coll,
    name='SuperKeet.app',
    icon='assets/SuperKeet.icns',
    bundle_identifier='com.cspenn.superkeet',
    info_plist={
        'NSMicrophoneUsageDescription': 'SuperKeet needs microphone access to transcribe your speech.',
        'nsHighResolutionCapable': 'True',
    },
)
