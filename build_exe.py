import PyInstaller.__main__

PyInstaller.__main__.run([
    'resim.py',
    '--onefile',
    '--name=resim',
    '--distpath=dist',
    '--workpath=build',
    '--clean'
])