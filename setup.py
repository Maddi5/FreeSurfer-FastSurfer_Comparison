def Setup_Script():
    
    import sys
    import subprocess
    import os

    sys.path.append(os.getcwd())
    print(os.getcwd())

    print('Checking you have all packages needed for the codebase...')

    try:
        import numpy
    except:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy'])
    try:
        import nibabel
    except:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'nibabel'])
    try:
        import scipy
    except:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scipy'])
    try:
        import pytest
    except:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pytest'])
    try:
        import skimage
    except:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-image'])
    try:
        import medpy
    except:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'medpy'])
    try:
        import matplotlib.pyplot
    except:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib'])
    try:
        import seaborn
    except:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'seaborn'])
    try:
        import pandas
    except:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas'])

    print('Using cpu: ' + str(os.cpu_count()))
    print('All good, off we go!')

    return
