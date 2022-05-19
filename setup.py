import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

package = ["mediapipe","pyyaml","pandas","tqdm","seaborn","easydict"]

for each in package:
    install(package)