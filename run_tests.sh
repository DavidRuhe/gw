python -m venv .test_venv
source .test_venv/bin/activate
pip install -r requirements.txt
cd src/
python -m unittest **/*.py
rm -rf .test_venv
