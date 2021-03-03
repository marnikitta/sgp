#mypy --ignore-missing-imports ./ &&
python setup.py sdist &&
  twine upload dist/*
