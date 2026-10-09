#!/usr/bin/env bash

rm -rf dist; python -m build
python -m twine upload dist/*.tar.gz
rm -rf dist build *.egg-info

# NOTE: we only distribute the source code, because binary distribution requires a dock image, which is too complicated for hobby projects.
