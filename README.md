# sf-operations
Silver Fund operations team repo

# Setup
1. Clone the repo
2. Install uv with

curl -LsSf https://astral.sh/uv/install.sh | sh

3. Check that uv is installed by running

uv --version

If this returns an error you might need to add uv to your path. Run:

source $HOME/.local/bin/env

Restart your terminal for the changes to take effect.

4. Make a virtual environment by running 

uv sync

Activate it with 

source .venv/bin/activate