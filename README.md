# HUE_Bench

## Dependencies
Our project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage dependencies.
You only need to install uv and run the provided command:
```bash
uv sync
source .venv/bin/activate
```
to set up a local environment identical to the one used in our experiments.
## Run

Our project uses `./Config/config.yaml` to manage experiments. To run an example, please download the dataset from Kaggle and remove the `gocomics_download` and `gocomics_download_political` folders to the **root directory**, create an `./experiment` folder, modify the provided sample config file, and then execute the run:

```bash
bash ./bash_scripts/mcq_exp.sh
```

If you want to run our method DAD, please modify the `./Config/ours.yaml` file and run:
```bash
bash ./bash_scripts/mcq_exp_ours.sh
```
Attention that the `api_base` and `api_key` fields in `ouyrs.yaml` are used by both the MLLM and the reasoning model. If you are not using a unified API management platform, please modify lines 147–160 in the ./scripts/experiment/new_mcq.py file to use your own API.

