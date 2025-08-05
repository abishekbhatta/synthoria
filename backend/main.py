import modal

app = modal.app("synthoria")


# Define a custom Docker image for the Modal environment.                                                                     │
# This image is based on Debian Slim and includes all necessary dependencies.

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .run_commands(["git clone https://github.com/ace-step/ACE-Step.git /tmp/synthesizer-model", "cd /tmp/synthesizer-model && pip install ."])

    # Ensures when Modal app downloads a large model from Hugging Face for the first time, it saves it to a cache. 
    # On all subsequent runs, the app will find the model in that cache and load it directly.    
                                                                 
    .env({"HF_HOME" : "./cache/huggingface"})
    .add_local_python_source("prompts") 
)


# Utilized modal.Volume to ensures qwen-2 and synthesizer model are downloaded & 
# easy to share w/o re-downloading them again.

model_volume = modal.Volume.from_name("synthesizer-model", create_if_missing=True)
hf_volume = modal.Volume.from_name("qwen-2-hf-cache", create_if_missing=True)


