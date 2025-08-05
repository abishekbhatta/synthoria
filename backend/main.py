import modal

app = modal.app("synthoria")


# Define a custom Docker image for the Modal environment.                                                                     │
# This image is based on Debian Slim and includes all necessary dependencies.

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .run_commands(["git clone https://github.com/ace-step/ACE-Step.git /tmp/syntheziser-model", "cd /tmp/syntheziser-model && pip install ."])

    # Ensures when Modal app downloads a large model from Hugging Face for the first time, it saves it to a persistent cache. On all 
    # subsequent runs, the app will find the model in that cache and load it directly, avoiding repeated downloads and significantly
    # speeding up startup times.    
                                                                 
    .env({"HF_HOME" : "./cache/huggingface"}) 
)


