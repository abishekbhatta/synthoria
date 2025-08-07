import modal

app = modal.app("synthoria")


# Define a custom image for the Modal environment                                                                 
# This image is based on Debian Slim and includes all necessary dependencies


# Modal employs caching for imagefile if no changes

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .run_commands(["git clone https://github.com/ace-step/ACE-Step.git /tmp/synthesizer-model", "cd /tmp/synthesizer-model && pip install ."])

    # Ensures when Modal app downloads a LLM from Hugging Face for the first time, it saves it to a cache
    # On all subsequent runs, the app will find the model in that cache and load it directly 
                                                                 
    .env({"HF_HOME" : "./cache/huggingface"})
    .add_local_python_source("prompts") 
)



# Utilized modal.Volume to ensure qwen-2 and synthesizer model are downloaded & 
# easy to re-distribute w/o re-downloading them again

model_volume = modal.Volume.from_name("synthesizer-model", create_if_missing=True)
hf_volume = modal.Volume.from_name("qwen-2-hf-cache", create_if_missing=True)


# @app here is a decorator (Functions that take another function as a param)
# @app.cls() for a class & @app.function() for a function
@app.cls(

    image=image,    
    gpu="L40S",
    volumes={"./models": model_volume, "/.cache/huggingface": hf_volume},  
    scaledown_window= 15  # Keep container idle for extra 15s after a request is dealt with
                          # If concurrent request, speeds up request response as model already loaded into GPU's memory
)
class SynthoriaServer:
    @modal.enter() # Loads model if it was a cold start 
    def load_model(self):
        from acestep.pipeline_ace_step import ACEStepPipeline            # Importing ACEStepPipeline for loading synthesizer model
        from transformers import AutoModelForCausalLM, AutoTokenizer     # For Qwen-2 Model
        from diffusers import AutoPipelineForText2Image
        import torch

        # synthesizer model - for musics
        self.synthesizer_model = ACEStepPipeline(
            checkpoint_directory = "/models",
            dtype="bfloat16",
            torch_compile=False,
            cpu_offload=False,
            overlapped_decode=False
        )

        # qwen-2 7B model - for lyrics
        qwen2_model_id = "Qwen/Qwen2-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(qwen2_model_id)

        self.qwen2_model = AutoModelForCausalLM.from_pretrained(
            qwen2_model_id,
            torch_dtype="auto",
            device_map="auto",
            cache_dir = "/.cache/huggingface"
        )

        # sdxl-turbo model - for thumbnails
        self.sdxl_turbo_model = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", 
            torch_dtype=torch.float16, 
            variant="fp16", 
            cache_dir = "/.cache/huggingface"
        )

        self.sdxl_turbo_model.to("cuda")     # cuda moves the model to GPU
                                             # normally, sdxl_turbo model is loaded into CPU

    

        















