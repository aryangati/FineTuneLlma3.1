

# End-to-End LLM Fine-Tuning: Llama-3.1 on Legal Data with NeMo and NVIDIA NIM

This project demonstrates a complete, production-grade workflow for customizing a Large Language Model. It involves fine-tuning a **Llama-3.1-8B** model on a specialized legal dataset using **NVIDIA NeMo Framework** with a **LoRA adapter** and deploying the resulting model for high-performance inference using an **NVIDIA NIM (NVIDIA Inference Microservice)**.

The specific task is to generate concise, relevant titles for legal questions from the Law StackExchange dataset.

## Project Overview

The core objective is to showcase an end-to-end MLOps lifecycle for generative AI. This isn't just about model accuracy; it's about the entire process of adapting a powerful foundation model for a specific domain and making it available as a robust, scalable service.

This project is a direct demonstration of skills in:

  * **Parameter-Efficient Fine-Tuning (PEFT)**: Efficiently customizing LLMs without the massive cost of full retraining.
  * **Scalable AI Frameworks**: Using industry-standard tools like NVIDIA NeMo to manage complex training jobs.
  * **Production Deployment**: Containerizing and serving the fine-tuned model using NVIDIA NIM for optimized, multi-tenant inference.

-----

## Technical Architecture

The architecture is split into two distinct, industry-standard phases: the **Training Phase** and the **Deployment Phase**.

1.  **Phase 1: Fine-Tuning with NeMo Framework**

      * **Foundation Model**: We start with the `llama-3.1-8b-instruct.nemo` checkpoint, a powerful base model ready for customization.
      * **Dataset**: The project uses a curated version of the **Law StackExchange** dataset, preprocessed into an `input` (question) and `output` (human-written title) format suitable for supervised fine-tuning.
      * **Fine-Tuning Technique**: **LoRA (Low-Rank Adaptation)** is employed. This PEFT method freezes the base model's weights and injects small, trainable "adapter" layers. This drastically reduces the number of trainable parameters (from 8 billion to \~10.5 million), making the fine-tuning process faster and more memory-efficient while still achieving high performance on the target task.
      * **Framework**: **NVIDIA NeMo** orchestrates the entire training process. It handles the data loading, model configuration, and the execution of the fine-tuning script (`megatron_gpt_finetuning.py`), abstracting away much of the complexity of distributed training.

2.  **Phase 2: Deployment with NVIDIA NIM**

      * **Inference Service**: **NVIDIA NIM** is a containerized, pre-built microservice optimized for deploying generative AI models. It provides a production-ready, high-throughput inference server out of the box.
      * **Model Serving**: The NIM is launched as a Docker container. Crucially, it's configured to load both the original **Llama-3.1 base model** and our newly trained **LoRA adapter**.
      * **Dynamic Adapter Loading**: NIM can host multiple LoRA adapters simultaneously on top of a single base model. This is incredibly efficient, as requests can be routed to different "specialized" versions of the model without needing to load separate 8B-parameter models into GPU memory for each task.
      * **API Endpoint**: The running NIM exposes an OpenAI-compatible API endpoint (`/v1/completions`), allowing for seamless integration with any application or testing script using standard REST API calls.

-----

## Challenges & Learnings

This project highlights several real-world MLOps challenges and their solutions.

#### **Challenge 1: Efficiently Customizing a Massive Foundation Model**

  * **Problem**: Full fine-tuning of an 8-billion-parameter model is computationally prohibitive, requiring immense GPU resources and time.
  * **Solution**: By implementing **LoRA**, we reduced the trainable parameters by **\~99.8%**. This demonstrated an understanding of modern PEFT techniques, which are critical for making LLM customization feasible and cost-effective. The final LoRA adapter is only \~21 MB, compared to the \~15 GB base model.

#### **Challenge 2: Bridging the Gap from Training to Production**

  * **Problem**: A trained model checkpoint (`.nemo` file) is not an application. Making it a scalable, reliable service is a complex engineering task.
  * **Solution**: **NVIDIA NIM** was used to solve this directly. Instead of building a custom Flask/FastAPI server, we leveraged a production-grade, containerized solution. This demonstrates the ability to use industry tools to accelerate deployment, manage dependencies, and ensure high performance. The setup script (`setup-nim.sh`) shows the process of authenticating with NGC, pulling the NIM container, and mounting the LoRA adapters for serving.

-----

## Evaluation & Results

The model's performance was evaluated on its ability to generate titles that are semantically similar to the ground-truth titles written by humans.

  * **Metric**: **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** was used to measure the n-gram overlap between the model-generated titles and the reference titles.
  * **Results**: On a test set of 128 examples, the fine-tuned model achieved the following scores:
      * **ROUGE-1**: 40.09% (Measures unigram overlap; good for fluency)
      * **ROUGE-2**: 20.39% (Measures bigram overlap; good for phrase matching)
      * **ROUGE-L**: 35.80% (Measures longest common subsequence; good for overall similarity)

These results are strong for a summarization/title-generation task and are comparable to the established baseline, confirming that the LoRA fine-tuning was highly effective.

-----

## How to Run Locally

#### **Prerequisites**

  * NVIDIA GPU with CUDA installed
  * Docker and NVIDIA Container Toolkit
  * NGC CLI configured

#### **1. Setup and Download Model**

```bash
# Download and setup NGC CLI
wget https://raw.githubusercontent.com/brevdev/notebooks/main/assets/setup-ngc.sh
chmod +x setup-ngc.sh && ./setup-ngc.sh

# Download the Llama-3.1-8B-Instruct base model
./ngc-cli/ngc registry model download-version "nvidia/nemo/llama-3_1-8b-instruct-nemo:1.0"
```

#### **2. Prepare Data and Run Fine-Tuning**

```bash
# Download and preprocess the dataset
wget https://huggingface.co/datasets/bigmlguy2234/hf-law-qa-dataset/resolve/main/law-qa-curated.zip
unzip -j law-qa-curated.zip -d curated-data
python prepare_dataset.py # Assuming the preprocessing script is named this

# Run the NeMo fine-tuning script (adjust paths as needed)
torchrun --nproc_per_node=1 \
    /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \
    exp_manager.exp_dir=./results/Meta-llama3.1-8B-Instruct-titlegen \
    trainer.max_steps=50 \
    model.restore_from_path=./llama-3_1-8b-instruct-nemo_v1.0/llama3_1_8b_instruct.nemo \
    model.data.train_ds.file_names="[./curated-data/law-qa-train_preprocessed.jsonl]" \
    model.data.validation_ds.file_names="[./curated-data/law-qa-val_preprocessed.jsonl]" \
    model.peft.peft_scheme=lora
```

#### **3. Deploy with NVIDIA NIM**

```bash
# Download and run the NIM setup script
wget https://raw.githubusercontent.com/brevdev/notebooks/main/assets/setup-nim.sh -O setup-nim
chmod +x setup-nim
export NGC_API_KEY=<YOUR_NGC_API_KEY>
./setup-nim # This will pull the NIM container and start it with the trained LoRA
```

#### **4. Run Inference**

```python
import requests
import json

url = 'http://0.0.0.0:8000/v1/completions'
headers = {'Content-Type': 'application/json'}

prompt = """Generate a concise, engaging title for the following legal question... [Your Question Here] ... \nTITLE: """

data = {
    "model": "llama3.1-8b-law-titlegen", # The ID of our LoRA adapter
    "prompt": prompt,
    "max_tokens": 50
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
```
