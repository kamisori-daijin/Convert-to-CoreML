import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os

# Configuration for parsing command line arguments
parser = argparse.ArgumentParser(description="Convert a Hugging Face model to Core ML.")
parser.add_argument("--model", type=str, required=True, help="Path to the downloaded Hugging Face model directory (e.g., 'gemma-3-1b-it').")
args = parser.parse_args()

# Use the path received as a command line argument
downloaded_hf_model_dir = args.model

print(f"CoreMLTools Version: {ct.__version__}")
print(f"PyTorch Version: {torch.__version__}")
print(f"Transformers Version: {AutoTokenizer.__version__}")
print(f"Numpy Version: {np.__version__}")

try:
    # 1. Hugging Face model loading
    print(f"Loading Hugging Face model from '{downloaded_hf_model_dir}' into memory...")
    model = AutoModelForCausalLM.from_pretrained(downloaded_hf_model_dir, torch_dtype=torch.float16)
    model.eval()
    model.config.use_cache = False
    print("Model loaded and configured.")

    # 2. Create a wrapper model for Core ML conversion
    class GemmaCoreMLWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.model.config.use_cache = False

        def forward(self, input_ids, attention_mask):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=False,
                output_attentions=False,
                output_hidden_states=False
            )
            logits = outputs[0]
            return logits

    wrapped_model = GemmaCoreMLWrapper(model)
    print("Wrapper model prepared.")

    # 3. Prepare dummy inputs for TorchScript tracing
    max_seq_length = 1024
    tokenizer = AutoTokenizer.from_pretrained(downloaded_hf_model_dir)

    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, 10), dtype=torch.long)
    dummy_attention_mask = torch.ones(1, 10, dtype=torch.long)

    # 4. Trace the wrapper model to TorchScript
    print("Tracing wrapper model to TorchScript...")
    traced_model = torch.jit.trace(wrapped_model, (dummy_input_ids, dummy_attention_mask))
    print("TorchScript tracing complete.")

    # 5. Convert TorchScript model to Core ML
    print("Converting TorchScript model to Core ML...")
    coreml_model = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(upper_bound=max_seq_length)), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, ct.RangeDim(upper_bound=max_seq_length)), dtype=np.int32)
        ],
        source="pytorch",
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16
    )

   
    model_name = os.path.basename(downloaded_hf_model_dir)
    output_filename = f"{model_name}-coreml.mlpackage"
    coreml_model.save(output_filename)
    print(f"CoreML model saved successfully to {output_filename}.")

except Exception as e:
    print(f"Conversion to CoreML failed: {e}")
