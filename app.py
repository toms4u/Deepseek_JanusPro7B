import streamlit as st
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from PIL import Image
import numpy as np
import os

# Set environment variables for MPS memory management
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.0'

# Set page config
st.set_page_config(page_title="Janus Demo", layout="wide")

# Determine device
def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

device = get_device()
st.info(f"Using device: {device}")

@st.cache_resource
def load_model():
    try:
        model_path = "deepseek-ai/Janus-Pro-7B"
        config = AutoConfig.from_pretrained(model_path)
        language_config = config.language_config
        language_config._attn_implementation = 'eager'
        
        # Load with proper precision
        if device == 'cuda':
            vl_gpt = AutoModelForCausalLM.from_pretrained(
                model_path,
                language_config=language_config,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16  # Consistent dtype for CUDA
            ).to(device)
        else:
            vl_gpt = AutoModelForCausalLM.from_pretrained(
                model_path,
                language_config=language_config,
                trust_remote_code=True
            ).float()  # Use float32 for CPU
            
        vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        return vl_gpt, vl_chat_processor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise e
    
def clear_memory():
    if device == 'mps':
        torch.mps.empty_cache()
    elif device == 'cuda':
        torch.cuda.empty_cache()

# Load model and processor
try:
    with st.spinner("Loading model... This might take a few minutes."):
        clear_memory()
        vl_gpt, vl_chat_processor = load_model()
        tokenizer = vl_chat_processor.tokenizer
        clear_memory()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

@torch.inference_mode()
def multimodal_understanding(image, question, seed, top_p, temperature):
    clear_memory()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    pil_image = Image.open(image).convert("RGB")
    
    conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>\n{question}",
            "images": [pil_image],
        },
        {"role": "Assistant", "content": ""},
    ]
    
    # Use matching dtype for CUDA
    if device == 'cuda':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
        
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=[pil_image],
        force_batchify=True
    ).to(device, dtype=dtype)  # Match model's dtype
    
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
    )
    
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer

def generate(input_ids, width, height, temperature=0.8, parallel_size=1, cfg_weight=5,
            image_token_num_per_image=576, patch_size=16):
    clear_memory()
    
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).to(device)
    
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id
            
    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(device)
    pkv = None
    
    progress_bar = st.progress(0)
    
    for i in range(image_token_num_per_image):
        progress_bar.progress(i / image_token_num_per_image)
        
        outputs = vl_gpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=pkv
        )
        pkv = outputs.past_key_values
        hidden_states = outputs.last_hidden_state
        logits = vl_gpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)
        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)
        
        if i % 50 == 0:
            clear_memory()
    
    progress_bar.progress(1.0)
    
    patches = vl_gpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, width // patch_size, height // patch_size]
    )
    return generated_tokens.to(dtype=torch.int), patches

def unpack(dec, width, height, parallel_size=1):
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)
    visual_img = np.zeros((parallel_size, width, height, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec
    return visual_img

@torch.inference_mode()
def generate_image(prompt, seed=None, guidance=5):
    clear_memory()
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    width = 384
    height = 384
    parallel_size = 1  # Reduced for memory constraints
    
    messages = [
        {'role': 'User', 'content': prompt},
        {'role': 'Assistant', 'content': ''}
    ]
    text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=messages,
        sft_format=vl_chat_processor.sft_format,
        system_prompt=''
    )
    text = text + vl_chat_processor.image_start_tag
    input_ids = torch.LongTensor(tokenizer.encode(text))
    output, patches = generate(
        input_ids,
        width // 16 * 16,
        height // 16 * 16,
        cfg_weight=guidance,
        parallel_size=parallel_size
    )
    images = unpack(patches, width // 16 * 16, height // 16 * 16, parallel_size)
    return [Image.fromarray(images[i]).resize((1024, 1024), Image.LANCZOS) for i in range(parallel_size)]

# Streamlit UI
def main():
    st.title("Janus Demo")
    
    tab1, tab2 = st.tabs(["Image Understanding", "Image Generation"])
    
    with st.sidebar:
        st.header("Parameters")
        with st.expander("Image Understanding Settings", expanded=True):
            seed = st.number_input("Seed", value=42)
            top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.95, step=0.05)
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        
        with st.expander("Image Generation Settings", expanded=True):
            gen_seed = st.number_input("Generation Seed", value=12345)
            cfg_weight = st.slider("CFG Weight", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
    
    with tab1:
        st.header("Image Understanding")
        uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        question = st.text_input("Ask a question about the image")
        
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Image")
            if st.button("Analyze"):
                with st.spinner("Analyzing..."):
                    response = multimodal_understanding(uploaded_image, question, seed, top_p, temperature)
                    st.text_area("Response", response, height=200)
    
    with tab2:
        st.header("Text-to-Image Generation")
        prompt = st.text_area("Enter your prompt")
        
        if st.button("Generate"):
            with st.spinner("Generating images..."):
                generated_images = generate_image(prompt, gen_seed, cfg_weight)
                for idx, img in enumerate(generated_images):
                    st.image(img, caption=f"Generated Image {idx+1}", use_column_width=True)

if __name__ == "__main__":
    main() 