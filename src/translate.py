import os
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MarianMTModel, MarianTokenizer
import pysrt

# Available translation models
TRANSLATION_MODELS = {
    "m2m100_418M": {"source_lang": "en", "target_lang": "fa", "model_type": "m2m100"},
    "nllb-200-distilled-600M": {"source_lang": "eng_Latn", "target_lang": "fas_Arab", "model_type": "nllb"},
    "mbart-large-50-many-to-many": {"source_lang": "en_XX", "target_lang": "fa_IR", "model_type": "mbart"},
    "custom": {"source_lang": "en", "target_lang": "fa", "model_type": "custom"}  # Default values, will be overridden
}

def load_translation_model(model_name, device=None, custom_model_path=None, custom_source_lang="en", custom_target_lang="fa"):
    """
    Load translation model based on model name
    
    Args:
        model_name (str): Name of the translation model
        device (str): Device to use for inference (cuda or cpu)
        custom_model_path (str): Path to custom model directory or HF model ID
        custom_source_lang (str): Source language code for custom model
        custom_target_lang (str): Target language code for custom model
        
    Returns:
        tuple: (model, tokenizer, source_lang, target_lang)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Handle custom model configuration
    if model_name == "custom":
        if not custom_model_path:
            raise ValueError("Custom model path must be provided when using 'custom' model")
        
        # Update custom model parameters
        TRANSLATION_MODELS["custom"]["source_lang"] = custom_source_lang
        TRANSLATION_MODELS["custom"]["target_lang"] = custom_target_lang
        
        # Try to auto-detect model type from the directory structure or metadata
        model_type = detect_model_type(custom_model_path)
        TRANSLATION_MODELS["custom"]["model_type"] = model_type
    elif model_name not in TRANSLATION_MODELS:
        raise ValueError(f"Translation model {model_name} not supported")
    
    source_lang = TRANSLATION_MODELS[model_name]["source_lang"]
    target_lang = TRANSLATION_MODELS[model_name]["target_lang"]
    model_type = TRANSLATION_MODELS[model_name]["model_type"]
    
    # Use the custom model path for custom models or standard path for built-in models
    model_path = custom_model_path if model_name == "custom" else f"facebook/{model_name}"
    
    print(f"Loading translation model {model_name} ({model_type}) from {model_path}...")
    
    # Load model based on the model type
    if model_type == "m2m100":
        tokenizer = M2M100Tokenizer.from_pretrained(model_path)
        model = M2M100ForConditionalGeneration.from_pretrained(model_path).to(device)
    elif model_type == "nllb":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    elif model_type == "mbart":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    elif model_type == "marian":
        tokenizer = MarianTokenizer.from_pretrained(model_path)
        model = MarianMTModel.from_pretrained(model_path).to(device)
    else:
        # Fall back to auto-detection
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    
    return model, tokenizer, source_lang, target_lang, model_type

def detect_model_type(model_path):
    """
    Attempt to detect the model type from a custom model path
    
    Args:
        model_path (str): Path to the model directory or HF model ID
        
    Returns:
        str: Detected model type
    """
    # Check if this is a local path
    if os.path.exists(model_path):
        # Try to find configuration files that indicate the model type
        if os.path.exists(os.path.join(model_path, "config.json")):
            import json
            with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
                config = json.load(f)
                
                if "architectures" in config:
                    arch = config["architectures"][0].lower()
                    if "m2m100" in arch:
                        return "m2m100"
                    elif "nllb" in arch or "nllb" in model_path.lower():
                        return "nllb"
                    elif "mbart" in arch:
                        return "mbart"
                    elif "marian" in arch:
                        return "marian"
    
    # Try to infer from the path name
    model_path_lower = model_path.lower()
    if "m2m100" in model_path_lower:
        return "m2m100"
    elif "nllb" in model_path_lower:
        return "nllb"
    elif "mbart" in model_path_lower:
        return "mbart"
    elif "marian" in model_path_lower or "opus" in model_path_lower:
        return "marian"
    
    # Default to auto-detection (will use AutoModel)
    print(f"Could not determine model type for {model_path}. Using auto-detection.")
    return "auto"

def translate_text(text, model, tokenizer, source_lang, target_lang, model_type="auto"):
    """
    Translate text using the specified model
    
    Args:
        text (str): Text to translate
        model: Translation model
        tokenizer: Tokenizer for the model
        source_lang (str): Source language code
        target_lang (str): Target language code
        model_type (str): Type of the model (m2m100, nllb, mbart, marian, auto)
        
    Returns:
        str: Translated text
    """
    if not text.strip():
        return ""
        
    # Set the source language if applicable
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = source_lang
    
    # Different handling based on model type
    if model_type == "m2m100":
        # M2M100 model
        tokenizer.src_lang = source_lang
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        gen_kwargs = {"forced_bos_token_id": tokenizer.get_lang_id(target_lang)}
    elif model_type == "nllb":
        # NLLB model
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        gen_kwargs = {"forced_bos_token_id": tokenizer.lang_code_to_id[target_lang]}
    elif model_type == "mbart":
        # mBART model
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        gen_kwargs = {"forced_bos_token_id": tokenizer.lang_code_to_id[target_lang]}
    elif model_type == "marian":
        # Marian model
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        gen_kwargs = {}
    else:
        # Auto-detection (try different methods based on available attributes)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        if hasattr(tokenizer, "get_lang_id"):
            gen_kwargs = {"forced_bos_token_id": tokenizer.get_lang_id(target_lang)}
        elif hasattr(tokenizer, "lang_code_to_id") and target_lang in tokenizer.lang_code_to_id:
            gen_kwargs = {"forced_bos_token_id": tokenizer.lang_code_to_id[target_lang]}
        else:
            gen_kwargs = {}
    
    # Generate the translation
    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs, max_length=512)
        
        # Decode the translation
        translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return translation
    except Exception as e:
        print(f"Translation error: {str(e)}")
        print(f"Falling back to original text for: '{text[:50]}...'")
        return text  # Return original text on error

def translate_subtitles(srt_path, model_name="m2m100_418M", output_dir="output", device=None,
                        custom_model_path=None, custom_source_lang="en", custom_target_lang="fa"):
    """
    Translate subtitles to Persian or other target language
    
    Args:
        srt_path (str): Path to the SRT file
        model_name (str): Name of the translation model
        output_dir (str): Directory to save the translated subtitle file
        device (str): Device to use for inference (cuda or cpu)
        custom_model_path (str): Path to custom model directory or HF model ID
        custom_source_lang (str): Source language code for custom model
        custom_target_lang (str): Target language code for custom model
        
    Returns:
        str: Path to the translated subtitle file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    model, tokenizer, source_lang, target_lang, model_type = load_translation_model(
        model_name, device, custom_model_path, custom_source_lang, custom_target_lang
    )
    
    # Get language code for filename (default to 'fa' if not provided)
    lang_code = custom_target_lang if model_name == "custom" else target_lang.split('_')[0]
    if lang_code in ["fas_Arab", "fa_IR"]:
        lang_code = "fa"
    
    # Load the SRT file
    print(f"Loading subtitles from {srt_path}")
    subs = pysrt.open(srt_path, encoding='utf-8')
    
    # Translate each subtitle
    total_subs = len(subs)
    print(f"Translating {total_subs} subtitles to {target_lang}")
    
    for i, sub in enumerate(subs):
        if i % 10 == 0:  # Progress update every 10 subtitles
            print(f"Translating subtitle {i+1}/{total_subs}")
        sub.text = translate_text(sub.text, model, tokenizer, source_lang, target_lang, model_type)
    
    # Save the translated SRT file
    translated_srt_path = os.path.join(output_dir, os.path.splitext(os.path.basename(srt_path))[0] + f'_{lang_code}.srt')
    subs.save(translated_srt_path, encoding='utf-8')
    
    print(f"Translation completed. Saved to {translated_srt_path}")
    return translated_srt_path