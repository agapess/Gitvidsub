import os
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MarianMTModel, MarianTokenizer, MT5ForConditionalGeneration
import pysrt
import re

# Available translation models
TRANSLATION_MODELS = {
    "m2m100_418M": {"source_lang": "en", "target_lang": "fa", "model_type": "m2m100"},
    "nllb-200-distilled-600M": {"source_lang": "eng_Latn", "target_lang": "fas_Arab", "model_type": "nllb"},
    "mbart-large-50-many-to-many": {"source_lang": "en_XX", "target_lang": "fa_IR", "model_type": "mbart"},
    # New high-quality translation models
    "mt5-parsinlu": {
        "source_lang": "en", 
        "target_lang": "fa", 
        "model_type": "mt5",
        "model_path": "persiannlp/mt5-base-parsinlu-translation_en_fa",
        "description": "MT5-base fine-tuned on ParsiNLU for English to Persian translation"
    },
    "mt5-parsinlu-large": {
        "source_lang": "en", 
        "target_lang": "fa", 
        "model_type": "mt5",
        "model_path": "persiannlp/mt5-large-parsinlu-translation_en_fa",
        "description": "MT5-large fine-tuned on ParsiNLU for English to Persian translation (higher quality)"
    },
    "nllb-1B8": {
        "source_lang": "eng_Latn", 
        "target_lang": "fas_Arab", 
        "model_type": "nllb",
        "model_path": "facebook/nllb-200-1.3B",
        "description": "NLLB 1.3B model for high-quality translation with 200 languages including Persian"
    },
    "custom": {"source_lang": "en", "target_lang": "fa", "model_type": "custom"}  # Default values, will be overridden
}

def list_available_models():
    """List all available translation models with descriptions"""
    print("Available translation models:")
    for name, details in TRANSLATION_MODELS.items():
        if name != "custom":  # Don't show the custom placeholder
            description = details.get("description", "")
            print(f"- {name}: {description}")

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
        tuple: (model, tokenizer, source_lang, target_lang, model_type)
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
        model_path = custom_model_path
    elif model_name not in TRANSLATION_MODELS:
        raise ValueError(f"Translation model {model_name} not supported. Use list_available_models() to see available models.")
    else:
        # Use model details from the dictionary
        model_config = TRANSLATION_MODELS[model_name]
        source_lang = model_config["source_lang"]
        target_lang = model_config["target_lang"]
        model_type = model_config["model_type"]
        # Use specific model path if provided, otherwise use default pattern
        if "model_path" in model_config:
            model_path = model_config["model_path"]
        else:
            model_path = f"facebook/{model_name}"
    
    print(f"Loading translation model {model_name} ({model_type}) from {model_path}...")
    print(f"Source language: {source_lang}, Target language: {target_lang}")
    
    try:
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
        elif model_type == "mt5":
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = MT5ForConditionalGeneration.from_pretrained(model_path).to(device)
        else:
            # Fall back to auto-detection
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    
        return model, tokenizer, source_lang, target_lang, model_type
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Make sure you have a good internet connection and enough disk space.")
        raise

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
                    elif "mt5" in arch or "t5" in arch:
                        return "mt5"
    
    # Try to infer from the path name
    model_path_lower = model_path.lower()
    if "m2m100" in model_path_lower:
        return "m2m100"
    elif "nllb" in model_path_lower:
        return "nllb"
    elif "mbart" in model_path_lower:
        return "mbart"
    elif "mt5" in model_path_lower or "t5" in model_path_lower:
        return "mt5"
    elif "marian" in model_path_lower or "opus" in model_path_lower:
        return "marian"
    elif "persian" in model_path_lower or "parsi" in model_path_lower or "fa" in model_path_lower:
        # Heuristic for Persian-specific models, most likely to be MT5 based
        if "translation" in model_path_lower:
            return "mt5"
    
    # Default to auto-detection (will use AutoModel)
    print(f"Could not determine model type for {model_path}. Using auto-detection.")
    return "auto"

def clean_mt5_translation(translation, original_text):
    """
    Clean MT5 translation by removing common artifacts
    
    Args:
        translation (str): The translated text that might contain artifacts
        original_text (str): The original text that was translated
        
    Returns:
        str: Cleaned translated text
    """
    # Remove the specific prefix "ترجمه انگلیسی به فارسی :" (Translation from English to Persian:)
    translation = re.sub(r'^ترجمه انگلیسی به فارسی\s*:?\s*', '', translation)
    
    # Also try other variations of the prefix pattern
    translation = re.sub(r'^ترجمه از انگلیسی به فارسی\s*:?\s*', '', translation)
    translation = re.sub(r'^ترجمه\s*:?\s*', '', translation)
    
    # Remove "translated from: [original text]" pattern that appears in some MT5 models
    translation = re.sub(r'(?:translated from:?\s*|ترجمه از:?\s*).*$', '', translation, flags=re.IGNORECASE | re.DOTALL)
    
    # Try to detect if the original text is included at the end and remove it
    if original_text and len(original_text) > 3 and original_text in translation:
        translation = translation.replace(original_text, "").strip()
    
    # Remove any "Persian:" or "Farsi:" prefixes
    translation = re.sub(r'^(?:persian:?|farsi:?)\s*', '', translation, flags=re.IGNORECASE)
    
    # Check for any common prefix patterns
    prefixes = [
        "english to persian:", "english to persian translation:",
        "translated text:", "translation:", "result:", "output:",
        "en to fa:", "en->fa:", "english->persian:"
    ]
    
    for prefix in prefixes:
        if translation.lower().startswith(prefix):
            translation = translation[len(prefix):].strip()
    
    # Fix excess whitespace
    translation = re.sub(r'\s+', ' ', translation).strip()
    
    return translation

def test_translation_on_sample(model, tokenizer, model_type, source_lang, target_lang):
    """Test the translation on a sample text to understand output format"""
    sample_text = "Hello, this is a test."
    print("\nTesting translation on sample text to detect format issues:")
    print(f"Input: '{sample_text}'")
    
    # Do raw translation to see exactly what model outputs
    if model_type == "mt5":
        prefix = "translate English to Persian: "
        inputs = tokenizer(prefix + sample_text, return_tensors="pt").to(model.device)
    else:
        # Generic handling for other model types
        inputs = tokenizer(sample_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128)
    
    raw_translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"Raw output: '{raw_translation}'")
    
    # Clean up the translation
    clean_translation = clean_mt5_translation(raw_translation, sample_text)
    print(f"Cleaned output: '{clean_translation}'")
    print("Format detection complete. Adjusting cleanup patterns based on results.\n")
    
    # Analyze the pattern and update the clean_mt5_translation function if needed
    if raw_translation != clean_translation:
        # Found a pattern to clean up, extract it for debugging
        prefix_part = raw_translation[:raw_translation.find(clean_translation)].strip()
        if prefix_part:
            print(f"Detected prefix pattern to remove: '{prefix_part}'")
        
        # If there's still a prefix in the cleaned output, we need to update our patterns
        common_prefixes = ["ترجمه", "translation", "persian", "farsi", "translate"]
        for prefix in common_prefixes:
            if clean_translation.lower().startswith(prefix.lower()):
                print(f"Warning: Cleanup might not be complete. Still starts with '{prefix}'")

def translate_text(text, model, tokenizer, source_lang, target_lang, model_type="auto"):
    """
    Translate text using the specified model
    
    Args:
        text (str): Text to translate
        model: Translation model
        tokenizer: Tokenizer for the model
        source_lang (str): Source language code
        target_lang (str): Target language code
        model_type (str): Type of the model (m2m100, nllb, mbart, marian, mt5, auto)
        
    Returns:
        str: Translated text
    """
    if not text.strip():
        return ""
    
    # Store original text for post-processing cleanup
    original_text = text.strip()
    
    # Pre-process text for better translation
    # Remove excessive whitespace that might confuse the model
    text = " ".join(text.split())
    
    # Set the source language if applicable
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = source_lang
    
    # Different handling based on model type
    if model_type == "m2m100":
        # M2M100 model
        tokenizer.src_lang = source_lang
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        gen_kwargs = {
            "forced_bos_token_id": tokenizer.get_lang_id(target_lang),
            "num_beams": 5,  # Use beam search for better quality
            "length_penalty": 1.0  # Slightly prefer longer translations for completeness
        }
    elif model_type == "nllb":
        # NLLB model
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        gen_kwargs = {
            "forced_bos_token_id": tokenizer.lang_code_to_id[target_lang],
            "num_beams": 5,
            "length_penalty": 1.0
        }
    elif model_type == "mbart":
        # mBART model
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        gen_kwargs = {
            "forced_bos_token_id": tokenizer.lang_code_to_id[target_lang],
            "num_beams": 5,
            "length_penalty": 1.0
        }
    elif model_type == "marian":
        # Marian model
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        gen_kwargs = {
            "num_beams": 5,
            "length_penalty": 1.0
        }
    elif model_type == "mt5":
        # MT5 model - for persiannlp/mt5-base-parsinlu-translation_en_fa
        # Format input as expected by the model, but try without the prefix
        # This might help avoid the model adding its own prefix to the output
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        gen_kwargs = {
            "num_beams": 5,
            "length_penalty": 1.0,
            "do_sample": False,  # Disable sampling for more deterministic output
            "early_stopping": True  # Stop when all beam hypotheses reached the EOS token
        }
    else:
        # Auto-detection (try different methods based on available attributes)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        gen_kwargs = {"num_beams": 5, "length_penalty": 1.0}
        
        if hasattr(tokenizer, "get_lang_id"):
            gen_kwargs["forced_bos_token_id"] = tokenizer.get_lang_id(target_lang)
        elif hasattr(tokenizer, "lang_code_to_id") and target_lang in tokenizer.lang_code_to_id:
            gen_kwargs["forced_bos_token_id"] = tokenizer.lang_code_to_id[target_lang]
    
    # Generate the translation
    try:
        with torch.no_grad():
            # Add max_length parameter to avoid generating too long sequences
            # but make it generous enough for typical subtitles
            outputs = model.generate(**inputs, **gen_kwargs, max_length=256)
        
        # Decode the translation
        translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Apply aggressive cleaning to remove prefixes, especially for MT5 models
        translation = clean_mt5_translation(translation, original_text)
        
        return translation
    except Exception as e:
        print(f"Translation error: {str(e)}")
        print(f"Falling back to original text for: '{text[:50]}...'")
        return text  # Return original text on error

def translate_subtitles(srt_path, model_name="mt5-parsinlu", output_dir="output", device=None,
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
    
    # Test translation on sample text to understand format issues
    if model_type == "mt5":
        test_translation_on_sample(model, tokenizer, model_type, source_lang, target_lang)
    
    # Get language code for filename (default to 'fa' if not provided)
    lang_code = custom_target_lang if model_name == "custom" else target_lang.split('_')[0]
    if lang_code in ["fas_Arab", "fa_IR"]:
        lang_code = "fa"
    
    # Load the SRT file
    print(f"Loading subtitles from {srt_path}")
    subs = pysrt.open(srt_path, encoding='utf-8')
    
    # Translate each subtitle
    total_subs = len(subs)
    print(f"Translating {total_subs} subtitles to {target_lang} using {model_name} model")
    
    for i, sub in enumerate(subs):
        if i % 10 == 0:  # Progress update every 10 subtitles
            print(f"Translating subtitle {i+1}/{total_subs}")
        sub.text = translate_text(sub.text, model, tokenizer, source_lang, target_lang, model_type)
    
    # Save the translated SRT file
    translated_srt_path = os.path.join(
        output_dir, 
        os.path.splitext(os.path.basename(srt_path))[0] + f'_{lang_code}.srt'
    )
    subs.save(translated_srt_path, encoding='utf-8')
    
    print(f"Translation completed. Saved to {translated_srt_path}")
    return translated_srt_path