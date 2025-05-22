# Using Custom Translation Models

This guide explains how to use your own custom translation models with the Video Subtitle Translator application.

## Supported Model Types

The application supports the following types of translation models:

1. **M2M100** - Facebook's Many-to-Many multilingual translation models
2. **NLLB** - No Language Left Behind models
3. **mBART** - Multilingual BART models
4. **MarianMT** - Marian Machine Translation models (including OPUS-MT)
5. **Other Seq2Seq models** - Any model compatible with Hugging Face's `AutoModelForSeq2SeqLM`

## How to Use Custom Models

### Option 1: Using Hugging Face Models

You can use any translation model from the [Hugging Face Model Hub](https://huggingface.co/models?pipeline_tag=translation):

1. Select "custom" from the Translation Model dropdown
2. In the "Custom Model Path/HF Model ID" field, enter the Hugging Face model ID
   - Example: `Helsinki-NLP/opus-mt-en-fa` for English to Persian translation
   - Example: `facebook/nllb-200-distilled-600M` for NLLB model
3. Set the source and target language codes according to the model's requirements
   - For OPUS-MT models: `en`, `fa`, etc.
   - For M2M100: `en`, `fa`, etc.
   - For NLLB: `eng_Latn`, `fas_Arab`, etc. 
   - For mBART: `en_XX`, `fa_IR`, etc.

### Option 2: Using Locally Saved Models

You can also use models that you've fine-tuned and saved locally:

1. Save your model using Hugging Face's `save_pretrained()` method
2. Select "custom" from the Translation Model dropdown
3. In the "Custom Model Path/HF Model ID" field, enter the full path to your model directory
   - Example: `C:\my_models\my_english_to_persian_model`
   - Example: `/home/user/models/my_english_to_persian_model`
4. Set the appropriate source and target language codes

## Language Codes

Different model types use different language code formats:

- **M2M100**: `en`, `fa`, `ar`, etc.
- **NLLB**: `eng_Latn`, `fas_Arab`, `ara_Arab`, etc.
- **mBART**: `en_XX`, `fa_IR`, `ar_AR`, etc.
- **MarianMT/OPUS**: `en`, `fa`, `ar`, etc.

Make sure to use the correct format for your specific model.

## Troubleshooting

If you encounter issues:

1. **Model loading errors**: Check if the model path or Hugging Face ID is correct
2. **Language code errors**: Verify you're using the correct language code format for your model type
3. **Out of memory errors**: Try using a smaller model or reducing batch size
4. **Incorrect translations**: Ensure the model is actually trained for your language pair

## Examples of Good Translation Models for Persian

- `Helsinki-NLP/opus-mt-en-fa` - OPUS-MT English to Persian
- `facebook/m2m100_418M` - M2M100 multilingual (418M parameters)
- `facebook/nllb-200-distilled-600M` - NLLB model with 200 languages