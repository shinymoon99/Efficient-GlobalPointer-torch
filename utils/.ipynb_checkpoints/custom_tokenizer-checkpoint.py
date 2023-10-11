from transformers import BertTokenizerFast

# Define your unknown token
unknown_token = "[UNK]"

# Create a custom tokenizer class
class CustomBertTokenizer(BertTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def tokenize(self, text, max_length=None, truncation=True):
        # Split the text into chunks using Chinese characters as separators
        chunks = []
        current_chunk = ""
        
        for char in text:
            if ord(char) < 128:
                current_chunk += char
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = ""
                chunks.append(char)
        
        if current_chunk:
            chunks.append(current_chunk)
        
        # Tokenize the chunks
        tokenized_text = []
        for chunk in chunks:
            if all(ord(char) < 128 for char in chunk):
                tokenized_text.append(unknown_token)
            else:
                sub_tokens = super().tokenize(chunk)
                tokenized_text.extend(sub_tokens)
        
        # Calculate offsets mapping
        offset_mapping = []
        offset = 0
        for token in tokenized_text:
            offset_mapping.append((offset, offset + len(token)))
            offset += len(token)
        
        return {
            "offset_mapping": offsets,
        }
    def encode_plus_with_mapping(self, text, max_length=None, truncation=True):
        # Split the text into chunks using Chinese characters as separators
        chunks = []
        current_chunk = ""
        
        for char in text:
            if ord(char) < 128:
                current_chunk += char
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = ""
                chunks.append(char)
        
        if current_chunk:
            chunks.append(current_chunk)
        
        # Tokenize the chunks
        tokenized_text = []
        offsets = []
        offset = 0
        for chunk in chunks:
            if all(ord(char) < 128 for char in chunk):
                tokenized_text.append(unknown_token)
                offsets.append((offset, offset + len(chunk)))
            else:
                sub_tokens = super().tokenize(chunk)
                tokenized_text.extend(sub_tokens)
                offsets.extend([(offset, offset + len(sub_token)) for sub_token in sub_tokens])
            offset += len(chunk)
        
        # Truncate and pad if necessary
        if max_length is not None and len(tokenized_text) > max_length:
            tokenized_text = tokenized_text[:max_length]
            offsets = offsets[:max_length]
        
        # Create input features
        input_ids = super().convert_tokens_to_ids(tokenized_text)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offsets,
        }
