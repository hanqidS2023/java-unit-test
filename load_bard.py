import torch
import re
from transformers import BartForConditionalGeneration, BartTokenizer

class BartModel:
    def __init__(self):
        self.model_name = "facebook/bart-large"
        # Load model into memory
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
        # Load tokenizer into memory
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self._load_model_from_file()

    # Load the state dict from the saved model file
    def _load_model_from_file(self):
        saved_model_file_path = "model_facebook_bart-large_1713941998.pth"
        state_dict = torch.load(saved_model_file_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        print(f"Loaded model: {saved_model_file_path}\n")

    # Remove comments from Java code
    def _remove_java_comments(self, java_code):  # Remove single-line comments
        # Remove single-line comments
        java_code = re.sub(r'//.*', '', java_code)

        # Remove multi-line comments
        java_code = re.sub(r'/\*.*?\*/', '', java_code, flags=re.DOTALL)

        # Remove newlines and additional spaces
        java_code = ' '.join(java_code.split())
        return java_code

    # Generate code completions for the input code snippet
    def get_prediction_test_by_input_string(self, input_string):
        input_string = self._remove_java_comments(input_string)
        inputs = self.tokenizer(input_string, padding="max_length", truncation=True, max_length=256)
        input_ids = torch.tensor(inputs.input_ids).unsqueeze(0).to(self.model.device)
        attention_mask = torch.tensor(inputs.attention_mask).unsqueeze(0).to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=256,
                num_beams=4,
                early_stopping=False
            )

        prediction_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return prediction_text