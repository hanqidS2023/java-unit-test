from unsloth import FastLanguageModel

class LlamaModel:
    def __init__(self):
        model_path = "llama-3"
        max_seq_length = 2048
        dtype = None
        load_in_4bit = True
        
        self.prompt = """
        ### src_fm:
        {}
        ### src_fm_fc_ms_ff:
        {}
        ### target:
        {}
        """
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )
        FastLanguageModel.for_inference(self.model) # Enable native 2x faster inference

    def predict(self, fm, fm_fc_ms_ff):
        generated_text = self.gernerate_text(fm=fm, fm_fc_ms_ff=fm_fc_ms_ff)
        parsed_generated_text =  self.parse_for_demo(generated_text)
        return parsed_generated_text

    def gernerate_text(self, fm = "", fm_fc_ms_ff = ""):

        inputs = self.tokenizer(
            [
        self.prompt.format(
            fm, # input fm
            fm_fc_ms_ff, # fm_fc_ms_ff
            "", # output
        )
        ], return_tensors = "pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens = 256, use_cache = True)
        return self.tokenizer.batch_decode(outputs)[0]

    def parse_for_demo(self, input_string):
    
        def remove_before_first_target(input_string):
            target_index = input_string.find('### target:')
            return input_string[target_index:] if target_index != -1 else input_string
        def remove_after_first_num(input_string):
            target_index = input_string.find('###')
            return input_string[:target_index] if target_index != -1 else input_string
        def extract_target_content(input_string):
            segments = input_string.split('### target:')
            extracted_content = []
            for segment in segments[1:]:
                end_index = segment.find('### target:')
                if end_index == -1:
                    seg_stripped = segment.strip()
                else:
                    seg_stripped = segment[:end_index].strip()
                if seg_stripped: 
                    extracted_content.append(seg_stripped)
            return extracted_content
        
        input_string = remove_before_first_target(input_string)
        extracted_content = extract_target_content(input_string)
        combined_string = "\n".join( extracted_content)
        combined_string = remove_after_first_num(combined_string)
        return combined_string