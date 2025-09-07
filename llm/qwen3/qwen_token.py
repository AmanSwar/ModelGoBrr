import re
from tokenizers import Tokenizer

from pathlib import Path

class Qwen3Tokenizer:
    #reserved tokens -> SPECEAL
    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
        "<think>", "</think>"
    ]
    #split the specials
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>|<think>|</think>)")

    def __init__(
        self,
        tokenizer_file_path="llm/Qwen3-0.6B/tokenizer.json",
        repo_id=None,
        apply_chat_template=True,
        add_gen_prompt=False,
        add_thinking=False
    ):
        self.apply_chat_template = apply_chat_template
        self.add_gen_prompt = add_gen_prompt
        self.add_thinking = add_thinking

        tok_file_path = Path(tokenizer_file_path)

        #init base tokenizer
        self.tok = Tokenizer.from_file(str(tok_file_path))

        #spcl tokens ids
        self.spcl_to_id = {}

        for t in self._SPECIALS:
            tid = self.tok.token_to_id(t)
            if tid is not None:
                self.spcl_to_id[t] = tid

        self.pad_token_id = self.spcl_to_id["<|endoftext|>"]
        self.eos_token_id = self.pad_token_id

        eos_token = "<|im_end|>"

        if eos_token in self.spcl_to_id:
            self.eos_token_id = self.spcl_to_id[eos_token]

    def _wrap_chat(self , msg):
        s = f"<|im_start|>user\n{msg}<|im_end|>\n"
        if self.add_gen_prompt:
            s += "<|im_start|>assistant"
            if self.add_thinking:
                s += "\n"
            else:
                s += "\n<think>\n\n</think>\n\n"
        return s

    def encode( 
        self,
        text : str,
        chat_wrapped = None
    ):

        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template

        stripped_text = text.strip()

        #check if text is a spcl token
        if stripped_text in self.spcl_to_id and "\n" not in stripped_text:
            return [self.spcl_to_id[stripped_text]]

        if chat_wrapped:
            text = self._wrap_chat(text)

        ids = []

        for part in filter(None , self._SPLIT_RE.split(text)):
            if part in self.spcl_to_id:
                ids.append(self.spcl_to_id[part])

            else:
                ids.extend(self.tok.encode(part).ids)

        return ids
    

    def decode(self,  ids):

        return self.tok.decode(ids , skip_special_tokens=False)


if __name__ == "__main__":
    tokenizer_file_path = "/home/aman/code/model_go_brr/llm/Qwen3-0.6B/tokenizer.json"

    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_file_path,
        add_gen_prompt=True,
        add_thinking=True,
    )

    prompt = "How many 'r' is in 'Strawberry'"

    input_token_ids = tokenizer.encode(prompt)
    text = tokenizer.decode(input_token_ids)
    print(text)

    print(input_token_ids)
