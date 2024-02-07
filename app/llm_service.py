import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Iterator, Union

from app.datamodel import ChatMessage
from app.settings import (
    llm_model,
    timing_decorator,
    has_mps,
    disable_grad,
)


class LLMService:
    @timing_decorator
    def __init__(self):
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            LlamaTokenizerFast,
        )
        import torch

        disable_grad()

        kwargs = dict(
            device_map="mps" if has_mps else "cpu",
            torch_dtype=torch.float16 if has_mps else torch.float32,
        )
        logging.info("Starting with args: %s", kwargs)
        self._llm = AutoModelForCausalLM.from_pretrained(
            llm_model,
            **kwargs,
        )

        logging.info("Model Loaded, Device Map: %s", self._llm.hf_device_map)

        self._tokenizer: Union[
            AutoTokenizer, LlamaTokenizerFast
        ] = AutoTokenizer.from_pretrained(llm_model)
        self._device = "mps" if has_mps else "cpu"
        self._tpe = ThreadPoolExecutor(1, initializer=disable_grad)

    @timing_decorator
    def complete(self, data: List[ChatMessage], **kwargs) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        # only take last 10 messages as context due to OOM error
        text_template = self._tokenizer.apply_chat_template(
            [d.model_dump() for d in data[-10:]], tokenize=False
        )
        logging.info("Prompt: %s", text_template)
        encoded_values = self._tokenizer.encode(text_template, return_tensors="pt")
        if "max_input_length" in kwargs:
            encoded_values = encoded_values[:, -kwargs.get("max_input_length", 2048) :]
        encoded_values = encoded_values.to(self._device)
        logging.info("Prompt Size: %d", encoded_values.shape[-1])
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True)

        yield "<START>"
        # issue the normal call in background
        future = self._tpe.submit(
            lambda: self._llm.generate(encoded_values, streamer=streamer, **kwargs)
        )
        message_buffer = []
        last_res = None
        for token in streamer:
            logging.info("Got reply token: %s", token)
            message_buffer.append(token)
            new_res = "".join(message_buffer).rstrip("</s>").replace("  ", " ")
            if not last_res or last_res != new_res:
                yield new_res
                last_res = new_res

        yield "<FINISH>"

        logging.info("Finished streaming")


if __name__ == "__main__":
    llm = LLMService()

    dx = []
    for text in llm.complete(
        [ChatMessage(role="user", content="Whats the story behind the Eiffel tower?")],
        max_new_tokens=5,
    ):
        dx.append(text)

    print(dx)
