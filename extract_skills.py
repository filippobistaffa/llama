from typing import Optional

import fire
import numpy as np

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    description_file: str = "description.txt",
):

    with open(description_file) as f:
        description = "".join(f.readlines()).strip()

    prompt = f'Extract a list of "specialist tasks" belonging to the Australian Skills Classification that can be acquired with a course with the following description:\n\n{description}'

    dialogs = [
        [
            {
                "role": "system",
                "content": 'Only answer with "specialist tasks" from the Australian Skills Classification'
            },
            {
                "role": "user",
                "content": prompt
            },
        ]
    ]

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    answers = [result["generation"]["content"].strip() for result in results]
    for answer in answers:
        print(answer)


if __name__ == "__main__":
    fire.Fire(main)
