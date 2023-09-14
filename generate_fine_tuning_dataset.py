from typing import Optional

import fire
import numpy as np
import pandas as pd

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    skills_dataset: str = "skills.csv",
    max_n_skills: int = 5,
    batch_size: int = 1,
    iterations: int = 1,
    dataset_file: Optional[str] = None,
):

    skills_entire_list = pd.read_csv(skills_dataset, header=None).values.ravel()
    df = pd.DataFrame()

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    for i in range(iterations):

        samples = []
        strings = []
        prompts = []

        for b in range(batch_size):
            n_skills = np.random.randint(1, max_n_skills + 1)
            sample = np.random.choice(skills_entire_list, size=n_skills)
            samples.append(list(sample))
            string = "[ " + " | ".join(sample) + " ]"
            strings.append(string)
            prompts.append(f"Generate a very short description of a course that allows one to acquire the following {n_skills} professional skills taken from the Australian Skills Classification Framework: {string}")

        dialogs = [
            [
                {
                    "role": "system",
                    "content": 'Do not start the response with "Sure, here is..." or "Here is...", rather provide the description directly'
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ] for prompt in prompts
        ]

        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        descriptions = [result["generation"]["content"].strip() for result in results]
        df = pd.concat([df, pd.DataFrame({"Description": descriptions, "Skills": samples})])

    if dataset_file is not None:
        df.to_csv(dataset_file, index=False)
        print(f"{dataset_file} saved!")
    else:
        for (i, (description, sample)) in enumerate(zip(descriptions, samples)):
            print("Skills:")
            for skill in sample:
                print(f"- {skill}")
            print(f"\nDescription:\n{description}")
            if i != batch_size - 1:
                print("\n===============\n")


if __name__ == "__main__":
    fire.Fire(main)
