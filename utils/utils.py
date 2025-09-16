import os
import json
import random
import json
import os
import numpy as np
from pathlib import Path
from typing import Iterable, Union, Any


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def pre_load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    return list(pre_load_jsonl(file))


def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved to", save_path)


def read_json(fname):
    with open(fname, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, fname, indent=None):
    with open(fname, "w", encoding="utf-8") as f:
        print("Saved to", fname)
        return json.dump(obj, f, indent=indent, ensure_ascii=False)


def load_prompt(data_name, prompt_type):
    if data_name == "math":
        if prompt_type in ["cot", "pot"]:
            prompt_path = rf"prompts/math/{prompt_type}.md"
        elif prompt_type in [
            "self-generated knowledge cot",
            "self-generated knowledge pot",
        ]:
            prompt_path = (
                rf"prompts/math/self-generated knowledge/{prompt_type.split()[-1]}.md"
            )
        elif prompt_type in ["knowledge cot", "knowledge pot"]:
            prompt_path = rf"prompts/math/knowledge/{prompt_type.split()[-1]}.md"
        elif prompt_type in ["summarize knowledge cot", "summarize knowledge pot"]:
            prompt_path = (
                rf"prompts/math/summarize knowledge/{prompt_type.split()[-1]}.md"
            )
        elif prompt_type == "math_make_tool":
            prompt_path = rf"prompts/math/tool/tool_make.md"
        elif prompt_type == "math_use_tool":
            prompt_path = rf"prompts/math/tool/tool_use.md"
        elif prompt_type in ["math_use_tool_by_calling"]:
            prompt_path = rf"prompts/math/tool/tool_use_by_calling.md"
        elif prompt_type == "math_use_tool_by_retrieval":
            prompt_path = rf"prompts/math/tool/tool_use_by_check.md"
        elif prompt_type == "question_parse":
            prompt_path = rf"prompts/math/question_parse.md"
        elif prompt_type == "tool_planing":
            prompt_path = rf"prompts/math/tool/tool_planing.md"
        elif prompt_type == "math_make_tool_from_pot":
            prompt_path = rf"prompts/math/tool/tool_make_from_pot.md"
        else:
            prompt_path = None

    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as fp:
            prompt = fp.read().strip() + "\n\n"
    else:
        print(f"Error: prompt file {prompt_path} not found")
        prompt = ""
    return prompt


if __name__ == "__main__":
    dump_json({}, "data/tpot/result_3/1.json", indent=None)
