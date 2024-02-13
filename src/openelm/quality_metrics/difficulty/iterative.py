import torch
import ast
import json
import jsonlines
from openelm.quality_metrics.difficulty.judge import judge_parallel

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, CodeLlamaTokenizer


iterative_prompt_template = """Your task is to solve a programming puzzle. A programming puzzle is defined through a function f that takes as input a first argument which is the solution, as well as some additional keyword arguments. To solve the puzzle, you must write a function g, taking the same set of keyword arguments, such that `f(g()) is True`

Let's see some examples of puzzles and solutions first:

```python
{few_shot_examples}
```

You will have several attempts to write a solution. I'll execute `assert f(g())` and if the solution doesn't work you will have an opportunity to get feedback and write another solution. Remember you only need to write the `g` function.
The puzzle is:

```python
{puzzle}
```

Solution attempt 1:"""


g_output_feedback_template = """The Python interpreter gives the following feedback:
```python
>>> g()
>>> {g_output}
```
Let's try this again.

Solution attempt {num_attempt}:"""


g_not_parsed_feedback_template = """The Python interpreter gives the following feedback:
```python
>>> {g_parse_error}
```
Let's try this again.

Solution attempt {num_attempt}:"""


empty_g_feedback_template = """No valid Python function has been created.:
Let's try this again.

Solution attempt {num_attempt}:"""


def extract_g(string_outs):
    try:
        parsed = ast.parse(string_outs.split('```python')[-1].split('```')[0])
        functions = [el for el in parsed.body if isinstance(el, ast.FunctionDef)]
        if not functions:
            return False, ''
        return True, ast.unparse(functions[0])
        # get first function
    except SyntaxError as e:
        return False, e


def solve_iterative(puzzle_str, model, tokenizer, max_attempts, prompt_template, few_shot_examples, max_new_tokens=512):
    attempt_number = 1
    prompt = prompt_template.format(few_shot_examples=few_shot_examples, puzzle=puzzle_str)
    solution_list = []

    # interleave model generation with interpreter output
    while attempt_number < max_attempts:
        inputs = tokenizer(prompt, return_tensors='pt', ).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.8)
        string_outputs = tokenizer.batch_decode(outputs)[0]
        # maybe apply some filtering here

        # extract g
        successful_extract, res = extract_g(string_outputs)
        if not successful_extract:
            # we got a syntax error, give this as feedback
            attempt_number += 1
            if not res:
                feedback = empty_g_feedback_template.format(num_attempt=attempt_number)
            else:
                feedback = g_not_parsed_feedback_template.format(g_parse_error=str(res), num_attempt=attempt_number)
            prompt += '\n'
            prompt += feedback
        else:
            # g parsed fine, try to see if the solution is valid
            solution_list.append(res)
            src_code = f"{puzzle_str}\n\n{res}\n\nassert(f(g()))"
            judgements = judge_parallel([src_code])
            if judgements[0]:
                # success, exit
                return True, attempt_number, solution_list
            else:
                # failure, give feedback and restart
                attempt_number += 1
                g_output = f"{res}\ng()"
                feedback = g_output_feedback_template.format(g_output=g_output, num_attempt=attempt_number)
                prompt += '\n'
                prompt += feedback

    # we reached the end, return
    return False, attempt_number, solution_list


def iterative_solving_loop(
        puzzle_path,
        model_id,
        max_attempts=5,  # maximum amounts of feedback calls to the python interpreter
):
    # load file
    all_puzzles = json.load(open(puzzle_path, 'r'))
    few_shot_examples = open('difficulty/few_shot_examples.txt', 'r').read()

    # define out name
    puzzle_save_path_jsonl = puzzle_path.replace('.json', '_iterative_solved.jsonl')
    puzzle_save_path_json = puzzle_path.replace('.json', '_iterative_solved.json')
    puzzles_to_save = []
    total_puzzles_to_save = []

    # load model and tokneizer
    if 'codellama' in model_id:
        tokenizer = CodeLlamaTokenizer.from_pretrained(model_id, local_files_only=True)
    elif 'llama' in model_id:
        tokenizer = LlamaTokenizer.from_pretrained(model_id, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        # quantization_config=quantization_config,
        device_map="auto",
        local_files_only=True
    )
    # model.cuda()
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model.config.use_cache = True
    model = torch.compile(model)

    # for each puzzle batch try solving it several times with interpreter feedback and record
    # version with batch size 1, make bigger batch size later
    for i, puzzle in enumerate(all_puzzles):
        program_str = puzzle['program_str'] if 'program_str' in puzzle else puzzle['sat'].replace(
            'def sat', 'def f')

        solved, num_attempts, solution_list = solve_iterative(
            program_str,
            model,
            tokenizer,
            max_attempts,
            iterative_prompt_template,
            few_shot_examples,
        )
        puzzle['solved'] = solved
        puzzle['num_attempts'] = num_attempts
        puzzle['solution_list'] = solution_list
        puzzles_to_save.append(puzzle)

        if (i + 1) % 10 == 0:
            # append puzzles
            with jsonlines.open(puzzle_save_path_jsonl, mode='a') as writer:
                writer.write(puzzles_to_save)

            total_puzzles_to_save += puzzles_to_save
            puzzles_to_save = []

    # save file
    json.dump(total_puzzles_to_save, open(puzzle_save_path_json))


# test
if __name__ == "__main__":
    iterative_solving_loop('puzzles_dev.json', "facebook/opt-1.3b")

