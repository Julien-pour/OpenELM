from typing import List

from quality_metrics.gpt_judgements.prompts import five_fold_ranking_prompt


def build_prompt(puzzle_strs: List[str]):
    puzzle_text = ''
    for i, p in enumerate(puzzle_strs): 
        puzzle_text += 'Puzzle {i}:\n' + p + '\n'
    return five_fold_ranking_prompt.format(examples=puzzle_text)


def get_five_fold_results(puzzle_strs):
    pass