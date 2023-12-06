# takes in a json file and model id, computes various quality metrics on the data
from quality_metrics.difficulty.difficulties import eval_puzzle_loop


def eval_quality(
    puzzle_path,
    model_id,
    batch_size=2,
    out_file_name=None,
):
    eval_puzzle_loop(
        puzzle_path=puzzle_path,
        model_id=model_id,
        batch_size=batch_size
    )


def eval_iterative_solving(
    puzzle_path,
    model_id,
    batch_size=2,
):
    pass
