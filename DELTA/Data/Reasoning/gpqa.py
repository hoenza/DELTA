import logging
import random
from typing import Dict

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

from .data_set import Data_set
from .utils import extract_answer, rouge_score

logger = logging.getLogger(__name__)


class Gpqa(Data_set):

    def load_raw_data(self) -> pd.DataFrame:
        """
        Download the dataset from huggingface datasets.
        """
        return load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train").to_pandas()

    def create_groundtruth_field(self, row: Dict) -> Dict:
        row["groundtruth"] = row["Pre-Revision Correct Answer"]
        return row

    def create_prompt_field(self, row: Dict) -> Dict:
        question = row["Pre-Revision Question"]
        correct_answer = row["Pre-Revision Correct Answer"]
        incorrect_1 = row["Pre-Revision Incorrect Answer 1"]
        incorrect_2 = row["Pre-Revision Incorrect Answer 2"]
        incorrect_3 = row["Pre-Revision Incorrect Answer 3"]
        
        # Create list of all answers and shuffle to randomize order
        all_answers = [correct_answer, incorrect_1, incorrect_2, incorrect_3]
        random.shuffle(all_answers)
        
        # Find the position of correct answer after shuffling to update groundtruth
        correct_position = all_answers.index(correct_answer)
        correct_letter = chr(ord('A') + correct_position)
        
        # Create the multiple choice format
        choices = []
        for i, answer in enumerate(all_answers):
            choice_letter = chr(ord('A') + i)
            choices.append(f"{choice_letter}. {answer}")
        
        choices_text = "\n".join(choices)
        
        row["prompt"] = (
            f"{question}\n\n\n{choices_text}\n\n"
            "Please reason step-by-step and put your choice letter without any other text with \\boxed{{}} in the end."
        )
        
        # Update groundtruth to the new letter position after shuffling
        row["groundtruth"] = correct_letter
        
        return row

    def _calc_accuracy(self, row: Dict, approach: str) -> Dict:
        model_output: str = extract_answer(row[f"output_{approach}"], "multiple_choice")
        groundtruth: str = row["groundtruth"]
        
        # For multiple choice, we need to check if the model output matches the correct answer
        # This assumes extract_answer can handle extracting choice letters from boxed format
        row[f"accuracy_{approach}"] = (model_output.strip().upper() == groundtruth.strip().upper())
        row[f"final_output_{approach}"] = model_output
        return row


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("peiyi9979/mistral-7b-sft")
    dataset = Gpqa(tokenizer=tokenizer)

    import pdb
    pdb.set_trace()