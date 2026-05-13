import logging
from typing import Dict, Optional, Union, List

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

from .data_set import Data_set
from .utils import extract_answer, math_equal, rouge_score

logger = logging.getLogger(__name__)


class AIME(Data_set):

    def __init__(self, tokenizer, year_filter: Optional[Union[int, List[int]]] = None, **kwargs):
        """
        Initialize AIME dataset.
        
        Args:
            tokenizer: The tokenizer to use
            year_filter: Optional year(s) to filter the dataset. 
                        Can be a single year (e.g., 2024) or a list of years (e.g., [2023, 2024, 2025])
            **kwargs: Additional arguments passed to parent class
        """
        self.year_filter = year_filter
        print(f'year_filter: {self.year_filter}')
        super().__init__(tokenizer, **kwargs)

    def load_raw_data(self) -> pd.DataFrame:
        """
        Download the dataset from huggingface datasets.
        """
        # Load the main AIME dataset (1983-2024, but missing 2024 round I)
        df1 = load_dataset("qq8933/AIME_1983_2024", split="train").to_pandas()
        
        # Load the complete 2024 AIME dataset
        df_2024_complete = load_dataset("Maxwell-Jia/AIME_2024", split="train").to_pandas()
        
        # Load the 2025 AIME dataset
        df2 = load_dataset("yentinglin/aime_2025", split="train").to_pandas()
        
        # Process the complete 2024 dataset
        df_2024_processed = self._process_2024_dataset(df_2024_complete)
        
        # Remove 2024 entries from df1 to avoid duplicates
        df1_without_2024 = df1[df1['Year'] != 2024].copy()
        
        # Rename columns in df2 to match df1's structure
        df2 = df2.rename(columns={
            'problem': 'Question',
            'answer': 'Answer',
            'year': 'Year'
        })
        
        # Add missing columns to df2 to match df1's structure
        df2['ID'] = df2['Year'].astype(str) + '-' + (df2.index + 1).astype(str)
        df2['Problem Number'] = df2.index + 1
        df2['Part'] = ''
        
        # Select only the columns that exist in df1
        common_columns = ['ID', 'Year', 'Problem Number', 'Question', 'Answer', 'Part']
        df2 = df2[common_columns]
        
        # Concatenate all datasets
        df = pd.concat([df1_without_2024, df_2024_processed, df2], ignore_index=True)
        
        # Apply year filter if specified
        if self.year_filter is not None:
            print(f'year filter is not None: {self.year_filter}')
            if isinstance(self.year_filter, int):
                # Single year filter
                df = df[df['Year'] == self.year_filter].reset_index(drop=True)
                logger.info(f"Filtered dataset to year {self.year_filter}. Remaining rows: {len(df)}")
            elif isinstance(self.year_filter, list):
                # Multiple years filter
                df = df[df['Year'].isin(self.year_filter)].reset_index(drop=True)
                logger.info(f"Filtered dataset to years {self.year_filter}. Remaining rows: {len(df)}")
        
        # Sort by Year (descending) and ID (ascending)
        df = df.sort_values(['Year', 'ID'], ascending=[False, True]).reset_index(drop=True)
        
        return df

    def _process_2024_dataset(self, df_2024: pd.DataFrame) -> pd.DataFrame:
        """
        Process the complete 2024 dataset to match the expected structure.
        
        Args:
            df_2024: DataFrame from Maxwell-Jia/AIME_2024
            
        Returns:
            Processed DataFrame with consistent structure
        """
        processed_rows = []
        
        for _, row in df_2024.iterrows():
            # Parse the ID format: '2024-II-4' -> year=2024, round=II, problem=4
            id_parts = row['ID'].split('-')
            year = int(id_parts[0])
            round_num = id_parts[1]  # 'I' or 'II'
            problem_num = int(id_parts[2])
            
            processed_row = {
                'ID': row['ID'],
                'Year': year,
                'Problem Number': problem_num,
                'Question': row['Problem'],
                'Answer': row['Answer'],
                'Part': round_num  # Store round information in Part field
            }
            processed_rows.append(processed_row)
        
        return pd.DataFrame(processed_rows)

    def create_groundtruth_field(self, row: Dict) -> Dict:
        row["groundtruth"] = row[
            "Answer"
        ]  # No need to extract the answer because it is already in good format
        return row

    def create_prompt_field(self, row: Dict) -> Dict:
        # TODO(hjh): Create more complex prompts
        row["prompt"] = (
            row["Question"] + "\nMake sure the final answer is standalone and in latex format."
        )
        return row

    def _calc_accuracy(self, row: Dict, approach: str) -> Dict:

        model_output: str = extract_answer(row[f"output_{approach}"], "aime")
        groundtruth: str = str(row["groundtruth"])

        row[f"accuracy_{approach}"] = math_equal(model_output, groundtruth)
        row[f"final_output_{approach}"] = model_output
        # row[f"accuracy_{approach}"] = rouge_score(model_output, groundtruth)
        return row


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("peiyi9979/mistral-7b-sft")
    
    # Example usage with different year filter options:
    # dataset = AIME(tokenizer=tokenizer, year_filter=2024)         # Complete 2024 (both rounds)
    # dataset = AIME(tokenizer=tokenizer, year_filter=[2023, 2024]) # Multiple years
    # dataset = AIME(tokenizer=tokenizer, year_filter=[2024, 2025]) # Recent years only
    dataset = AIME(tokenizer=tokenizer)                             # All years

    import pdb

    pdb.set_trace()