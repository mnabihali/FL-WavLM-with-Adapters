import torch
import torch.nn as nn
import torchaudio
import numpy as np
import pandas as pd
import os
import random
from tqdm.notebook import tqdm
from transformers import Wav2Vec2Processor
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union


class LibriSpeechDataset(object):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    def __init__(self, csv_file, lthresh=None):
        self.df = pd.read_csv(csv_file)
        self.lthresh = lthresh

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.df.loc[idx]
        array, sampling_rate = torchaudio.load(data['wav'])

        if self.lthresh:
            array = array.numpy().flatten()[:self.lthresh]
        else:
            array = array.numpy().flatten()
        array = self.processor(array, sampling_rate=sampling_rate).input_values[0]

        text = data['wrd']
        file = data['wav']

        with self.processor.as_target_processor():
            labels = self.processor(text).input_ids
        sample = {'input_values': array,
                  'labels': labels,
                  'files': file}
        return sample


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    preprocess: Optional[bool] = False

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        if self.preprocess:
            input_features = []
            label_features = []
            # chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
            for data in features:
                #     text = re.sub(chars_to_ignore_regex, '', data[2]).lower() + " "
                inputs = self.processor(data[0].numpy().flatten(), sampling_rate=data[1]).input_values[0]
                with self.processor.as_target_processor():
                    labels = self.processor(data[2]).input_ids
                input_features.append({'input_values': inputs})
                label_features.append({'input_ids': labels})
        else:
            # split inputs and labels since they have to be of different lenghts and need
            # different padding methods
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


def load_dataloaders(csv_files_directory):
    data_loaders = []
    # Get list of CSV files in the directory
    csv_files = [f for f in os.listdir(csv_files_directory) if f.endswith('.csv')]
    for file_name in csv_files:
        csv_file_path = os.path.join(csv_files_directory, file_name)
        dataset = LibriSpeechDataset(csv_file_path)  # Assuming LibriData function returns a Dataset object
        processor = dataset.processor
        collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collator, shuffle=True,
                                                  num_workers=0, pin_memory=True)
        data_loaders.append(data_loader)
    return data_loaders, processor
