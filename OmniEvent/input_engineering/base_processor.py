import os
import json
import torch
import logging

from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union
from collections import defaultdict

logger = logging.getLogger(__name__)


class EDInputExample(object):
    """A single training/test example for event detection.

    A single training/test example for event detection, representing the basic information of an event trigger,
    including its example id, the source text it is within, its start and end position, and the label of the event.

    Attributes:
        example_id (`Union[int, str]`):
            A string or an integer for the unique id of the example.
        text (`str`):
            A string representing the source text the event trigger is within.
        trigger_left (`int`, `optional`, defaults to `None`):
            An integer indicating the left position of the event trigger.
        trigger_right (`int`, `optional`, defaults to `None`):
            An integer indicating the right position of the event trigger.
        labels (`str`, `optional`, defaults to `None`):
            A string indicating the event type of the trigger.
    """

    def __init__(self,
                 example_id,
                 text,
                 trigger_left=None,
                 trigger_right=None,
                 labels=None,
                 **kwargs):
        """Constructs a EDInputExample.

        Args:
            example_id: Unique id for the example.
            text: List of str. The untokenized text.
            trigger_left: Left position of trigger.
            trigger_right: Light position of tigger.
            labels: Event type of the trigger
        """
        self.example_id = example_id
        self.text = text
        self.trigger_left = trigger_left
        self.trigger_right = trigger_right
        self.labels = labels
        self.kwargs = kwargs


class EDInputFeatures(object):
    """Input features of an instance for event detection.

    Input features of an instance for event detection, representing the basic features of an event trigger, including
    its example id, the indices of tokens in the vocabulary, attention masks, segment token indices, start and end
    position, and the label of the event.

    Attributes:
        example_id (`Union[int, str]`):
            A string or an integer for the unique id of the example.
        input_ids (`List[int]`):
            A list of integers representing the indices of input sequence tokens in the vocabulary.
        attention_mask (`List[int]`):
            A list of integers (in 0/1) for masks to avoid attention on padding tokens.
        token_type_ids (`List[int]`, `optional`, defaults to `None`):
            A list of integers indicating the first and second portions of the inputs.
        trigger_left (`int`, `optional`, defaults to `None`):
            An integer indicating the left position of the event trigger.
        trigger_right (`int`, `optional`, defaults to `None`):
            An integer indicating the right position of the event trigger.
        labels (`str`, `optional`, defaults to `None`):
            A string indicating the label of the event.
    """

    def __init__(self,
                 example_id: Union[int, str],
                 input_ids: List[int],
                 attention_mask: List[int],
                 token_type_ids: Optional[List[int]] = None,
                 trigger_left: Optional[int] = None,
                 trigger_right: Optional[int] = None,
                 labels: Optional[List[int]] = None) -> None:
        """Constructs an `EDInputFeatures`."""
        self.example_id = example_id
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.trigger_left = trigger_left
        self.trigger_right = trigger_right
        self.labels = labels


class EAEInputExample(object):
    """A single training/test example for event argument extraction.

    A single training/test example for event argument extraction, representing the basic information of an event
    trigger, including its example id, the source text it is within, the predicted and actual event type, the input
    template for the Machine Reading Comprehension (MRC) paradigm, the start and end position of the event trigger and
    argument, and the label of the event.

    Attributes:
        example_id (`Union[int, str]`):
            A string or an integer for the unique id of the example.
        text (`str`):
            A string representing the source text the event trigger and argument is within.
        pred_type (`str`):
            A string indicating the event type predicted by the model.
        true_type (`str`):
            A string indicating the actual event type from the annotation.
        input_template:
            The input template for the MRC paradigm.
        trigger_left (`int`, `optional`, defaults to `None`):
            An integer indicating the left position of the event trigger.
        trigger_right (`int`, `optional`, defaults to `None`):
            An integer indicating the right position of the event trigger.
        argument_left (`int`, `optional`, defaults to `None`):
            An integer indicating the left position of the argument mention.
        argument_right (`int`, `optional`, defaults to `None`):
            An integer indicating the right position of the argument mention.
        argument_role (`str`, `optional`, defaults to `None`):
            A string indicating the argument role of the argument mention.
        labels (`str`, `optional`, defaults to `None`):
            A string indicating the label of the event.
    """

    def __init__(self,
                 example_id: Union[int, str],
                 text: Union[str, List[str]],
                 pred_type: str,
                 true_type: str,
                 trigger_id: Union[int, str] = None,
                 input_template: Optional[str] = None,
                 trigger_left: Optional[int] = None,
                 trigger_right: Optional[int] = None,
                 argument_left: Optional[int] = None,
                 argument_right: Optional[int] = None,
                 argument_role: Optional[str] = None,
                 labels: Optional[Union[str, List[str]]] = None,
                 **kwargs):
        """Constructs a `EAEInputExample`."""
        self.example_id = example_id
        self.text = text
        self.pred_type = pred_type
        self.true_type = true_type
        self.trigger_id = trigger_id
        self.input_template = input_template
        self.trigger_left = trigger_left
        self.trigger_right = trigger_right
        self.argument_left = argument_left
        self.argument_right = argument_right
        self.argument_role = argument_role
        self.labels = labels
        self.kwargs = kwargs


class EAEInputFeatures(object):
    """Input features of an instance for event argument extraction.

    Input features of an instance for event argument extraction, representing the basic features of an argument mention,
    including its example id, the indices of tokens in the vocabulary, the attention mask, segment token indices, the
    start and end position of the event trigger and argument mention, and the event type of the trigger.

    Attributes:
        example_id (`Union[int, str]`):
            A string or an integer for the unique id of the example.
        input_ids (`List[int]`):
            A list of integers representing the indices of input sequence tokens in the vocabulary.
        attention_mask (`List[int]`):
            A list of integers (in 0/1) for masks to avoid attention on padding tokens.
        token_type_ids (`List[int]`, `optional`, defaults to `None`):
            A list of integers indicating the first and second portions of the inputs.
        trigger_left (`int`, `optional`, defaults to `None`):
            An integer for the left position of the event trigger.
        trigger_right (`int`, `optional`, defaults to `None`):
            An integer for the right position of the event trigger.
        argument_left (`int`, `optional`, defaults to `None`):
            An integer for the left position of the argument mention.
        argument_right (`int`, `optional`, defaults to `None`):
            An integer for the right position of the argument mention.
        labels (`str`, `optional`, defaults to `None`):
            A string indicating the event type of the trigger.
    """

    def __init__(self,
                 example_id: Union[int, str],
                 input_ids: List[int],
                 attention_mask: List[int],
                 token_type_ids: Optional[List[int]] = None,
                 trigger_left: Optional[int] = None,
                 trigger_right: Optional[int] = None,
                 argument_left: Optional[int] = None,
                 argument_right: Optional[int] = None,
                 start_positions: Optional[int] = None,
                 end_positions: Optional[int] = None,
                 labels: Optional[Union[str, List[str]]] = None) -> None:
        """Constructs an `EAEInputFeatures`."""
        self.example_id = example_id
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.trigger_left = trigger_left
        self.trigger_right = trigger_right
        self.argument_left = argument_left
        self.argument_right = argument_right 
        self.start_positions = start_positions
        self.end_positions = end_positions
        self.labels = labels


class EEInputExample(object):
    """A single training/test example for event extraction. Only for sequence generation paradigm.
    """

    def __init__(self,
                 example_id,
                 text,
                 labels=None,
                 **kwargs):
        """Constructs a EEInputExample.

        Args:
            example_id: Unique id for the example.
            text: List of str. The untokenized text.
            trigger_left: Left position of trigger.
            trigger_right: Light position of tigger.
            labels: Event type of the trigger
        """
        self.example_id = example_id
        self.text = text
        self.labels = labels
        self.kwargs = kwargs


class EEInputFeatures(object):
    """Input features of an instance for event extraction.
    """

    def __init__(self,
                 example_id: Union[int, str],
                 input_ids: List[int],
                 attention_mask: List[int],
                 token_type_ids: Optional[List[int]] = None,
                 trigger_left: Optional[int] = None,
                 trigger_right: Optional[int] = None,
                 labels: Optional[List[int]] = None) -> None:
        """Constructs an `EDInputFeatures`."""
        self.example_id = example_id
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels = labels


class EDDataProcessor(Dataset):
    """Base class of data processor for event detection.

    The base class of data processor for event detection, which would be inherited to construct task-specific data
    processors.

    Attributes:
        config:
            The pre-defined configurations of the execution.
        tokenizer (`str`):
            The tokenizer method proposed for the tokenization process.
        examples (`List[EDInputExample]`):
            A list of `EDInputExample`s constructed based on the input dataset.
        input_features (`List[EDInputFeatures]`):
            A list of `EDInputFeatures`s corresponding to the `EDInputExample`s.
    """

    def __init__(self,
                 config,
                 tokenizer) -> None:
        """Constructs an `EDDataProcessor`."""
        self.config = config
        self.tokenizer = tokenizer
        self.examples = []
        self.input_features = []
        self.is_overflow = []

    def read_examples(self,
                      input_file: str):
        """Obtains a collection of `EDInputExample`s for the dataset."""
        raise NotImplementedError

    def convert_examples_to_features(self):
        """Converts the `EDInputExample`s into `EDInputFeatures`s."""
        raise NotImplementedError

    def _truncate(self,
                  outputs: dict,
                  max_seq_length: int):
        """Truncates the sequence that exceeds the maximum length."""
        is_truncation = False
        if len(outputs["input_ids"]) > max_seq_length:
            print("An instance exceeds the maximum length.")
            is_truncation = True
            for key in ["input_ids", "attention_mask", "token_type_ids", "offset_mapping"]:
                if key not in outputs:
                    continue
                outputs[key] = outputs[key][:max_seq_length]
        return outputs, is_truncation

    def get_ids(self) -> List[Union[int, str]]:
        """Returns the id of the examples."""
        ids = []
        for example in self.examples:
            ids.append(example.example_id)
        return ids

    def __len__(self) -> int:
        """Returns the length of the examples."""
        return len(self.input_features)

    def __getitem__(self,
                    index: int) -> Dict[str, torch.Tensor]:
        """Obtains the features of a given example index and converts them into a dictionary."""
        features = self.input_features[index]
        data_dict = dict(
            input_ids=torch.tensor(features.input_ids, dtype=torch.long),
            attention_mask=torch.tensor(features.attention_mask, dtype=torch.float32)
        )
        if features.token_type_ids is not None and self.config.return_token_type_ids:
            data_dict["token_type_ids"] = torch.tensor(features.token_type_ids, dtype=torch.long)
        if features.trigger_left is not None:
            data_dict["trigger_left"] = torch.tensor(features.trigger_left, dtype=torch.float32)
        if features.trigger_right is not None:
            data_dict["trigger_right"] = torch.tensor(features.trigger_right, dtype=torch.float32)
        if features.labels is not None:
            data_dict["labels"] = torch.tensor(features.labels, dtype=torch.long)
        return data_dict

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Collates the samples in batches."""
        output_batch = dict()
        for key in batch[0].keys():
            output_batch[key] = torch.stack([x[key] for x in batch], dim=0)
        if self.config.truncate_in_batch:
            input_length = int(output_batch["attention_mask"].sum(-1).max())
            for key in ["input_ids", "attention_mask", "token_type_ids"]:
                if key not in output_batch:
                    continue
                output_batch[key] = output_batch[key][:, :input_length]
            if "labels" in output_batch and len(output_batch["labels"].shape) == 2:
                if self.config.truncate_seq2seq_output:
                    output_length = int((output_batch["labels"] != -100).sum(-1).max())
                    output_batch["labels"] = output_batch["labels"][:, :output_length]
                else:
                    output_batch["labels"] = output_batch["labels"][:, :input_length]
        return output_batch


class EAEDataProcessor(Dataset):
    """Base class of data processor for event argument extraction.

    The base class of data processor for event argument extraction, which would be inherited to construct task-specific
    data processors.

    Attributes:
        config:
            The pre-defined configurations of the execution.
        tokenizer:
            The tokenizer method proposed for the tokenization process.
        is_training (`bool`):
            A boolean variable indicating the state is training or not.
        examples (`List[EDInputExample]`):
            A list of `EDInputExample`s constructed based on the input dataset.
        input_features (`List[EAEInputFeatures]`):
            A list of `EAEInputFeatures`s corresponding to the `EAEInputExample`s.
        data_for_evaluation (`dict`):
            A dictionary representing the evaluation data.
        event_preds (`list`):
            A list of event prediction data if the file exists.
    """

    def __init__(self,
                 config,
                 tokenizer,
                 pred_file: str,
                 is_training: bool) -> None:
        """Constructs a EAEDataProcessor."""
        self.config = config
        self.tokenizer = tokenizer
        self.is_training = is_training
        if hasattr(config, "role2id"):
            self.config.role2id["X"] = -100
        self.examples = []
        self.input_features = []
        self.is_overflow = []
        # data for trainer evaluation 
        self.data_for_evaluation = {}
        # event prediction file path 
        if pred_file is not None:
            if not os.path.exists(pred_file):
                logger.warning("%s doesn't exist.We use golden triggers" % pred_file)
                self.event_preds = None
            else:
                self.event_preds = json.load(open(pred_file))
        else:
            logger.warning("Event predictions is none! We use golden triggers.")
            self.event_preds = None

    def read_examples(self,
                      input_file: str):
        """Obtains a collection of `EAEInputExample`s for the dataset."""
        raise NotImplementedError

    def convert_examples_to_features(self):
        """Converts the `EAEInputExample`s into `EAEInputFeatures`s."""
        raise NotImplementedError

    def get_data_for_evaluation(self) -> Dict[str, Union[int, List[str]]]:
        """Obtains the data for evaluation."""
        self.data_for_evaluation["pred_types"] = self.get_pred_types()
        self.data_for_evaluation["true_types"] = self.get_true_types()
        self.data_for_evaluation["ids"] = self.get_ids()
        self.data_for_evaluation["trigger_ids"] = self.get_trigger_ids()
        self.data_for_evaluation["examples"] = self.examples
        if self.examples[0].argument_role is not None:
            self.data_for_evaluation["roles"] = self.get_roles()
        return self.data_for_evaluation

    def get_pred_types(self) -> List[str]:
        """Obtains the event type predicted by the model."""
        pred_types = []
        for example in self.examples:
            pred_types.append(example.pred_type)
        return pred_types

    def get_true_types(self) -> List[str]:
        """Obtains the actual event type from the annotation."""
        true_types = []
        for example in self.examples:
            true_types.append(example.true_type)
        return true_types

    def get_roles(self) -> List[str]:
        """Obtains the role of each argument mention."""
        roles = []
        for example in self.examples:
            roles.append(example.argument_role)
        return roles

    def _truncate(self,
                  outputs: Dict[str, List[int]],
                  max_seq_length: int):
        """Truncates the sequence that exceeds the maximum length."""
        is_truncation = False
        if len(outputs["input_ids"]) > max_seq_length:
            print("An instance exceeds the maximum length.")
            is_truncation = True
            for key in ["input_ids", "attention_mask", "token_type_ids", "offset_mapping"]:
                if key not in outputs:
                    continue
                outputs[key] = outputs[key][:max_seq_length]
        return outputs, is_truncation

    def get_ids(self) -> List[Union[int, str]]:
        """Returns the id of the examples."""
        ids = []
        for example in self.examples:
            ids.append(example.example_id)
        return ids
    
    def get_trigger_ids(self):
        trigger_ids = []
        for example in self.examples:
            trigger_ids.append(example.trigger_id)
        return trigger_ids

    def get_single_pred(self, trigger_idx, input_file, true_type):
        if self.is_training or "train" in input_file or self.config.golden_trigger or self.event_preds is None:
            pred_type = true_type
        else:
            pred_type = self.event_preds[trigger_idx]
        return pred_type

    def __len__(self) -> int:
        """Returns the length of the examples."""
        return len(self.input_features)

    def __getitem__(self,
                    index: int) -> Dict[str, torch.Tensor]:
        """Returns the features of a given example index in a dictionary."""
        features = self.input_features[index]
        data_dict = dict(
            input_ids=torch.tensor(features.input_ids, dtype=torch.long),
            attention_mask=torch.tensor(features.attention_mask, dtype=torch.float32)
        )
        if features.token_type_ids is not None and self.config.return_token_type_ids:
            data_dict["token_type_ids"] = torch.tensor(features.token_type_ids, dtype=torch.long)
        if features.trigger_left is not None:
            data_dict["trigger_left"] = torch.tensor(features.trigger_left, dtype=torch.long)
        if features.trigger_right is not None:
            data_dict["trigger_right"] = torch.tensor(features.trigger_right, dtype=torch.long)
        if features.argument_left is not None:
            data_dict["argument_left"] = torch.tensor(features.argument_left, dtype=torch.long)
        if features.argument_right is not None:
            data_dict["argument_right"] = torch.tensor(features.argument_right, dtype=torch.long)
        if features.start_positions is not None:
            data_dict["start_positions"] = torch.tensor(features.start_positions, dtype=torch.long)
        if features.end_positions is not None:
            data_dict["end_positions"] = torch.tensor(features.end_positions, dtype=torch.long)
        if features.labels is not None:
            data_dict["labels"] = torch.tensor(features.labels, dtype=torch.long)
        return data_dict

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Collates the samples in batches."""
        output_batch = dict()
        for key in batch[0].keys():
            output_batch[key] = torch.stack([x[key] for x in batch], dim=0)
        if self.config.truncate_in_batch:
            input_length = int(output_batch["attention_mask"].sum(-1).max())
            for key in ["input_ids", "attention_mask", "token_type_ids"]:
                if key not in output_batch:
                    continue
                output_batch[key] = output_batch[key][:, :input_length]
            if "labels" in output_batch and len(output_batch["labels"].shape) == 2:
                if self.config.truncate_seq2seq_output:
                    output_length = int((output_batch["labels"] != -100).sum(-1).max())
                    output_batch["labels"] = output_batch["labels"][:, :output_length]
                else:
                    output_batch["labels"] = output_batch["labels"][:, :input_length]
        return output_batch


class EEDataProcessor(Dataset):
    def __init__(self,
                 config,
                 tokenizer,
                 is_train_set) -> None:
        """Constructs an `EDDataProcessor`."""
        self.config = config
        self.tokenizer = tokenizer
        self.is_train_set = is_train_set
        self.examples = defaultdict(list)
        self.input_features = defaultdict(list)
        self.dataset_counter = list()

    def read_examples(self,
                      input_file: str):
        """Obtains a collection of `EEInputExample`s for the dataset."""
        raise NotImplementedError

    def convert_examples_to_features(self):
        """Converts the `EEInputExample`s into `EDInputFeatures`s."""
        raise NotImplementedError

    def _truncate(self,
                  outputs: dict,
                  max_seq_length: int):
        """Truncates the sequence that exceeds the maximum length."""
        is_truncation = False
        if len(outputs["input_ids"]) > max_seq_length:
            print("An instance exceeds the maximum length.")
            is_truncation = True
            for key in ["input_ids", "attention_mask", "token_type_ids", "offset_mapping"]:
                if key not in outputs:
                    continue
                outputs[key] = outputs[key][:max_seq_length]
        return outputs, is_truncation

    def __len__(self) -> int:
        """Returns the length of the examples."""
        return 160000

    def example_proportional_mixing(self) -> EEInputFeatures:
        import numpy as np; import random
        total_counter = sum(self.dataset_counter)
        p = np.array(self.dataset_counter) / total_counter
        task_idx = np.random.choice(np.arange(len(self.dataset_counter)), p=p)
        task_features = self.input_features[task_idx]
        return random.choice(task_features)

    def __getitem__(self,
                    index: int) -> Dict[str, torch.Tensor]:
        """Obtains the features of a given example index and converts them into a dictionary."""
        # features = self.input_features[index]
        features = self.example_proportional_mixing()
        data_dict = dict(
            input_ids=torch.tensor(features.input_ids, dtype=torch.long),
            attention_mask=torch.tensor(features.attention_mask, dtype=torch.float32)
        )
        if features.token_type_ids is not None and self.config.return_token_type_ids:
            data_dict["token_type_ids"] = torch.tensor(features.token_type_ids, dtype=torch.long)
        if features.labels is not None:
            data_dict["labels"] = torch.tensor(features.labels, dtype=torch.long)
        return data_dict

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Collates the samples in batches."""
        output_batch = dict()
        for key in batch[0].keys():
            output_batch[key] = torch.stack([x[key] for x in batch], dim=0)
        if self.config.truncate_in_batch:
            input_length = int(output_batch["attention_mask"].sum(-1).max())
            if self.config.truncate_seq2seq_output: # Seq2Seq
                for key in ["input_ids", "attention_mask", "token_type_ids"]:
                    if key not in output_batch:
                        continue
                    output_batch[key] = output_batch[key][:, :input_length]
                output_length = int((output_batch["labels"] != -100).sum(-1).max())
                output_batch["labels"] = output_batch["labels"][:, :output_length]
            else: # CausalLM left padding
                seq_start = self.config.max_seq_length - input_length
                for key in ["input_ids", "attention_mask", "labels", "token_type_ids"]:
                    if key not in output_batch:
                        continue
                    output_batch[key] = output_batch[key][:, seq_start:]
        return output_batch