import torch
import numpy as np
from typing import Tuple
from torchvision import transforms


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1

def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, n_tasks=None, mode='reservoir'):
        assert mode in ['ring', 'reservoir']
        self.buffer_size = buffer_size
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['examples', 'task_types', 'labels', 'logits']

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def init_tensors(self, examples, task_types,labels, logits):
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :param activations: tensor containing the activations of the network
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                if isinstance(attr, torch.Tensor):
                    setattr(self, attr_str, torch.zeros(
                        (self.buffer_size, *attr.shape[1:]),
                        dtype=attr.dtype,
                        device=attr.device
                    ))
                elif isinstance(attr, str):
                    setattr(self, attr_str, [""] * self.buffer_size)
                else:
                    raise TypeError(f"Unsupported type {type(attr)} for attribute {attr_str}")

        if labels is not None:
            self.labels = [0 for _ in range(self.buffer_size)]
        if logits is not None:
            self.logits = [0 for _ in range(self.buffer_size)]


    def add_data(self, examples, task_types, labels=None, logits=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_types: string containing the task types
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, task_types, labels, logits)

        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].cuda()

                self.task_types[index] = task_types
                if labels is not None:
                    if self.labels[index] is not None:
                        self.labels[index] = None
                    self.labels[index] = labels[i].cuda()
                if logits is not None:
                    if self.logits[index] is not None:
                        self.logits[index] = None
                    self.logits[index] = logits[i].cuda()  

    def get_data(self, size: int, transform: transforms=None, multiple_aug=False) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                  size=size, replace=False)
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).cuda(),)
        if multiple_aug:
            ret_tuple += (torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).cuda(),)

        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                if isinstance(attr, torch.Tensor):
                    ret_tuple += (attr[choice],)
                elif isinstance(attr, list):
                    choice_list = choice.tolist()
                    ret_tuple += ([attr[i] for i in choice_list],)
                else:
                    raise TypeError(f"Unsupported type {type(attr)} for attribute {attr_str}")

        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0
