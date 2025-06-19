<!-- markdownlint-disable -->

<a href="../trainer_pytorch/trainer_pytorch.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `trainer_pytorch.py`






---

## <kbd>class</kbd> `TrainerPytorch`




<a href="../trainer_pytorch/trainer_pytorch.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(
    model,
    optimizer,
    loss_fn,
    scheduler=None,
    device='cpu',
    project_name='pytorch_training',
    run_name=None,
    log_filename=None,
    use_wandb=True,
    use_accelerate=False,
    save_dir='models'
)
```

Initializes the TrainerPytorch class. 



**Args:**
 
 - <b>`model`</b> (torch.nn.Module):  The PyTorch model to train. 
 - <b>`optimizer`</b> (torch.optim.Optimizer):  The optimizer for training. 
 - <b>`loss_fn`</b> (callable):  The loss function to use. 
 - <b>`scheduler`</b> (torch.optim.lr_scheduler._LRScheduler, optional):  Learning rate scheduler. 
 - <b>`device`</b> (str, optional):  Device to use for training ('cuda' or 'cpu'). 
 - <b>`project_name`</b> (str, optional):  Weights & Biases project name. 
 - <b>`run_name`</b> (str, optional):  Name of the run in Weights & Biases. 
 - <b>`log_filename`</b> (str, optional):  Filename for logging training progress. 
 - <b>`use_wandb`</b> (bool, optional):  Whether to use Weights & Biases for logging. 
 - <b>`use_accelerate`</b> (bool, optional):  Whether to use Hugging Face Accelerate for distributed training. 
 - <b>`save_dir`</b> (str, optional):  Directory to save the trained model. 




---

<a href="../trainer_pytorch/trainer_pytorch.py#L297"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `close`

```python
close()
```

Cleans up resources after training. 

---

<a href="../trainer_pytorch/trainer_pytorch.py#L154"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `eval_epoch`

```python
eval_epoch(eval_loader)
```

Evaluates the model for one epoch. 

**Args:**
 
 - <b>`eval_loader`</b> (torch.utils.data.DataLoader):  DataLoader for the evaluation data. 

**Returns:**
 
 - <b>`float`</b>:  The average loss for the evaluation. 

---

<a href="../trainer_pytorch/trainer_pytorch.py#L117"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `eval_one_step`

```python
eval_one_step(inputs, targets)
```

Performs one evaluation step on the model. 

**Args:**
 
 - <b>`inputs`</b> (torch.Tensor):  Input data for the model. 
 - <b>`targets`</b> (torch.Tensor):  Target data for the model. 

**Returns:**
 
 - <b>`float`</b>:  The loss value for the evaluation step. 

---

<a href="../trainer_pytorch/trainer_pytorch.py#L280"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `load_model`

```python
load_model(load_path)
```

Loads the model, optimizer, and scheduler state from a file. 

**Args:**
 
 - <b>`load_path`</b> (str):  Path to the model state file. 

---

<a href="../trainer_pytorch/trainer_pytorch.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `move_to_device`

```python
move_to_device(batch)
```

Moves the input batch to the specified device. 

**Args:**
 
 - <b>`batch`</b> (torch.Tensor or tuple or list):  The input batch to move to the device.    

**Returns:**
 
 - <b>`torch.Tensor or tuple or list`</b>:  The batch moved to the specified device. 

---

<a href="../trainer_pytorch/trainer_pytorch.py#L219"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `predict`

```python
predict(inputs)
```

Makes predictions using the trained model. 

**Args:**
 
 - <b>`inputs`</b> (torch.Tensor):  Input data for making predictions. 

**Returns:**
 
 - <b>`torch.Tensor`</b>:  Model predictions. 

---

<a href="../trainer_pytorch/trainer_pytorch.py#L232"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `predict_samples`

```python
predict_samples(data_loader)
```

Makes predictions on a dataset using the trained model. 

**Args:**
 
 - <b>`data_loader`</b> (torch.utils.data.DataLoader):  DataLoader for the dataset to predict. 

**Returns:**
 
 - <b>`np.ndarray`</b>:  Model predictions for the dataset. 

---

<a href="../trainer_pytorch/trainer_pytorch.py#L249"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `save_model`

```python
save_model(save_path)
```

Saves the model, optimizer, and scheduler state to a file. 

**Args:**
 
 - <b>`save_path`</b> (str):  Path to save the model state. 

---

<a href="../trainer_pytorch/trainer_pytorch.py#L175"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `train`

```python
train(train_loader, eval_loader=None, epochs=1, patience=None)
```

Trains the model for a specified number of epochs. 

**Args:**
 
 - <b>`train_loader`</b> (torch.utils.data.DataLoader):  DataLoader for the training data. 
 - <b>`eval_loader`</b> (torch.utils.data.DataLoader, optional):  DataLoader for the evaluation data. 
 - <b>`epochs`</b> (int, optional):  Number of epochs to train the model. 
 - <b>`patience`</b> (int, optional):  Number of epochs to wait for improvement before early stopping. 

---

<a href="../trainer_pytorch/trainer_pytorch.py#L134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `train_epoch`

```python
train_epoch(train_loader)
```

Trains the model for one epoch. 

**Args:**
 
 - <b>`train_loader`</b> (torch.utils.data.DataLoader):  DataLoader for the training data. 

**Returns:**
 
 - <b>`float`</b>:  The average loss for the epoch. 

---

<a href="../trainer_pytorch/trainer_pytorch.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `train_one_step`

```python
train_one_step(inputs, targets)
```

Performs one training step on the model. 

**Args:**
 
 - <b>`inputs`</b> (torch.Tensor):  Input data for the model. 
 - <b>`targets`</b> (torch.Tensor):  Target data for the model. 

**Returns:**
 
 - <b>`float`</b>:  The loss value for the training step. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
