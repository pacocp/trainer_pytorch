import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import shutil

from trainer_pytorch import TrainerPytorch

def test_trainer():
    # Initialize the TrainerPytorch with a simple model, optimizer, and loss function
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    trainer = TrainerPytorch(model, optimizer, loss_fn, scheduler=scheduler,
                             device=device, use_wandb=True, use_accelerate=True)

    # generate dummy data
    inputs = torch.randn(100, 10)
    targets = torch.randn(100, 1)
    dataset = TensorDataset(inputs, targets)
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=10, shuffle=False)

    # Check if the model parameters have been updated
    initial_params = [param.clone() for param in model.parameters()]
    trainer.train(train_loader, eval_loader=eval_loader, epochs=5)
    updated_params = [param.clone() for param in trainer.model.parameters()]
    for initial, updated in zip(initial_params, updated_params):
        assert not torch.equal(initial, updated), "Model parameters should be updated after training"
    
    # Evaluate the model
    outputs = trainer.predict_samples(inputs[:5])

    assert outputs.shape == (5,), "Output shape should match the target shape"

    # Delete the generated wandb directory and the models directory
    shutil.rmtree("wandb", ignore_errors=True)
    shutil.rmtree("models", ignore_errors=True)

def test_trainer_nowandb():
    # Initialize the TrainerPytorch with a simple model, optimizer, and loss function
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    trainer = TrainerPytorch(model, optimizer, loss_fn, scheduler=scheduler,
                             device=device, use_wandb=True, use_accelerate=True)

    # generate dummy data
    inputs = torch.randn(100, 10)
    targets = torch.randn(100, 1)
    dataset = TensorDataset(inputs, targets)
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=10, shuffle=False)

    # Check if the model parameters have been updated
    initial_params = [param.clone() for param in model.parameters()]
    trainer.train(train_loader, eval_loader=eval_loader, epochs=5)
    updated_params = [param.clone() for param in trainer.model.parameters()]
    for initial, updated in zip(initial_params, updated_params):
        assert not torch.equal(initial, updated), "Model parameters should be updated after training"
    
    # Evaluate the model
    outputs = trainer.predict_samples(inputs[:5])

    assert outputs.shape == (5,), "Output shape should match the target shape"

    # Delete the generated wandb directory and the models directory
    shutil.rmtree("wandb", ignore_errors=True)
    shutil.rmtree("models", ignore_errors=True)

def test_trainer_nowandbnoaccelerate():
    # Initialize the TrainerPytorch with a simple model, optimizer, and loss function
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    trainer = TrainerPytorch(model, optimizer, loss_fn, scheduler=scheduler,
                             device=device, use_wandb=True, use_accelerate=True)

    # generate dummy data
    inputs = torch.randn(100, 10)
    targets = torch.randn(100, 1)
    dataset = TensorDataset(inputs, targets)
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=10, shuffle=False)

    # Check if the model parameters have been updated
    initial_params = [param.clone() for param in model.parameters()]
    trainer.train(train_loader, eval_loader=eval_loader, epochs=5)
    updated_params = [param.clone() for param in trainer.model.parameters()]
    for initial, updated in zip(initial_params, updated_params):
        assert not torch.equal(initial, updated), "Model parameters should be updated after training"
    
    # Evaluate the model
    outputs = trainer.predict_samples(inputs[:5])

    assert outputs.shape == (5,), "Output shape should match the target shape"

    # Delete the generated wandb directory and the models directory
    shutil.rmtree("wandb", ignore_errors=True)
    shutil.rmtree("models", ignore_errors=True)