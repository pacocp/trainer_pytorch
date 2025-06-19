# Trainer-Pytorch

Implementation of a boilerplate pytorch trainer, with [wandb](https://wandb.ai/) and Huggingface [accelerate](https://huggingface.co/docs/accelerate/index) integration. 

## Installation

```bash
pip install trainer-pytorch
```
## Usage

If you want to use accelerate and wandb:

```bash

# initialize wandb
wandb login

# initialize accelerate
accelerate launch
```

Then, you can initialize the trainer with your model, optimizer, loss function, and scheduler:

```python
# Initialize the TrainerPytorch with a simple model, optimizer, and loss function
model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
trainer = TrainerPytorch(model, optimizer, loss_fn, scheduler=scheduler, device=device, 
                         use_wandb=True, use_accelerate=True, save_dir='my_experiment')

# generate dummy data
inputs = torch.randn(100, 10)
targets = torch.randn(100, 1)
dataset = TensorDataset(inputs, targets)
train_loader = DataLoader(dataset, batch_size=10, shuffle=True)
eval_loader = DataLoader(dataset, batch_size=10, shuffle=False)

# Train the model
trainer.train(train_loader, eval_loader=eval_loader, epochs=5, patience=2)

# Evaluate the model
outputs = trainer.predict_samples(inputs[:5])
```
