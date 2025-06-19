import torch
import wandb
import os
import numpy as np
from accelerate import Accelerator
from tqdm import tqdm

class TrainerPytorch:
    def __init__(self, model, optimizer, loss_fn, scheduler=None, device='cuda' if torch.cuda.is_available() else 'cpu',
                 project_name='pytorch_training', run_name=None, log_filename=None, 
                 use_wandb=True, use_accelerate=False, save_dir='models'):
        """
        Initializes the TrainerPytorch class.

        Args:
            model (torch.nn.Module): The PyTorch model to train.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            loss_fn (callable): The loss function to use.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
            device (str, optional): Device to use for training ('cuda' or 'cpu').
            project_name (str, optional): Weights & Biases project name.
            run_name (str, optional): Name of the run in Weights & Biases.
            log_filename (str, optional): Filename for logging training progress.
            use_wandb (bool, optional): Whether to use Weights & Biases for logging.
            use_accelerate (bool, optional): Whether to use Hugging Face Accelerate for distributed training.
            save_dir (str, optional): Directory to save the trained model.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        self.device = device
        self.use_wandb = use_wandb
        self.use_accelerate = use_accelerate
        self.accelerator = Accelerator() if self.use_accelerate else None
        if not self.use_accelerate:
            self.logger = open(os.path.join(self.save_dir, log_filename), 'w') if log_filename else open(os.path.join(self.save_dir, 'training_log.txt'), 'w')
        else:
            self.logger = None
        if use_wandb:
            try:
                wandb.init(project=project_name, name=run_name, config={
                    'model': model.__class__.__name__,
                    'optimizer': optimizer.__class__.__name__,
                    'loss_fn': loss_fn.__class__.__name__,
                    'device': device
                })
            except Exception as e:
                print(f"Error initializing Weights & Biases: {e}")
                self.use_wandb = False
                self.logger = open(os.path.join(self.save_dir, log_filename), 'w') if log_filename else open(os.path.join(self.save_dir, 'training_log.txt'), 'w')
                self.logger.write("Weights & Biases logging disabled due to error.\n")

        if use_accelerate:
            try:
                self.model, self.optimizer, self.loss_fn = self.accelerator.prepare(
                    self.model, self.optimizer, self.loss_fn
                )
                self.device = self.accelerator.device
            except Exception as e:
                print(f"Error initializing Accelerate: {e}")
                self.use_accelerate = False
                self.model.to(self.device)
        else:
            self.model.to(self.device)

    def move_to_device(self, batch):
        """       
        Moves the input batch to the specified device.
        Args:
            batch (torch.Tensor or tuple or list): The input batch to move to the device.   
        Returns:
            torch.Tensor or tuple or list: The batch moved to the specified device.
        """
        if isinstance(batch, (list, tuple)):
            return [item.to(self.device) for item in batch]
        elif isinstance(batch, dict):
            return {k: v.to(self.device) for k, v in batch.items()}
        else:
            return batch.to(self.device)

    def train_one_step(self, inputs, targets):
        """
        Performs one training step on the model.
        Args:
            inputs (torch.Tensor): Input data for the model.
            targets (torch.Tensor): Target data for the model.
        Returns:
            float: The loss value for the training step.
        """
        self.model.train()
        inputs, targets = self.move_to_device((inputs, targets))
        if self.use_accelerate:
            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                self.accelerator.backward(loss)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
        else:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

        return loss.item()
    
    def eval_one_step(self, inputs, targets):
        """
        Performs one evaluation step on the model.
        Args:
            inputs (torch.Tensor): Input data for the model.
            targets (torch.Tensor): Target data for the model.
        Returns:
            float: The loss value for the evaluation step.
        """
        self.model.eval()
        inputs, targets = self.move_to_device((inputs, targets))
        with torch.inference_mode():
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
        
        return loss.item()
    
    def train_epoch(self, train_loader):
        """
        Trains the model for one epoch.
        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        Returns:
            float: The average loss for the epoch.
        """
        total_loss = 0.0
        for batch in tqdm(train_loader, desc="Training", leave=False):
            inputs, targets = batch
            loss = self.train_one_step(inputs, targets)
            total_loss += loss
        
        avg_loss = total_loss / len(train_loader)
        if self.use_wandb:
            wandb.log({'train_loss': avg_loss})
        
        return avg_loss
    
    def eval_epoch(self, eval_loader):
        """
        Evaluates the model for one epoch.
        Args:
            eval_loader (torch.utils.data.DataLoader): DataLoader for the evaluation data.
        Returns:
            float: The average loss for the evaluation.
        """
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
                inputs, targets = batch
                loss = self.eval_one_step(inputs, targets)
                total_loss += loss
        
        avg_loss = total_loss / len(eval_loader)
        if self.use_wandb:
            wandb.log({'eval_loss': avg_loss})

        return avg_loss
    
    def train(self, train_loader, eval_loader=None, epochs=1, patience=None):
        """
        Trains the model for a specified number of epochs.
        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
            eval_loader (torch.utils.data.DataLoader, optional): DataLoader for the evaluation data.
            epochs (int, optional): Number of epochs to train the model.
            patience (int, optional): Number of epochs to wait for improvement before early stopping.
        """
        best_loss = float('inf')
        patience_counter = 0
        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            train_loss = self.train_epoch(train_loader)
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}')
            if self.logger:
                if self.use_accelerate:
                    if self.accelerator.is_main_process:
                        self.logger.write(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}\n')
                self.logger.write(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}\n')
            
            if eval_loader is not None:
                eval_loss = self.eval_epoch(eval_loader)
                print(f'Epoch {epoch + 1}/{epochs}, Eval Loss: {eval_loss:.4f}')
                if self.logger:
                    if self.use_accelerate:
                        if self.accelerator.is_main_process:
                            self.logger.write(f'Epoch {epoch + 1}/{epochs}, Eval Loss: {eval_loss:.4f}\n')

            if best_loss > eval_loss:
                best_loss = eval_loss
                patience_counter = 0
                if self.use_wandb:
                    wandb.log({'best_eval_loss': best_loss})
                print(f'Saving model with best eval loss: {best_loss:.4f}')
                if self.logger:
                    if self.use_accelerate:
                        if self.accelerator.is_main_process:
                            self.logger.write(f'Saving model with best eval loss: {best_loss:.4f}\n')
                self.save_model(f'best_model.pt')
            else:
                patience_counter += 1
                if patience is not None and patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch + 1} due to no improvement.')
                    if self.logger:
                        if self.use_accelerate:
                            if self.accelerator.is_main_process:
                                self.logger.write(f'Early stopping at epoch {epoch + 1} due to no improvement.\n')
                    break

        self.save_model('final_model.pt')

        self.load_model(os.path.join(self.save_dir, 'best_model.pt'))
    
    def predict(self, inputs):
        """
        Makes predictions using the trained model.
        Args:
            inputs (torch.Tensor): Input data for making predictions.
        Returns:
            torch.Tensor: Model predictions.
        """
        inputs = self.move_to_device(inputs)
        with torch.inference_mode():
            outputs = self.model(inputs)
        return outputs
    
    def predict_samples(self, data_loader):
        """
        Makes predictions on a dataset using the trained model.
        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for the dataset to predict.
        Returns:
            np.ndarray: Model predictions for the dataset.
        """
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                outputs = self.model(batch)
                predictions.append(outputs.cpu().numpy())
            
        return np.concatenate(predictions, axis=0) if predictions else np.array([])
    
    def save_model(self, save_path):
        """
        Saves the model, optimizer, and scheduler state to a file.
        Args:
            save_path (str): Path to save the model state.
        """
        if not save_path.endswith('.pt'):
            save_path += '.pt'
        save_path = os.path.join(self.save_dir, save_path)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        if self.use_accelerate:
            self.accelerator.wait_for_everyone()
            # save all the states: model, optimizer, scheduler
            if self.scheduler is not None:
                self.accelerator.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict()
                }, save_path)
            else:
                # save only the model state dict and optimizer
                self.accelerator.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }, save_path)
        else:
            # save all the states: model, optimizer, scheduler
            if self.scheduler is not None:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict()
                }, save_path)
            else:
                # save only the model state dict and optimizer
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }, save_path)
        if self.use_wandb:
            wandb.save(save_path)
    
    def load_model(self, load_path):
        """
        Loads the model, optimizer, and scheduler state from a file.
        Args:
            load_path (str): Path to the model state file.
        """
        if os.path.exists(load_path):
            state_dict = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(state_dict['model_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            if 'scheduler_state_dict' in state_dict and self.scheduler is not None:
                self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            if self.use_wandb:
                wandb.restore(load_path)
        else:
            raise FileNotFoundError(f"Model file {load_path} does not exist.")
    
    def close(self):
        """
        Cleans up resources after training.
        """
        if self.use_wandb:
            wandb.finish()
        if self.use_accelerate:
            self.accelerator.end_training()
        torch.cuda.empty_cache()
        self.logger.close() if self.logger else None
        print("Training complete and resources cleaned up.")
    
    def __del__(self):
        """
        Destructor to ensure resources are cleaned up.
        """
        self.close()
    