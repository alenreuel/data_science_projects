from sklearn.metrics import accuracy_score

import torch
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

class ModelTraining:
    def __init__(self, model, data):
        
        # logs
        self.training_logs = {}
        self.training_logs["epoch"] = []
        self.training_logs["training_loss"] = []
        self.training_logs["training_acc"] = []
        self.training_logs["val_loss"] = []       
        self.training_logs["val_acc"] = []

        # model
        self.model = model.to(device)

        self.data = data

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr = 0.001)

    def get_mask(self, mask_name):
        return self.data[mask_name].reshape(-1,1).repeat(1,self.data["X"].shape[1])
    

    def train_model(self):
        

        torch.manual_seed(47)
            
        self.model.train()
        # Sets the gradients of all optimized tensors to zero
        self.optimizer.zero_grad()
        
        predictions = self.model((self.data["X"]*self.get_mask("train_mask")).to(device), self.data["edge_index"].to(device))
        
        # Compute loss (here CrossEntropyLoss)
        loss = F.cross_entropy(predictions[self.data["train_mask"]].float(), (self.data["y"][self.data["train_mask"]]).to(device))
        # BackProp + Gradient Descent
        (loss).backward()
        self.optimizer.step()
        
        # metrics
        self.training_logs["training_loss"].append(loss.item())
        train_accuracy = accuracy_score(self.data["y"][self.data["train_mask"]].cpu().numpy(), torch.argmax(predictions[self.data["train_mask"]],dim=1).cpu().numpy() )
        self.training_logs["training_acc"].append(train_accuracy)
        
        
            
    def val_model(self):
        self.model.eval()
        with torch.inference_mode():
            predictions = self.model((self.data["X"]*self.get_mask("val_mask")).to(device), self.data["edge_index"].to(device))
        
        loss = F.cross_entropy(predictions[self.data["val_mask"]].float(), (self.data["y"][self.data["val_mask"]]).to(device))
        
        # metrics
        self.training_logs["val_loss"].append(loss.item())
        val_accuracy = accuracy_score(self.data["y"][self.data["val_mask"]].cpu().numpy(), torch.argmax(predictions[self.data["val_mask"]],dim=1).cpu().numpy() )
        self.training_logs ["val_acc"].append(val_accuracy)

    def training_loop(self, model_path, n_epochs=101):
        best_loss = float("inf")

        for epoch in range(1,n_epochs):

            self.training_logs["epoch"].append(epoch)
            
            self.train_model()
            self.val_model()
            if self.training_logs["val_loss"][-1]<best_loss:
                torch.save(self.model.state_dict(), model_path)
                best_loss = self.training_logs["val_loss"][-1]
            if epoch%5==0:
                prog_str = ""
                for i in self.training_logs.keys():
                    prog_str += f" {i}: {self.training_logs[i][-1]} |"
                print(prog_str)
    
        

            
            
            
           