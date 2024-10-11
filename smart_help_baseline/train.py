import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import json
import tqdm
import os

from dataset import CustomDataset
from model import CustomModel

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = config["batch_size"]
epochs = config["num_epochs"]
lr = config["lr"]
log_interval = config["log_interval"]
eval_interval = config["eval_interval"]
save_interval = config["save_interval"]
num_workers = config["num_workers"]
save_dir = config["save_dir"]
max_train_data = config["max_train_data"]
max_val_data = config["max_validation_data"]
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print(batch_size, epochs, lr, log_interval, eval_interval, save_interval, num_workers)

def criterion(output, goal, constraint):
    goal_type, tar1 = goal.split([1, 1], dim=-1)
    goal_type = goal_type.squeeze(-1)
    tar1 = tar1.squeeze(-1)
    # tar2 = tar2.squeeze(-1)
    goal_predict, tar_index_1_predict, constraint_predict = output
    batch_size = goal.shape[0]

    goal_predict_index = torch.argmax(goal_predict, dim=-1)
    goal_predict_success_num = torch.sum(goal_predict_index == goal_type)
    goal_acc = goal_predict_success_num / batch_size
    tar_index_1_predict_index = torch.argmax(tar_index_1_predict, dim=-1)
    tar_index_1_predict_success_num = torch.sum(tar_index_1_predict_index == tar1)
    tar_index_1_acc = tar_index_1_predict_success_num / batch_size
    # tar_index_2_predict_index = torch.argmax(tar_index_2_predict, dim=-1)
    # tar_index_2_predict_success_num = torch.sum(tar_index_2_predict_index == tar2)
    # tar_index_2_acc = tar_index_2_predict_success_num / batch_size
    
    constraint_loss = F.mse_loss(constraint_predict, constraint)
    goal_distribution = torch.zeros_like(goal_predict)
    goal_distribution[torch.arange(batch_size), goal_type] = 1
    goal_loss = F.cross_entropy(goal_predict, goal_distribution)
    tar1_distribution = torch.zeros_like(tar_index_1_predict)
    tar1_distribution[torch.arange(batch_size), tar1] = 1
    tar_index_1_loss = F.cross_entropy(tar_index_1_predict, tar1_distribution)
    # tar2_distribution = torch.zeros_like(tar_index_2_predict)
    # tar2_distribution[torch.arange(batch_size), tar2] = 1
    # tar_index_2_loss = F.cross_entropy(tar_index_2_predict, tar2_distribution) / 3
    # print(goal_predict, tar_index_1_predict, tar_index_2_predict, constraint_predict)
    # print(goal, tar1, tar2, constraint)
    # print(constraint_loss, goal_loss, tar_index_1_loss, tar_index_2_loss)
    total_loss = goal_loss + tar_index_1_loss + constraint_loss
    return total_loss, goal_loss, tar_index_1_loss, constraint_loss, goal_acc, tar_index_1_acc

def collate_fn(batch):
    constaint = [item["constraint"] for item in batch]
    goal = [item["goal"] for item in batch]
    agent_position = [[item["agent"][i]["position"] for i in range(len(item["agent"]))] for item in batch]
    agent_rotation = [[item["agent"][i]["forward"] for i in range(len(item["agent"]))] for item in batch]
    agent_action = [[item["agent"][i]["action"] for i in range(len(item["agent"]))] for item in batch]
    agent_status = [[[item["agent"][i]["status"]] for i in range(len(item["agent"]))] for item in batch]
    agent_held_objects = [[item["agent"][i]["held_objects"] for i in range(len(item["agent"]))] for item in batch]
    obj_id = [[[[item["objs"][i]["id"][j]] for j in range(len(item["objs"][i]["id"]))] for i in range(len(item["objs"]))] for item in batch]
    obj_weight = [[[[item["objs"][i]["weight"][j]] for j in range(len(item["objs"][i]["weight"]))] for i in range(len(item["objs"]))] for item in batch]
    obj_position = [[item["objs"][i]["position"] for i in range(len(item["objs"]))] for item in batch]
    obj_height = [[[[item["objs"][i]["height"][j]] for j in range(len(item["objs"][i]["height"]))] for i in range(len(item["objs"]))] for item in batch]
    return {
        "constraint": torch.tensor(constaint),
        "goal": torch.tensor(goal),
        "agent_position": torch.tensor(agent_position),
        "agent_rotation": torch.tensor(agent_rotation),
        "agent_action": torch.tensor(agent_action),
        "agent_status": torch.tensor(agent_status),
        "agent_held_objects": torch.tensor(agent_held_objects),
        "obj_id": torch.tensor(obj_id),
        "obj_weight": torch.tensor(obj_weight),
        "obj_position": torch.tensor(obj_position),
        "obj_height": torch.tensor(obj_height)
    }

train_dataset = CustomDataset(config["train_json"], max_data = max_train_data)
train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, collate_fn = collate_fn)
val_dataset = CustomDataset(config["val_json"], max_data = max_val_data)
val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, collate_fn = collate_fn)

model = nn.DataParallel(CustomModel()).to(device)
optimizer = optim.Adam(model.parameters(), lr = lr)
tensorboard_writer = SummaryWriter(config["tensorboard_dir"])

model.train()

for epoch in tqdm.tqdm(range(epochs)):
    total_train_loss = 0
    total_train_goal_loss = 0
    total_train_tar_index_1_loss = 0
    total_train_tar_index_2_loss = 0
    total_train_constraint_loss = 0
    total_train_goal_acc = 0
    total_train_tar_index_1_acc = 0
    total_train_tar_index_2_acc = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        agent_position = batch["agent_position"].to(device)
        agent_rotation = batch["agent_rotation"].to(device)
        agent_action = batch["agent_action"].to(device)
        agent_status = batch["agent_status"].to(device)
        agent_held_objects = batch["agent_held_objects"].to(device)
        obj_id = batch["obj_id"].to(device)
        obj_weight = batch["obj_weight"].to(device)
        obj_position = batch["obj_position"].to(device)
        obj_height = batch["obj_height"].to(device)
        goal = batch["goal"].to(device)
        constraint = batch["constraint"].to(device)
        print(agent_position.shape)
        print(agent_rotation.shape)
        print(agent_action.shape)
        print(agent_status.shape)
        print(agent_held_objects.shape)
        print(obj_id.shape)
        print(obj_weight.shape)
        print(obj_position.shape)
        print(obj_height.shape)
        
        output = model(agent_pos = agent_position, 
                       agent_rot = agent_rotation, 
                       agent_action = agent_action, 
                       agent_status = agent_status, 
                       agent_held = agent_held_objects,
                       obj_id = obj_id, 
                       obj_weight = obj_weight, 
                       obj_pos = obj_position, 
                       obj_height = obj_height, 
                       ignore_classifiers = False)
        total_loss, goal_loss, tar_index_1_loss, constraint_loss, goal_acc, tar_index_1_acc = criterion(output, goal, constraint)
        total_train_loss += total_loss.item()
        total_train_goal_loss += goal_loss.item()
        total_train_tar_index_1_loss += tar_index_1_loss.item()
        # total_train_tar_index_2_loss += tar_index_2_loss.item()
        total_train_constraint_loss += constraint_loss.item()
        total_train_goal_acc += goal_acc.item()
        total_train_tar_index_1_acc += tar_index_1_acc.item()
        # total_train_tar_index_2_acc += tar_index_2_acc.item()

        total_loss.backward()
        optimizer.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_goal_loss = total_train_goal_loss / len(train_dataloader)
    avg_tar_index_1_loss = total_train_tar_index_1_loss / len(train_dataloader)
    # avg_tar_index_2_loss = total_train_tar_index_2_loss / len(train_dataloader)
    avg_constraint_loss = total_train_constraint_loss / len(train_dataloader)
    avg_goal_acc = total_train_goal_acc / len(train_dataloader)
    avg_tar_index_1_acc = total_train_tar_index_1_acc / len(train_dataloader)
    # avg_tar_index_2_acc = total_train_tar_index_2_acc / len(train_dataloader)

    if (epoch + 1) % eval_interval == 0:
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            total_val_goal_loss = 0
            total_val_tar_index_1_loss = 0
            total_val_tar_index_2_loss = 0
            total_val_constraint_loss = 0
            total_val_goal_acc = 0
            total_val_tar_index_1_acc = 0
            total_val_tar_index_2_acc = 0
            for batch in val_dataloader:
                agent_position = batch["agent_position"].to(device)
                agent_rotation = batch["agent_rotation"].to(device)
                agent_action = batch["agent_action"].to(device)
                agent_status = batch["agent_status"].to(device)
                agent_held_objects = batch["agent_held_objects"].to(device)
                obj_id = batch["obj_id"].to(device)
                obj_weight = batch["obj_weight"].to(device)
                obj_position = batch["obj_position"].to(device)
                obj_height = batch["obj_height"].to(device)
                goal = batch["goal"].to(device)
                constraint = batch["constraint"].to(device)
                output = model(agent_pos = agent_position, 
                       agent_rot = agent_rotation, 
                       agent_action = agent_action, 
                       agent_status = agent_status, 
                       agent_held = agent_held_objects,
                       obj_id = obj_id, 
                       obj_weight = obj_weight, 
                       obj_pos = obj_position, 
                       obj_height = obj_height, 
                       ignore_classifiers = False)
                total_loss, goal_loss, tar_index_1_loss, constraint_loss, goal_acc, tar_index_1_acc = criterion(output, goal, constraint)
                total_val_loss += total_loss.item()
                total_val_goal_loss += goal_loss.item()
                total_val_tar_index_1_loss += tar_index_1_loss.item()
                # total_val_tar_index_2_loss += tar_index_2_loss.item()
                total_val_constraint_loss += constraint_loss.item()
                total_val_goal_acc += goal_acc.item()
                total_val_tar_index_1_acc += tar_index_1_acc.item()
                # total_val_tar_index_2_acc += tar_index_2_acc.item()
            
            avg_val_loss = total_val_loss / len(val_dataloader)
            avg_val_goal_loss = total_val_goal_loss / len(val_dataloader)
            avg_val_tar_index_1_loss = total_val_tar_index_1_loss / len(val_dataloader)
            # avg_val_tar_index_2_loss = total_val_tar_index_2_loss / len(val_dataloader)
            avg_val_constraint_loss = total_val_constraint_loss / len(val_dataloader)
            avg_val_goal_acc = total_val_goal_acc / len(val_dataloader)
            avg_val_tar_index_1_acc = total_val_tar_index_1_acc / len(val_dataloader)
            # avg_val_tar_index_2_acc = total_val_tar_index_2_acc / len(val_dataloader)

            tensorboard_writer.add_scalar('Avg Val Loss', avg_val_loss, epoch + 1)
            tensorboard_writer.add_scalar('Avg Val Goal Loss', avg_val_goal_loss, epoch + 1)
            tensorboard_writer.add_scalar('Avg Val Tar Index 1 Loss', avg_val_tar_index_1_loss, epoch + 1)
            # tensorboard_writer.add_scalar('Avg Val Tar Index 2 Loss', avg_val_tar_index_2_loss, epoch + 1)
            tensorboard_writer.add_scalar('Avg Val Constraint Loss', avg_val_constraint_loss, epoch + 1)
            tensorboard_writer.add_scalar('Avg Val Goal Acc', avg_val_goal_acc, epoch + 1)
            tensorboard_writer.add_scalar('Avg Val Tar Index 1 Acc', avg_val_tar_index_1_acc, epoch + 1)
            # tensorboard_writer.add_scalar('Avg Val Tar Index 2 Acc', avg_val_tar_index_2_acc, epoch + 1)

            # print(f"Epoch: {epoch + 1}, Avg Val Loss: {avg_loss}")

    if (epoch + 1) % log_interval == 0:
        tensorboard_writer.add_scalar('Avg Train Loss', avg_train_loss, epoch + 1)
        tensorboard_writer.add_scalar('Avg Train Goal Loss', avg_goal_loss, epoch + 1)
        tensorboard_writer.add_scalar('Avg Train Tar Index 1 Loss', avg_tar_index_1_loss, epoch + 1)
        # tensorboard_writer.add_scalar('Avg Train Tar Index 2 Loss', avg_tar_index_2_loss, epoch + 1)
        tensorboard_writer.add_scalar('Avg Train Constraint Loss', avg_constraint_loss, epoch + 1)
        tensorboard_writer.add_scalar('Avg Train Goal Acc', avg_goal_acc, epoch + 1)
        tensorboard_writer.add_scalar('Avg Train Tar Index 1 Acc', avg_tar_index_1_acc, epoch + 1)
        # tensorboard_writer.add_scalar('Avg Train Tar Index 2 Acc', avg_tar_index_2_acc, epoch + 1)
        # print(f"Epoch: {epoch + 1}, Avg Train Loss: {avg_train_loss}")

    if (epoch + 1) % save_interval == 0:
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_{epoch + 1}.pth"))
        print(f"Saved model at epoch {epoch + 1}")


tensorboard_writer.close()