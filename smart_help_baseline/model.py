from typing import Any
import numpy as np
import torch
import torch.nn as nn

ACTION_TYPES = 7
OBJECT_TYPES = 52


class AgentEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int = 128,
    ) -> None:
        super().__init__()

        output_dim = int(output_dim)

        _hidden_size = int(output_dim / 4)

        self.action_type_encoder = nn.Sequential(        
            nn.Embedding(ACTION_TYPES, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        self.action_obj1_encoder = nn.Sequential(        
            nn.Embedding(OBJECT_TYPES, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        self.action_obj2_encoder = nn.Sequential(        
            nn.Embedding(OBJECT_TYPES, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        self.action_status_encoder = nn.Sequential(
            nn.Linear(1, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(_hidden_size * 4, _hidden_size * 4),
            nn.ReLU(),
            nn.Linear(_hidden_size * 4, _hidden_size * 2)
        )

        self.pos_encoder = nn.Sequential(
            nn.Linear(3, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        self.rot_encoder = nn.Sequential(
            nn.Linear(3, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        self.object_type_encoder = nn.Sequential(
            nn.Embedding(7, int(_hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(int(_hidden_size / 2), int(_hidden_size / 2))
        )

        self.object_encoder = nn.Sequential(
            nn.Embedding(OBJECT_TYPES, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        self.held_object_encoder = nn.Sequential(
            nn.Linear(_hidden_size * 9, _hidden_size * 4),
            nn.ReLU(),
            nn.Linear(_hidden_size * 4, _hidden_size * 2)
        )
        
        self.agent_encoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(_hidden_size * 6, output_dim)
        )


    def forward(self, agent_pos, agent_rot, agent_action, action_status, agent_held):
        action_type, action_obj1, action_obj2 = agent_action.split([1, 1, 1], dim=-1)
        agent_held1_type, agent_held_1, agent_held2_type, agent_held_2 = agent_held.split([1, 4, 1, 4], dim=-1)

        if torch.cuda.is_available():
            action_type = action_type.to(torch.long).cuda()
            action_obj1 = action_obj1.to(torch.long).cuda()
            action_obj2 = action_obj2.to(torch.long).cuda()
            agent_held1_type = agent_held1_type.to(torch.long).cuda()
            agent_held_1 = agent_held_1.to(torch.long).cuda()
            agent_held2_type = agent_held2_type.to(torch.long).cuda()
            agent_held_2 = agent_held_2.to(torch.long).cuda()
            action_status = action_status.to(torch.float).cuda()
            agent_pos = agent_pos.to(torch.float).cuda()
            agent_rot = agent_rot.to(torch.float).cuda()
        else:
            action_type = action_type.to(torch.long)
            action_obj1 = action_obj1.to(torch.long)
            action_obj2 = action_obj2.to(torch.long)
            agent_held1_type = agent_held1_type.to(torch.long)
            agent_held_1 = agent_held_1.to(torch.long)
            agent_held2_type = agent_held2_type.to(torch.long)
            agent_held_2 = agent_held_2.to(torch.long)
            action_status = action_status.to(torch.float)
            agent_pos = agent_pos.to(torch.float)
            agent_rot = agent_rot.to(torch.float)

        action_type_embedding = self.action_type_encoder(action_type).squeeze(-2)
        action_obj1_embedding = self.action_obj1_encoder(action_obj1).squeeze(-2)
        action_obj2_embedding = self.action_obj2_encoder(action_obj2).squeeze(-2)
        action_status_embedding = self.action_status_encoder(action_status).squeeze(-2)
        _action_embedding = torch.cat([action_type_embedding, action_obj1_embedding, action_obj2_embedding, action_status_embedding], dim=-1)
        action_embedding = self.action_encoder(_action_embedding)
        pos_embedding = self.pos_encoder(agent_pos)
        rot_embedding = self.rot_encoder(agent_rot)

        agent_held1_type_embedding = self.object_type_encoder(agent_held1_type).squeeze(-2)
        agent_held2_type_embedding = self.object_type_encoder(agent_held2_type).squeeze(-2)
        agent_held1_embedding = self.object_encoder(agent_held_1)
        agent_held2_embedding = self.object_encoder(agent_held_2)
        agent_held1_embedding = agent_held1_embedding.view(agent_held1_embedding.shape[0], agent_held1_embedding.shape[1], -1)
        agent_held2_embedding = agent_held2_embedding.view(agent_held2_embedding.shape[0], agent_held1_embedding.shape[1], -1)
        _held_objs_embedding = torch.cat([agent_held1_type_embedding, agent_held1_embedding, agent_held2_type_embedding, agent_held2_embedding], dim=-1)
        held_objs_embedding = self.held_object_encoder(_held_objs_embedding)
        _agents_embedding = torch.cat(
            [
                action_embedding,
                held_objs_embedding, 
                pos_embedding, 
                rot_embedding, 
            ],
            dim=-1
        )
        agents_embedding = self.agent_encoder(_agents_embedding)

        return agents_embedding

class ObjectEncoder(nn.Module):
    def __init__(
        self, 
        output_dim: int = 128,
        transformer_n_head: int = 8,
        transformer_dropout: float = 0.2,
    ) -> None:
        super().__init__()

        output_dim = int(output_dim)
        _hidden_size = int(output_dim / 4)

        self.type_encoder = nn.Sequential(
            nn.Embedding(OBJECT_TYPES, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        self.obj_embedding = nn.Parameter(data=torch.randn(1, output_dim))

        self.height_encoder = nn.Sequential(
            nn.Linear(1, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        self.weight_encoder = nn.Sequential(
            nn.Linear(1, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        self.pos_encoder = nn.Sequential(
            nn.Linear(3, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _hidden_size)
        )

        self.state_encoder = nn.Sequential(
            nn.Linear(_hidden_size * 4, _hidden_size * 4),
            nn.ReLU(),
            nn.Linear(_hidden_size * 4, _hidden_size * 4)
        )

        self.transformer_encoder_layer = nn.modules.TransformerEncoderLayer(
            d_model=output_dim, 
            nhead=transformer_n_head, 
            dropout=transformer_dropout, 
            dim_feedforward=output_dim,
            batch_first=True
        )

        self.transformer_encoder = nn.modules.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=4
        )

    def forward(
        self, 
        objs_type_id, 
        objs_weight, 
        objs_pos, 
        objs_height,
    ):
        if torch.cuda.is_available():
            objs_type_id = objs_type_id.to(torch.long).cuda()
            objs_weight = objs_weight.to(torch.float).cuda()
            objs_pos = objs_pos.to(torch.float).cuda()
            objs_height = objs_height.to(torch.float).cuda()
        else:
            objs_type_id = objs_type_id.to(torch.long)
            objs_weight = objs_weight.to(torch.float)
            objs_pos = objs_pos.to(torch.float)
            objs_height = objs_height.to(torch.float)
            

        # print("for debug object encoder", objs_type_id.device, torch.cuda.is_available())

        objs_type_embedding = self.type_encoder(objs_type_id).squeeze(-2)
        objs_height_embedding = self.height_encoder(objs_height)
        objs_weight_embedding = self.weight_encoder(objs_weight)
        objs_pos_embedding = self.pos_encoder(objs_pos)

        _objs_state_embedding = torch.cat(   
            [
                objs_type_embedding,
                objs_height_embedding, 
                objs_weight_embedding, 
                objs_pos_embedding, 
            ],
            dim=-1
        )
        # batch_size x window_size x observe_len x output_dim
        objs_state_embedding = self.state_encoder(_objs_state_embedding)

        batch_shape = objs_state_embedding.shape[0 : 2] 

        bs_obj_embedding = self.obj_embedding.repeat(*batch_shape, 1, 1)
        # batch_size x observe_len x output_dim -> batch_size x (observe_len+1) x output_dim
        objs_state_embedding = torch.cat(
            [
                bs_obj_embedding, 
                objs_state_embedding
            ],
            dim=-2
        )
        if torch.cuda.is_available():
            objs_type_id = objs_type_id.squeeze(-1).cuda()
            src_key_padding_mask = (objs_type_id == 0).cuda()
            has_objs = torch.sum(objs_type_id, dim=-1, keepdim=True).bool().cuda()
            src_key_padding_mask = torch.eq(src_key_padding_mask, has_objs)
            obj_embedding_mask = torch.zeros((*src_key_padding_mask.shape[:2], 1), dtype=torch.bool).cuda()
        
        else:
            objs_type_id = objs_type_id.squeeze(-1)
            src_key_padding_mask = (objs_type_id == 0)
            has_objs = torch.sum(objs_type_id, dim=-1, keepdim=True).bool()
            src_key_padding_mask = torch.eq(src_key_padding_mask, has_objs)
            obj_embedding_mask = torch.zeros((*src_key_padding_mask.shape[:2], 1), dtype=torch.bool)
        
        src_key_padding_mask = torch.cat([obj_embedding_mask, src_key_padding_mask], dim=-1)
        # embedding_shape = (observe_len+1) x output_dim
        embedding_shape = objs_state_embedding.shape[2 : ]  
        
        # batch_size x window_size x (observe_len+1) x output_dim -> (batch_size x window_size) x (observe_len+1) x output_dim
        objs_state_embedding_reshaped = objs_state_embedding.view(-1, *embedding_shape)
        src_key_padding_mask_reshaped = src_key_padding_mask.view(-1, src_key_padding_mask.shape[-1])

      
        _obj_observation_embedding = self.transformer_encoder(
            objs_state_embedding_reshaped, src_key_padding_mask=src_key_padding_mask_reshaped
        ).view(*batch_shape, *embedding_shape)
        
        # batch_size x window_size x (observe_len+1) x output_dim -> batch_size x window_size x output_dim
        obj_observation_embedding = _obj_observation_embedding[..., 0, :]

        return obj_observation_embedding

class OpponentModeling(nn.Module):
    def __init__(
        self,
        transformer_nhead: int = 2,
        transformer_dropout: float = 0.1,
        input_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_layers = 4
        self.hidden_size = 256

        self.time_sumarize_model = nn.Sequential(
            nn.Linear(5 * 256, 1028), 
            nn.ReLU(),
            nn.Linear(1028, 1028),
            nn.ReLU(),
            nn.Linear(1028, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.transformer_encoder_layer = nn.modules.TransformerEncoderLayer(
            d_model=128, 
            nhead=transformer_nhead, 
            dropout=transformer_dropout, 
            dim_feedforward=128,
            batch_first=True
        )

        self.transformer_encoder_layer_goal = nn.modules.TransformerEncoderLayer(
            d_model=128, 
            nhead=transformer_nhead, 
            dropout=transformer_dropout, 
            dim_feedforward=256,
            batch_first=True
        )

        self.goal_encoder = nn.modules.TransformerEncoder(
            self.transformer_encoder_layer_goal,
            num_layers=4
        )

        self.constraint_encoder = nn.modules.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=4
        )

        self.tar_index_1_predict = nn.Sequential(
            nn.Linear(256, 256), 
            nn.ReLU(), 
            nn.Linear(256, 256), 
            nn.ReLU(), 
            nn.Linear(256, 128), 
        )

        # self.tar_index_2_predict = nn.Sequential(
        #     nn.Linear(384, 384), 
        #     nn.ReLU(), 
        #     nn.Linear(384, 256), 
        #     nn.ReLU(), 
        #     nn.Linear(256, 128), 
        # )
        
        self.goal_classifier = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, ACTION_TYPES), 
            nn.Softmax(dim=-1)
        )

        self.tar_index_1_classifier = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, OBJECT_TYPES),
            nn.Softmax(dim=-1)
        )

        # self.tar_index_2_classifier = nn.Sequential(
        #     nn.Linear(128, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, OBJECT_TYPES),
        #     nn.Softmax(dim=-1)
        # )

        self.constraint_MLP = nn.Sequential(
            nn.Linear(128, 128), 
            nn.ReLU(), 
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, 5),
            nn.Sigmoid(), 
        )

    def forward(self, obj_feature, agents_embedding):

        total_feature = torch.cat((obj_feature, agents_embedding), dim=-1)

        batch_size = total_feature.shape[0]
        window_size = total_feature.shape[1]
        total_feature = total_feature.view(batch_size, window_size * 256)
        # print(total_feature.shape)

        time_feature = self.time_sumarize_model(total_feature)
        goal_feature = self.goal_encoder(time_feature)
        constraint_feature = self.constraint_encoder(time_feature)
        
        tar_1_feature = self.tar_index_1_predict(torch.cat([time_feature, goal_feature], dim=-1))
        # tar_2_feature = self.tar_index_2_predict(torch.cat([time_feature, goal_feature, tar_1_feature], dim=-1))

        # print(goal_feature.shape, tar_1_feature.shape, tar_2_feature.shape, constraint_feature.shape)
        # return goal_feature, tar_1_feature, tar_2_feature, constraint_feature
        return goal_feature, tar_1_feature, constraint_feature

    def estimate_subtask_and_type(self, obj_feature, agents_embedding):
        goal_feature, tar_1_feature, constraint_feature = self.forward(obj_feature, agents_embedding)
        goal_predict = self.goal_classifier(goal_feature)

        tar_index_1_predict = self.tar_index_1_classifier(tar_1_feature)
        # tar_index_2_predict = self.tar_index_2_classifier(tar_2_feature)

        constraint_predict = self.constraint_MLP(constraint_feature)

        # print(goal_predict.shape, tar_index_1_predict.shape, tar_index_2_predict.shape, constraint_predict.shape)

        return goal_predict, tar_index_1_predict, constraint_predict
    
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.agent_encoder = AgentEncoder()
        self.object_encoder = ObjectEncoder()
        self.opponent_modeling = OpponentModeling()

    def forward(self, agent_pos, agent_rot, agent_action, agent_status, agent_held, obj_id, obj_weight, obj_pos, obj_height, ignore_classifiers = True):
        # print(agent_pos.shape, agent_rot.shape, agent_action.shape, agent_status.shape, agent_held.shape)
        agent_embedding = self.agent_encoder(agent_pos, agent_rot, agent_action, agent_status, agent_held)

        # print(obj_id.shape, obj_weight.shape, obj_pos.shape, obj_height.shape)
        obj_embedding = self.object_encoder(obj_id, obj_weight, obj_pos, obj_height)

        if ignore_classifiers:
            goal_feature, tar_1_feature, constraint_feature = self.opponent_modeling(obj_embedding, agent_embedding)
            return goal_feature, tar_1_feature, constraint_feature
        else:
            goal_predict, tar_index_1_predict, constraint_predict = self.opponent_modeling.estimate_subtask_and_type(obj_embedding, agent_embedding)
            return goal_predict, tar_index_1_predict, constraint_predict