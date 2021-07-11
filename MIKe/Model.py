from MIKe.PositionalEmbedding import PositionalEmbedding
from MIKe.TransformerSeqEncoderDecoder import *
from MIKe.Utils import neginf
from MIKe.Utils import _generate_square_subsequent_mask
from MIKe.Utils import build_map
from MIKe.Utils import to_sentence
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn.modules.normalization import LayerNorm


class UserInitiativeSelector(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.project_c = nn.Linear(args.hidden_size, args.hidden_size)
        self.project_k = nn.Linear(args.hidden_size, args.hidden_size)
        self.project_v = nn.Linear(args.hidden_size, args.hidden_size)
        #self.att = BilinearAttention(768, 768, 768)

    def forward(self, context_pooling, knowledge_pooling, knowledge_piece_mask, episode_mask, mode="train"):
        # context_pooling [batch * max_episode_length, hidden_size])
        # knowledge_pooling [batch * max_episode_length, max_knowledge_pool, hidden_size]
        # knowledge_piece_mask [batch, max_episode_length, max_knowledge_pool]
        # episode_mask [batch, max_episode_length]

        knowledge_piece_mask = knowledge_piece_mask.reshape(-1, knowledge_piece_mask.size()[-1])  # [batch * max_episode_length, max_knowledge_pool]
        #episode_mask = episode_mask.reshape(-1) # [batch * max_episode_length]
        #attention_mask = torch.bmm(episode_mask.unsqueeze(-1).unsqueeze(-1).float(), knowledge_piece_mask.unsqueeze(1).float()).bool() # [batch * max_episode_length, 1, max_knowledge_pool]

        context_pooling_pro = self.project_c(context_pooling)  # [batch * max_episode_length, hidden_size]
        knowledge_pooling_pro_k = self.project_k(knowledge_pooling) # [batch * max_episode_length, max_knowledge_pool, hidden_size]


        # [batch * max_episode_length, max_knowledge_pool, hidden_size] bmm [batch * max_episode_length, hidden_size, 1] -> [batch * max_episode_length, max_knowledge_pool]
        knowledge_score = torch.bmm(knowledge_pooling_pro_k, context_pooling_pro.unsqueeze(-1)).squeeze(-1)
        knowledge_score.masked_fill_(~knowledge_piece_mask, neginf(knowledge_score.dtype)) # [batch * max_episode_length, max_knowledge_pool]
        knowledge_dist = F.softmax(knowledge_score, 1)  # [batch * max_episode_length, max_knowledge_pool]

        knowledge_pooling_pro_v = self.project_v(knowledge_pooling)  # [batch * max_episode_length, max_knowledge_pool, hidden_size]
        attention_vector = torch.bmm(knowledge_dist.unsqueeze(1), knowledge_pooling_pro_v).squeeze(1) # [batch * max_episode_length, hidden_size]

        # attention_vector [batch * max_episode_length, hidden_size]
        # knowledge_dist [batch * max_episode_length, max_knowledge_pool]
        return attention_vector, knowledge_dist, knowledge_score


class SystemInitiativeSelector(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args

        encoder_norm = LayerNorm(args.hidden_size)

        self.knowledge_dependency_transformer_layer = TransformerEncoderLayer(args.k_hidden_size, nhead=args.k_n_heads,dim_feedforward=args.k_ffn_size,dropout=0.1, activation='gelu')
        self.knowledge_dependency_transformer = TransformerEncoder(self.knowledge_dependency_transformer_layer, num_layers=args.k_n_layers, norm=encoder_norm)

        self.turn_embedding_matrix = nn.Embedding(args.max_episode_length, args.embedding_size)  # [max_episode_length, embedding_size]
        self.start_embedding_matrix = nn.Embedding(1, args.embedding_size)

        self.project_s = nn.Linear(args.hidden_size, args.hidden_size)
        self.project_k = nn.Linear(args.hidden_size, args.hidden_size)
        self.project_v = nn.Linear(args.hidden_size, args.hidden_size)

    def forward(self, knowledge_pooling_use, knowledge_pooling, knowledge_piece_mask, episode_mask, mode="train"):
        # knowledge_pooling_use [batch, max_episode_length, hidden_size]
        # knowledge_pooling [batch * max_episode_length, max_knowledge_pool, hidden_size]
        # knowledge_piece_mask  [batch, max_episode_length, max_knowledge_pool]
        # episode_mask [batch, max_episode_length]

        batch_size, max_episode_length = episode_mask.size()

        knowledge_piece_mask = knowledge_piece_mask.reshape(-1, knowledge_piece_mask.size()[-1])  # [batch * max_episode_length, max_knowledge_pool]


        start_embedding = self.start_embedding_matrix(torch.zeros([batch_size, 1], device=episode_mask.device).long())  # [batch, 1, hidden_size]
        input_embedding = torch.cat([start_embedding, knowledge_pooling_use[:, :(max_episode_length-1)]], dim=1) # [batch, max_episode_length, hidden_size]
        turn_embedding = self.turn_embedding_matrix(torch.arange(max_episode_length, device=episode_mask.device)).unsqueeze(0)  # [1, max_episode_length, hidden_size]
        input_embedding = input_embedding + turn_embedding  # [batch, max_episode_length, hidden_size]


        # [batch, max_episode_length, hidden_size]
        state = self.knowledge_dependency_transformer(input_embedding.transpose(0, 1), mask=_generate_square_subsequent_mask(max_episode_length)).transpose(0, 1)

        state_pro = self.project_s(state).reshape(batch_size * max_episode_length, -1)  # [batch * max_episode_length, hidden_size]
        knowledge_pooling_pro_k = self.project_k(knowledge_pooling)  # [batch * max_episode_length, max_knowledge_pool, hidden_size]


        # [batch * max_episode_length, max_knowledge_pool, hidden_size] bmm [batch * max_episode_length, hidden_size, 1] -> [batch * max_episode_length, max_knowledge_pool]
        knowledge_score = torch.bmm(knowledge_pooling_pro_k, state_pro.unsqueeze(-1)).squeeze(-1)
        knowledge_score.masked_fill_(~knowledge_piece_mask, neginf(knowledge_score.dtype)) # [batch * max_episode_length, max_knowledge_pool]
        knowledge_dist = F.softmax(knowledge_score, 1)  # [batch * max_episode_length, max_knowledge_pool]

        knowledge_pooling_pro_v = self.project_v(knowledge_pooling)  # [batch * max_episode_length, max_knowledge_pool, hidden_size]
        attention_vector = torch.bmm(knowledge_dist.unsqueeze(1), knowledge_pooling_pro_v).squeeze(1) # [batch * max_episode_length, hidden_size]

        # attention_vector [batch * max_episode_length, hidden_size]
        # knowledge_dist [batch * max_episode_length, max_knowledge_pool]
        return attention_vector, knowledge_dist, knowledge_score


class TeacherInitiativeDiscriminator(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.project_s = nn.Linear(args.hidden_size, 1)

        encoder_norm = LayerNorm(args.hidden_size)

        self.knowledge_dependency_transformer_layer = TransformerEncoderLayer(args.k_hidden_size, nhead=args.k_n_heads,dim_feedforward=args.k_ffn_size,dropout=0.1, activation='gelu')
        self.knowledge_dependency_transformer = TransformerEncoder(self.knowledge_dependency_transformer_layer, num_layers=args.k_n_layers, norm=encoder_norm)

        self.turn_embedding_matrix = nn.Embedding(args.max_episode_length+1, args.embedding_size)  # [max_episode_length, embedding_size]
        self.start_embedding_matrix = nn.Embedding(1, args.embedding_size)


    def forward(self, knowledge_pooling_use, episode_mask, mode="train"):
        # knowledge_pooling_use [batch, max_episode_length, hidden_size]
        # episode_mask [batch, max_episode_length]

        batch_size, max_episode_length = episode_mask.size()

        # normal_training
        start_embedding = self.start_embedding_matrix(torch.zeros([batch_size, 1], device=episode_mask.device).long())  # [batch, 1, hidden_size]
        input_embedding = torch.cat([start_embedding, knowledge_pooling_use], dim=1) # [batch, 1+max_episode_length, hidden_size]
        turn_embedding = self.turn_embedding_matrix(torch.arange(max_episode_length+1, device=episode_mask.device)).unsqueeze(0)  # [1, 1+max_episode_length, hidden_size]
        input_embedding = input_embedding + turn_embedding  # [batch, 1+max_episode_length, hidden_size]

        # input_embedding.transpose(0, 1) [1+max_episode_length, batch, hidden_size]
        # state [batch, 1+max_episode_length, hidden_size]
        state = self.knowledge_dependency_transformer(input_embedding.transpose(0, 1), mask=_generate_square_subsequent_mask(input_embedding.size(1))).transpose(0, 1)

        # [batch, 1+max_episode_length, 1]
        state_pro = self.project_s(state)
        state_pro = state_pro[:, 1:].reshape(-1, 1)  # [batch * max_episode_length, 1]
        user_prob = torch.sigmoid(state_pro)  # [batch * max_episode_length, 1]


        # self_supervised learning
        input_embedding = torch.cat(# [batch, 1+max_episode_length, hidden_size]
            [self.start_embedding_matrix(torch.zeros([batch_size, 1], device=episode_mask.device).long()), knowledge_pooling_use], dim=1)

        key_mask = torch.cat([torch.tensor([True] * batch_size, device=episode_mask.device).unsqueeze(1), episode_mask],dim=1)  # [batch, 1+max_episode_length]

        active_num = torch.sum(key_mask.long(), dim=1)  # [batch]

        deleted_input_embedding = []
        deleted_index = []
        deleted_key_mask = []
        for i in range(batch_size):

            delete_single_index = torch.randint(1, active_num[i]-1, (1,))[0]
            deleted_single_input = input_embedding[i][torch.arange(input_embedding[i].size(0)) != delete_single_index] # [max_episode_length, hidden_size]
            deleted_single_key_mask = key_mask[i][torch.arange(key_mask[i].size(0)) != delete_single_index]

            deleted_input_embedding.append(deleted_single_input)
            deleted_index.append(delete_single_index)
            deleted_key_mask.append(deleted_single_key_mask)

        deleted_input_embedding = torch.stack(deleted_input_embedding, dim=0).cuda() # [batch, max_episode_length, hidden_size]
        deleted_index = torch.stack(deleted_index, dim=0).cuda()  # [batch]
        deleted_key_mask = torch.stack(deleted_key_mask, dim=0).cuda()  # [batch, max_episode_length]

        turn_embedding = self.turn_embedding_matrix(torch.arange(max_episode_length, device=episode_mask.device)).unsqueeze(0)  # [1, max_episode_length, hidden_size]
        deleted_input_embedding = deleted_input_embedding + turn_embedding  # [batch, max_episode_length, hidden_size]

        # [batch, max_episode_length, hidden_size]
        state = self.knowledge_dependency_transformer(deleted_input_embedding.transpose(0, 1),
                                                       mask=_generate_square_subsequent_mask(deleted_input_embedding.size(1)),
                                                       src_key_padding_mask=~deleted_key_mask).transpose(0, 1)

        state_pro_ss = self.project_s(state)  # [batch, max_episode_length, 1]
        user_prob_ss = torch.sigmoid(state_pro_ss)  # [batch, max_episode_length, 1]


        # user_prob [batch * max_episode_length, 1]
        # state_pro [batch * max_episode_length, 1]
        # user_prob_ss [batch, max_episode_length, 1]
        # state_pro_ss [batch, max_episode_length, 1]
        # deleted_index [batch]
        # deleted_key_mask [batch, max_episode_length]
        return user_prob, state_pro, user_prob_ss, state_pro_ss, deleted_index, deleted_key_mask


class StudentInitiativeDiscriminator(nn.Module):
    def __init__(self, args=None):
        super().__init__()

        encoder_norm = LayerNorm(args.hidden_size)

        self.knowledge_dependency_transformer_layer = TransformerEncoderLayer(args.k_hidden_size, nhead=args.k_n_heads,dim_feedforward=args.k_ffn_size,dropout=0.1, activation='gelu')
        self.knowledge_dependency_transformer = TransformerEncoder(self.knowledge_dependency_transformer_layer, num_layers=args.k_n_layers, norm=encoder_norm)


        self.turn_embedding_matrix = nn.Embedding(args.max_episode_length, args.embedding_size)  # [max_episode_length, embedding_size]
        self.start_embedding_matrix = nn.Embedding(1, args.embedding_size)


        self.project_f = nn.Linear(5 * args.hidden_size, 1)

    def forward(self, context_pooling, knowledge_pooling_use, episode_mask, user_attention_vector, system_attention_vector, mode="train"):
        # context_pooling [batch * max_episode_length, hidden_size]
        # knowledge_pooling_use [batch, max_episode_length, hidden_size]
        # episode_mask [batch, max_episode_length]
        # user_attention_vector [batch * max_episode_length, hidden_size]
        # system_attention_vector [batch * max_episode_length, hidden_size]
        batch_size, max_episode_length = episode_mask.size()

        start_embedding = self.start_embedding_matrix(torch.zeros([batch_size, 1], device=episode_mask.device).long())  # [batch, 1, hidden_size]
        input_embedding = torch.cat([start_embedding, knowledge_pooling_use[:, :(max_episode_length - 1)]], dim=1)  # [batch, max_episode_length, hidden_size]
        turn_embedding = self.turn_embedding_matrix(torch.arange(max_episode_length, device=episode_mask.device)).unsqueeze(0)  # [1, max_episode_length, hidden_size]
        input_embedding = input_embedding + turn_embedding  # [batch, max_episode_length, hidden_size]


        # [batch, max_episode_length, hidden_size]
        state = self.knowledge_dependency_transformer(input_embedding.transpose(0, 1), mask=_generate_square_subsequent_mask(input_embedding.size(1))).transpose(0, 1)

        state = state.reshape(batch_size*max_episode_length, -1)  # [batch * max_episode_length, hidden_size]
        fusion = torch.cat([state, context_pooling, user_attention_vector, system_attention_vector, user_attention_vector-system_attention_vector], dim=1)  # [batch * max_episode_length, 4 * hidden_size]
        fusion_pro = self.project_f(fusion) # [batch * max_episode_length, 1]
        user_prob = torch.sigmoid(fusion_pro)  # [batch * max_episode_length, 1]

        #state_pro = self.project_s(state).reshape(batch_size*max_episode_length, -1) # [batch * max_episode_length, hidden_size]
        #context_pro = self.project_u(context_pooling) # [batch * max_episode_length, hidden_size]

        #fusion = torch.cat([state_pro, context_pro], dim=1) # [batch * max_episode_length, 2 * hidden_size]
        #fusion_pro = self.project_f(fusion) # [batch * max_episode_length, 1]
        #user_prob = torch.sigmoid(fusion_pro) # [batch * max_episode_length, 1]

        return user_prob, fusion_pro


class MIKe(nn.Module):
    def __init__(self, vocab2id, id2vocab, args):
        super().__init__()

        self.id2vocab = id2vocab
        self.vocab_size = len(id2vocab)
        self.args = args

        self.enc = TransformerSeqEncoder(args, vocab2id, id2vocab, None)  # num_layers, num_heads, src_vocab_size, hidden_size, emb_matrix=None
        self.dec = TransformerSeqDecoder(args, vocab2id, id2vocab, self.enc.enc.embeddings)  # num_memories, num_layers, nhead, tgt_vocab_size, hidden_size, emb_matrix=None

        self.user_initiative_selector = UserInitiativeSelector(args=args)
        self.system_initiative_selector = SystemInitiativeSelector(args=args)
        self.teacher_initiative_discriminator = TeacherInitiativeDiscriminator(args=args)
        self.student_initiative_discriminator = StudentInitiativeDiscriminator(args=args)

    def encoding_layer(self, data):
        context = data['context']  # [batch, max_episode_length, context_len]
        knowledge_token = data['knowledge_pool']   # [batch, max_episode_length, max_knowledge_pool, knowledge_sentence_len]

        context = context.reshape(-1, context.size()[-1])  # [batch * max_episode_length, context_len]
        knowledge_token = knowledge_token.reshape(-1, knowledge_token.size()[-2], knowledge_token.size()[-1]) # [batch * max_episode_length, max_knowledge_pool, knowledge_sentence_len]


        context_mask = context.ne(0).detach()  # [batch * max_episode_length, context_len]
        knowledge_token_mask = knowledge_token.ne(0).detach()  # [batch * max_episode_length, max_knowledge_pool, knowledge_sentence_len]

        context_encoded = self.enc(context.unsqueeze(1), context_mask.unsqueeze(1))
        knowledge_pool_encoded = self.enc(knowledge_token, knowledge_token_mask)

        return {'context_encoded': (context_encoded[0].squeeze(1), context_encoded[1].squeeze(1)),  # [batch * max_episode_length, context_len, hidden_size]
                "context_mask": context_mask,  # [batch * max_episode_length, hidden_size]
                'knowledge_pool_encoded': knowledge_pool_encoded, # [batch * max_episode_length, max_knowledge_pool, knowledge_sentence_len, hidden_size]
                "knowledge_token_mask": knowledge_token_mask}  # [batch * max_episode_length, max_knowledge_pool, knowledge_sentence_len]

    def mixed_initiative_knowledge_selection_layer(self, data, encoded_state):
        knowledge_piece_mask = data["knowledge_piece_mask"]  # [batch, max_episode_length, max_knowledge_pool]
        knowledge_label = data["knowledge_label"]  # [batch, max_episode_length]

        batch_size, max_episode_length = knowledge_label.size()

        # get history knowledge
        knowledge_label = knowledge_label.reshape(-1)  # [batch * max_episode_length]
        knowledge_pooling = encoded_state['knowledge_pool_encoded'][1]  # [batch * max_episode_length, max_knowledge_pool, hidden_size]
        batch_size_max_episode_length, max_knowledge_pool, hidden_size = knowledge_pooling.size()
        offsets = torch.arange(batch_size_max_episode_length, device=knowledge_label.device) * max_knowledge_pool + knowledge_label # [batch * max_episode_length]
        flatten_knowledge_pooling = knowledge_pooling.reshape(batch_size_max_episode_length * max_knowledge_pool, -1) # [batch * max_episode_length * max_knowledge_pool, hidden_size]
        knowledge_pooling_use = flatten_knowledge_pooling[offsets]  # [batch * max_episode_length, hidden_size]
        knowledge_pooling_use = knowledge_pooling_use.reshape(batch_size, max_episode_length, hidden_size) # [batch, max_episode_length, hidden_size]

        user_attention_vector, user_initiative_dist, user_initiative_score = self.user_initiative_selector(encoded_state['context_encoded'][1], encoded_state['knowledge_pool_encoded'][1], knowledge_piece_mask, data['episode_mask'], self.args.mode)
        system_attention_vector, system_initiative_dist, system_initiative_score = self.system_initiative_selector(knowledge_pooling_use, encoded_state['knowledge_pool_encoded'][1], knowledge_piece_mask, data['episode_mask'], self.args.mode)

        if self.args.mode == "inference":
            s_user_prob, s_state_pro = self.student_initiative_discriminator(encoded_state['context_encoded'][1], knowledge_pooling_use, data['episode_mask'], user_attention_vector, system_attention_vector, self.args.mode)


        elif self.args.mode == "train":
            # s_user_prob [batch * max_episode_length, 1]
            # s_state_pro [batch * max_episode_length, 1]

            # t_user_prob [batch * max_episode_length, 1]
            # t_state_pro [batch * max_episode_length, 1]

            # t_user_prob_ss [batch, max_episode_length, 1]
            # t_state_pro_ss [batch, max_episode_length, 1]

            # deleted_index [batch]
            t_user_prob, t_state_pro, t_user_prob_ss, t_state_pro_ss, deleted_index, deleted_key_mask = self.teacher_initiative_discriminator(knowledge_pooling_use, data['episode_mask'], self.args.mode)
            s_user_prob, s_state_pro = self.student_initiative_discriminator(encoded_state['context_encoded'][1], knowledge_pooling_use, data['episode_mask'], user_attention_vector, system_attention_vector, self.args.mode)


        final_dist = torch.mul(user_initiative_dist, s_user_prob.expand_as(user_initiative_dist)) + torch.mul(system_initiative_dist, 1 - s_user_prob.expand_as(system_initiative_dist))


        # select the knowledge
        if self.args.mode == "inference":
            _, knowledge_ids = final_dist.max(1)  # [batch * max_episode_length]
        elif self.args.mode == "train":
            knowledge_ids = knowledge_label  # [batch * max_episode_length]

        batch_size_max_episode_length, max_knowledge_pool, knowledge_sentence_len, hidden_size = encoded_state['knowledge_pool_encoded'][0].size()

        offsets = torch.arange(batch_size_max_episode_length, device=final_dist.device) * max_knowledge_pool + knowledge_ids  # [batch * max_episode_length]

        knowledge_selected = encoded_state['knowledge_pool_encoded'][0].reshape(-1, knowledge_sentence_len, hidden_size)[offsets]  # [batch * max_episode_length, knowledge_sentence_len, hidden_size]
        knowledge_selected_mask = encoded_state['knowledge_token_mask'].reshape(-1, knowledge_sentence_len)[offsets]  # [batch * max_episode_length, knowledge_sentence_len]
        knowledge_selected_index = data['knowledge_pool'].reshape(-1, knowledge_sentence_len)[offsets]  # [batch * max_episode_length, knowledge_sentence_len]

        memory1 = encoded_state['context_encoded'][0]  # [batch * max_episode_length, context_len, hidden_size]
        memory2 = knowledge_selected  # [batch * max_episode_length, knowledge_sentence_len, hidden_size]
        memory_mask1 = encoded_state['context_mask']  # [batch * max_episode_length, hidden_size]
        memory_mask2 = knowledge_selected_mask  # [batch * max_episode_length, knowledge_sentence_len]

        if self.args.mode == "inference":
            return {'memory1': memory1,
                    'memory2': memory2,
                    'memory_mask1': memory_mask1,
                    'memory_mask2': memory_mask2,
                    'knowledge_selected_index': knowledge_selected_index,
                    'final_dist': final_dist,
                    'user_initiative_dist': user_initiative_dist,
                    'system_initiative_dist': system_initiative_dist}

        elif self.args.mode == "train":
            return {'memory1': memory1,
                    'memory2': memory2,
                    'memory_mask1': memory_mask1,
                    'memory_mask2': memory_mask2,
                    'knowledge_selected_index': knowledge_selected_index,
                    'final_dist': final_dist,
                    'user_initiative_dist': user_initiative_dist,
                    'system_initiative_dist': system_initiative_dist,
                    "t_user_prob": t_user_prob,
                    "t_state_pro": t_state_pro,
                    "s_user_prob": s_user_prob,
                    "s_state_pro": s_state_pro,
                    "t_user_prob_ss": t_user_prob_ss,
                    "deleted_index": deleted_index,
                    "deleted_key_mask": deleted_key_mask
                    }

    def decoding_layer(self, data, memory):
        batch_size, max_episode_length, context_len = data["context"].size()
        source_map = build_map(torch.cat([data["context"].reshape(-1, context_len), memory["knowledge_selected_index"]], dim=1), max=self.vocab_size)
        dec_outputs, gen_outputs, extended_gen_outputs, output_indices = self.dec(
            [memory['memory1'], memory['memory2']], encode_masks=[memory['memory_mask1'], memory['memory_mask2']],
            groundtruth_index=data['response'].reshape(batch_size*max_episode_length, -1), source_map=source_map)

        # extended_gen_outputs [batch * max_episode_length, max_target_length, tgt_vocab_size]
        # output_indexes [batch * max_episode_length, max_target_length]
        return extended_gen_outputs, output_indices

    def to_sentence(self, data, batch_indices):
        return to_sentence(batch_indices, self.id2vocab)

    def do_train(self, data):
        encoded_state = self.encoding_layer(data)
        memory = self.mixed_initiative_knowledge_selection_layer(data, encoded_state)
        rg = self.decoding_layer(data, memory)

        _, final_ks_pred = memory['final_dist'].detach().max(1)  # [batch * max_episode_length]
        _, user_ks_pred = memory['user_initiative_dist'].detach().max(1)  # [batch * max_episode_length]
        _, system_ks_pred = memory['system_initiative_dist'].detach().max(1)  # [batch * max_episode_length]

        final_ks_acc = accuracy_score(data['knowledge_label'].reshape(-1).cpu(), final_ks_pred.cpu(), sample_weight=data['episode_mask'].reshape(-1).cpu())
        user_ks_acc = accuracy_score(data['knowledge_label'].reshape(-1).cpu(), user_ks_pred.cpu(), sample_weight=data['episode_mask'].reshape(-1).cpu())
        system_ks_acc = accuracy_score(data['knowledge_label'].reshape(-1).cpu(), system_ks_pred.cpu(), sample_weight=data['episode_mask'].reshape(-1).cpu())

        # Knowlegde selection loss
        masked_knowledge_label = data['knowledge_label'].reshape(-1).masked_fill(~data['episode_mask'].reshape(-1), -1) # [batch * max_episode_length]
        loss_ks = F.nll_loss((memory['final_dist'] + 1e-8).log(), masked_knowledge_label, ignore_index=-1)

        # MSE Loss
        loss_mse = F.mse_loss(memory['s_user_prob'].squeeze(-1), memory['t_user_prob'].squeeze(-1).detach(), reduction='none')  # [batch * max_episode_length]
        mse_mask = data['episode_mask'].reshape(-1).float()  # [batch * max_episode_length]
        loss_mse = torch.sum(loss_mse * mse_mask)/torch.sum(mse_mask)

        # Self-Supervised loss
        predict = memory['t_user_prob_ss'].squeeze(-1)  # [batch, max_episode_length]
        target = torch.zeros_like(predict, device=predict.device).scatter_(1, memory['deleted_index'].unsqueeze(-1), 1).detach() # [batch, max_episode_length]
        loss_ss = F.binary_cross_entropy(predict, target, reduction='none')
        ss_mask = memory['deleted_key_mask'].float()  # [batch, max_episode_length]
        ss_mask[:, 0] = 0. # 把start的位置mask掉
        loss_ss = torch.sum(loss_ss * ss_mask)/torch.sum(ss_mask)

        # generation loss
        loss_g = F.nll_loss((rg[0] + 1e-8).log().reshape(-1, rg[0].size(-1)), data['response'].reshape(-1), ignore_index=0)

        return loss_g, loss_ks, loss_mse, loss_ss, final_ks_acc, user_ks_acc, system_ks_acc, memory['s_user_prob'].mean()


    def do_inference(self, data):
        encoded_state = self.encoding_layer(data)
        memory = self.mixed_initiative_knowledge_selection_layer(data, encoded_state)
        rg = self.decoding_layer(data, memory)

        _, final_ks_pred = memory['final_dist'].max(1)  # [batch * max_episode_length]


        return rg[1], final_ks_pred

    def forward(self, data, method='train'):
        if method == 'train':
            return self.do_train(data)
        elif method == 'inference':
            return self.do_inference(data)

