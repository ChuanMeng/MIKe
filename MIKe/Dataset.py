from torch.utils.data import Dataset
from MIKe.Utils import *
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class Dataset(Dataset):

    def __init__(self, mode, episodes, query, passage, vocab2id, max_episode_length=None, max_knowledge_pool_when_train=None, max_knowledge_pool_when_inference=None, knowledge_sentence_len=None, context_len=None, max_dec_length=None, n=1E10):  # 1e10=1E10
        super(Dataset, self).__init__()

        self.mode = mode
        self.max_episode_length = max_episode_length

        self.context_len = context_len
        self.max_dec_length = max_dec_length

        if self.mode == "train":
            self.max_knowledge_pool = max_knowledge_pool_when_train
        elif self.mode == "inference":
            self.max_knowledge_pool = max_knowledge_pool_when_inference
        else:
            Exception("no ther mode")

        self.knowledge_sentence_len = knowledge_sentence_len

        self.episodes = episodes
        self.query = query
        self.passage = passage

        self.answer_file = episodes[0][0]['answer_file']

        self.vocab2id = vocab2id
        self.n = n

        self.episode_tensor = []
        self.load()

    def load(self):
        for id in range(len(self.episodes)):
            episode = self.episodes[id]
            episode_content = {"context": [], "response": [], "knowledge_pool": [], "knowledge_piece_mask": [], "knowledge_label" : []}
            for id_in_episode, example in enumerate(episode):
                if id_in_episode == self.max_episode_length:
                    break

                # process for context
                context = self.query[example['query_id']]
                context_ = [CLS_WORD] + context + [SEP_WORD]
                if len(context_) > self.context_len:
                    context_ = [CLS_WORD] + context_[-(self.context_len-1):]
                elif len(context_) < self.context_len:
                    context_ = context_ + [PAD_WORD] * (self.context_len - len(context_))

                assert len(context_) == self.context_len


                context_tensor = torch.tensor([self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in context_], requires_grad=False).long() # [context_len]
                episode_content["context"].append(context_tensor)

                # process for response
                response = example['response']
                response_ = response + [EOS_WORD]

                if len(response_) > self.max_dec_length:
                    response_ = response_[:self.max_dec_length]
                else:
                    response_ = response_ + [PAD_WORD] * (self.max_dec_length-len(response_))
                assert len(response_) == self.max_dec_length


                response_tensor = torch.tensor([self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in response_], requires_grad=False).long()
                episode_content["response"].append(response_tensor)  # [reponse_len]

                # process for knowledge pool
                if self.mode == "train" and len(example['knowledge_pool']) > self.max_knowledge_pool:
                #if len(example['knowledge_pool']) > self.max_knowledge_pool:
                    keepers = 1 + np.random.choice(len(example['knowledge_pool']) - 1, self.max_knowledge_pool, False)
                    # correct answer is always the first one
                    keepers[0] = 0
                    temp_sample_knowledge_pool = [example['knowledge_pool'][i] for i in keepers]
                else:
                    temp_sample_knowledge_pool = example['knowledge_pool'].copy()


                # text [[tokens],[tokens],...]
                knowledge_text_list = []
                for pid in temp_sample_knowledge_pool:
                    p = [CLS_WORD] + self.passage[pid]+ [SEP_WORD]
                    if len(p) > self.knowledge_sentence_len:
                        p = p[:self.knowledge_sentence_len-1]+[SEP_WORD]
                    elif len(p) < self.knowledge_sentence_len:
                        p = p + [PAD_WORD] * (self.knowledge_sentence_len - len(p))
                    assert len(p) == self.knowledge_sentence_len
                    knowledge_text_list.append(p)

                # process for knowledge mask
                knowledge_piece_mask = torch.zeros(self.max_knowledge_pool)
                knowledge_piece_mask[:len(knowledge_text_list)] = 1
                knowledge_piece_mask = knowledge_piece_mask == 1

                episode_content["knowledge_piece_mask"].append(knowledge_piece_mask)  # [max_knowledge_pool]

                while len(knowledge_text_list) < self.max_knowledge_pool:
                    knowledge_text_list.append([CLS_WORD] + [SEP_WORD]+ [PAD_WORD] * (self.knowledge_sentence_len-2))
                assert len(knowledge_text_list) == self.max_knowledge_pool


                knowledge_tensor = [torch.tensor([self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in p],
                                                 requires_grad=False).long() for p in knowledge_text_list]
                knowledge_tensor = torch.stack(knowledge_tensor)  # [max_knowledge_pool, knowledge_sentence_len]
                episode_content["knowledge_pool"].append(knowledge_tensor)

                # process for knowledge label
                assert temp_sample_knowledge_pool.index(example['knowledge_label'][0]) == 0
                #  单元素的list
                episode_content["knowledge_label"].append(torch.tensor([temp_sample_knowledge_pool.index(example['knowledge_label'][0])], requires_grad=False).long())

            # process episode
            assert len(episode_content["context"]) == len(episode_content["response"]) == len(
                episode_content["knowledge_pool"]) == len(episode_content["knowledge_piece_mask"]) == len(
                episode_content["knowledge_label"])

            episode_mask = torch.zeros(self.max_episode_length)  # [self.max_episode_length]
            episode_mask[:len(episode_content["context"])] = 1
            episode_mask = episode_mask == 1

            while len(episode_content["context"]) < self.max_episode_length:
                episode_content["context"].append(torch.tensor([self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in ([CLS_WORD]+[PAD_WORD] * (self.context_len-1))], requires_grad=False).long()) # [context_len]
                episode_content["response"].append(torch.tensor([self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in ([PAD_WORD] * self.max_dec_length)], requires_grad=False).long()) # [max_dec_length]
                episode_content["knowledge_piece_mask"].append(torch.zeros(self.max_knowledge_pool) == 1)
                episode_content["knowledge_label"].append(torch.tensor([0], requires_grad=False).long())

                knowledge_text_list = []
                while len(knowledge_text_list) < self.max_knowledge_pool:
                    knowledge_text_list.append([CLS_WORD] + [SEP_WORD]+ [PAD_WORD] * (self.knowledge_sentence_len-2))
                knowledge_tensor = [torch.tensor([self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in p],
                                                 requires_grad=False).long() for p in knowledge_text_list]
                knowledge_tensor = torch.stack(knowledge_tensor)  # [max_knowledge_pool, knowledge_sentence_len]
                episode_content["knowledge_pool"].append(knowledge_tensor)


            assert len(episode_content["context"]) == len(episode_content["response"]) == len(
                episode_content["knowledge_pool"]) == len(episode_content["knowledge_piece_mask"]) == len(
                episode_content["knowledge_label"]) == self.max_episode_length


            id_episode_tensor = torch.tensor([id]).long()  # 单tensor
            context_episode_tensor = torch.stack(episode_content["context"])  # [max_episode_length, context_len]
            response_episode_tensor = torch.stack(episode_content["response"])  # [max_episode_length, max_dec_length]
            knowledge_pool_episode_tensor = torch.stack(episode_content["knowledge_pool"])  # [max_episode_length, max_knowledge_pool, knowledge_sentence_len]
            knowledge_piece_mask_episode_tensor = torch.stack(episode_content["knowledge_piece_mask"])  # [max_episode_length, max_knowledge_pool]
            knowledge_label_episode_tensor = torch.cat(episode_content["knowledge_label"])  # [max_episode_length]

            self.episode_tensor.append(
                [id_episode_tensor, context_episode_tensor, response_episode_tensor, knowledge_pool_episode_tensor,
                 knowledge_piece_mask_episode_tensor, knowledge_label_episode_tensor, episode_mask])

            self.len = id + 1

            if id >= self.n:
                break

    def __getitem__(self, index):
        episode = self.episode_tensor[index]
        return [episode[0], episode[1], episode[2], episode[3], episode[4], episode[5], episode[6]]

    def __len__(self):
        return self.len

    def context_id(self, episode_id, example_id):
        return self.episodes[episode_id][example_id]['context_id']  # list

    def query_id(self, episode_id, example_id):
        return self.episodes[episode_id][example_id]['query_id']  # string

    def passage_id(self, episode_id, example_id):
        return self.episodes[episode_id][example_id]['knowledge_label']  # list

    def knowledge_pool(self, episode_id, example_id):
        return self.episodes[episode_id][example_id]['knowledge_pool']  # list


def collate_fn(data):
    id_episodes, context_episodes, response_episodes, knowledge_pool_episodes, knowledge_piece_mask_episodes, knowledge_label_episodes, episode_mask_episodes = zip(*data)

    return {'episode_id': torch.cat(id_episodes),  # [batch]
            'context': torch.stack(context_episodes),  # [batch, max_episode_length, context_len]
            'response': torch.stack(response_episodes),  # [batch, max_episode_length, max_dec_length]
            'knowledge_pool': torch.stack(knowledge_pool_episodes),  # [batch, max_episode_length, max_knowledge_pool, knowledge_sentence_len]
            'knowledge_piece_mask': torch.stack(knowledge_piece_mask_episodes),  # [batch, max_episode_length, max_knowledge_pool]
            'knowledge_label': torch.stack(knowledge_label_episodes),  # [batch, max_episode_length]
            'episode_mask': torch.stack(episode_mask_episodes)     # [batch, max_episode_length]
    }

