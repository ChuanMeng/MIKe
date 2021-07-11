import sys
sys.path.append('./')
from torch.utils.data import DataLoader
from evaluation.Eval_Rouge import *
from evaluation.Eval_Bleu import *
from evaluation.Eval_Meteor import *
from evaluation.Eval_F1 import *
from evaluation.Eval_Distinct import *
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import accuracy_score
from MIKe.Utils import *
import json
import os


def rounder(num, places):
    return round(num, places)


class CumulativeTrainer(object):
    def __init__(self, name, model, tokenizer, detokenizer, local_rank=None, accumulation_steps=None):
        super(CumulativeTrainer, self).__init__()
        self.local_rank = local_rank
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer
        self.name = name

        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model

        self.eval_model = self.model


    def predict(self, method, dataset, collate_fn, batch_size, epoch, output_path):
        #  changes the forward() behaviour of the module it is called upon. eg, it disables dropout and has batch norm use the entire population statistics
        self.eval_model.eval()

        with torch.no_grad():
            test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

            accumulative_final_ks_pred = []
            accumulative_knowledge_label = []
            accumulative_episode_mask = []
            systems = []

            for k, data in enumerate(test_loader, 0):
                print("doing {} / total {} in {}".format(k+1, len(test_loader), epoch))
                if torch.cuda.is_available():
                    data_cuda = dict()
                    for key, value in data.items():
                        if isinstance(value, torch.Tensor):
                            data_cuda[key] = value.cuda()
                        else:
                            data_cuda[key] = value
                    data = data_cuda

                # indices [batch * max_episode_length, max_target_length]
                # final_ks_pred [batch * max_episode_length]
                indices,  final_ks_pred = self.eval_model(data, method=method)

                accumulative_final_ks_pred.append(final_ks_pred)  # [[batch * max_episode_length],...]
                accumulative_knowledge_label.append(data['knowledge_label'].reshape(-1)) # [[batch * max_episode_length],...]
                accumulative_episode_mask.append(data['episode_mask'].reshape(-1))  # [[batch * max_episode_length],...]

                _, max_episode_length = data['episode_mask'].size()

                sents = self.eval_model.to_sentence(data, indices)  # [[tokens],[tokens]...batch * max_episode_length]

                for i in range(len(data['episode_id'])):
                    episode_id = data['episode_id'][i].item()
                    for example_id in range(max_episode_length):
                        if not data['episode_mask'][i][example_id]:
                            continue
                        offset = i * max_episode_length + example_id
                        systems.append([';'.join(dataset.context_id(episode_id, example_id)),
                                        dataset.query_id(episode_id, example_id),
                                        dataset.knowledge_pool(episode_id, example_id)[final_ks_pred[offset].item()],
                                        self.detokenizer(sents[offset])])

            accumulative_final_ks_pred = torch.cat(accumulative_final_ks_pred, dim=0)
            accumulative_knowledge_label = torch.cat(accumulative_knowledge_label, dim=0)
            accumulative_episode_mask = torch.cat(accumulative_episode_mask, dim=0)
            accumulative_final_ks_acc = accuracy_score(accumulative_knowledge_label.cpu(), accumulative_final_ks_pred.cpu(), sample_weight=accumulative_episode_mask.cpu())

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            output_path = os.path.join(output_path, str(epoch) + '.txt')

            file = codecs.open(output_path, "w", "utf-8")
            for i in range(len(systems)):
                file.write('\t'.join(systems[i])+os.linesep)
            file.close()
        return output_path, dataset.answer_file, {"ks_acc": rounder(100*(accumulative_final_ks_acc), 2)}

    def test(self, method, dataset, collate_fn, batch_size, dataset_name, epoch, output_path):
        #  disables tracking of gradients in autograd.
        # In this mode, the result of every computation will have requires_grad=False, even when the inputs have requires_grad=True.
        with torch.no_grad():
            run_file, answer_file, final_ks_acc = self.predict(method, dataset, collate_fn, batch_size, dataset_name+"_"+epoch, output_path)

        print("Start auotimatic evaluation")

        print("KNOW_ACC", final_ks_acc)

        f1 = eval_f1_file(run_file, answer_file, self.tokenizer, self.detokenizer)
        print("F1", f1)

        bleus = eval_bleu_file(run_file, answer_file, self.tokenizer, self.detokenizer)
        print("BLEU", bleus)

        rouges = eval_rouge_file(run_file, answer_file, self.tokenizer, self.detokenizer)
        print("ROUGE", rouges)

        meteors = eval_meteor_file(run_file, answer_file, self.tokenizer, self.detokenizer)
        print("METEOR", meteors)


        metric_output = {**f1, **bleus, **rouges, **meteors, **final_ks_acc}
        print({epoch+"_"+dataset_name: metric_output})

        try:
            with open(os.path.join(output_path, dataset_name + "_result.json"), 'r', encoding='utf-8') as r:
                result_log = json.load(r)
            result_log[epoch + "_" + dataset_name] = metric_output
            with open(os.path.join(output_path, dataset_name + "_result.json"), 'w', encoding='utf-8') as w:
                json.dump(result_log, w)

        except FileNotFoundError:
            with open(os.path.join(output_path, dataset_name + "_result.json"), 'w', encoding='utf-8') as w:
                result_log={}
                result_log[epoch+"_"+dataset_name] = metric_output
                json.dump(result_log, w)

        return None


if __name__ == '__main__':

    from transformers import BertTokenizer
    from evaluation.Eval_Multi_acc import *

    def bert_tokenizer():
        t = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True)  # do_lower_case Whether to lower case the input.
        return t.tokenize, t.vocab, t.ids_to_tokens

    def bert_detokenizer():
        def detokenizer(tokens):
            return ' '.join(tokens).replace(' ##', '').strip()
        return detokenizer


    tokenizer, vocab2id, id2vocab = bert_tokenizer()
    detokenizer = bert_detokenizer()


    dataset_name="test"
    run_file = "output/MIKe_Holl_E/test_13.txt"
    answer_file ="datasets/holl_e/holl_e.multi_answer"


    multi_acc = eval_multi_acc_file(run_file, answer_file)
    print(multi_acc)

    f1 = eval_f1_file(run_file, answer_file, tokenizer, detokenizer)
    print("F1", f1)

    bleus = eval_bleu_file(run_file, answer_file, tokenizer, detokenizer)
    print("BLEU", bleus)

    rouges = eval_rouge_file(run_file, answer_file, tokenizer, detokenizer)
    print("ROUGE", rouges)

    meteors = eval_meteor_file(run_file, answer_file, tokenizer, detokenizer)
    print("METEOR", meteors)



