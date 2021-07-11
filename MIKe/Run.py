import sys
sys.path.append('./')
from MIKe.Dataset import *
from torch import optim
from MIKe.CumulativeTrainer import *
import torch.backends.cudnn as cudnn
import argparse
from MIKe.Model import *
from dataset.Utils_MIKe import *
from transformers import get_constant_schedule
import os
import time


def inference(args):
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True
    print("torch_version:{}".format(torch.__version__))
    print("CUDA_version:{}".format(torch.version.cuda))
    print("cudnn_version:{}".format(cudnn.version()))

    data_path = args.base_data_path + args.dataset + '/'

    tokenizer, vocab2id, id2vocab = bert_tokenizer()
    detokenizer = bert_detokenizer()

    print('Vocabulary size', len(vocab2id))

    if os.path.exists(data_path + 'test_seen_MIKe.pkl') or os.path.exists(data_path + 'test_MIKe.pkl'):
        query = torch.load(data_path + 'query_MIKe.pkl')
        passage = torch.load(data_path + 'passage_MIKe.pkl')

        if args.dataset == "wizard_of_wikipedia":
            test_seen_episodes = torch.load(data_path + 'test_seen_MIKe.pkl')
            test_unseen_episodes = torch.load(data_path + 'test_unseen_MIKe.pkl')
            print("The number of test_seen_episodes:", len(test_seen_episodes))
            print("The number of test_unseen_episodes:", len(test_unseen_episodes))
        elif args.dataset == "holl_e":
            test_episodes = torch.load(data_path + 'test_MIKe.pkl')
            print("The number of test_episodes:", len(test_episodes))

    else:
        episodes, query, passage = load_default(args.dataset, data_path + args.dataset + '.answer',
                                                                   data_path + args.dataset + '.passage',
                                                                   data_path + args.dataset + '.pool',
                                                                   data_path + args.dataset + '.qrel',
                                                                   data_path + args.dataset + '.query',
                                                                   tokenizer)

        if args.dataset == "wizard_of_wikipedia":
            train_episodes, dev_episodes, test_seen_episodes, test_unseen_episodes = split_data(args.dataset, data_path + args.dataset + '.split', episodes)
            print("The number of test_seen_episodes:", len(test_seen_episodes))
            print("The number of test_unseen_episodes:", len(test_unseen_episodes))
            torch.save(test_seen_episodes, data_path + 'test_seen_MIKe.pkl')
            torch.save(test_unseen_episodes, data_path + 'test_unseen_MIKe.pkl')

        elif args.dataset == "holl_e":
            train_episodes, dev_episodes, test_episodes, = split_data(args.dataset, data_path + args.dataset + '.split', episodes)
            print("The number of test_episodes:", len(test_episodes))
            torch.save(test_episodes, data_path + 'test_MIKe.pkl')

        print("The number of train_episodes:", len(train_episodes))
        torch.save(query, data_path + 'query_MIKe.pkl')
        torch.save(passage, data_path + 'passage_MIKe.pkl')
        torch.save(train_episodes, data_path + 'train_MIKe.pkl')


    if args.dataset == "wizard_of_wikipedia":
        test_seen_dataset = Dataset(args.mode, test_seen_episodes, query, passage, vocab2id, args.max_episode_length,
                              args.max_knowledge_pool_when_train, args.max_knowledge_pool_when_inference,
                              args.knowledge_sentence_len, args.context_len, args.max_dec_length)

        test_unseen_dataset = Dataset(args.mode, test_unseen_episodes, query, passage, vocab2id, args.max_episode_length,
                              args.max_knowledge_pool_when_train, args.max_knowledge_pool_when_inference,
                              args.knowledge_sentence_len, args.context_len, args.max_dec_length)

    elif args.dataset == "holl_e":
        test_dataset = Dataset(args.mode, test_episodes, query, passage, vocab2id, args.max_episode_length,
                              args.max_knowledge_pool_when_train, args.max_knowledge_pool_when_inference,
                              args.knowledge_sentence_len, args.context_len, args.max_dec_length)

    saved_model_path = os.path.join(args.base_output_path + args.name + "/", 'model/')

    def inference(dataset, epoch=None):
        file =saved_model_path + str(epoch) + '.pkl'
        if os.path.exists(file):
            model = MIKe(vocab2id, id2vocab, args)

            model.load_state_dict(torch.load(file)["model"])
            trainer = CumulativeTrainer(args.name, model, tokenizer, detokenizer, None)

            if dataset == "wizard_of_wikipedia":
                print('inference {}'.format("test_seen_dataset"))
                trainer.test('inference', test_seen_dataset, collate_fn, args.inference_batch_size, 'test_seen', str(epoch), output_path=args.base_output_path + args.name+"/")
                print('inference {}'.format("test_unseen_dataset"))
                trainer.test('inference', test_unseen_dataset, collate_fn, args.inference_batch_size, 'test_unseen', str(epoch), output_path=args.base_output_path + args.name+"/")

            elif dataset == "holl_e":
                print('inference {}'.format("test_dataset"))
                trainer.test('inference', test_dataset, collate_fn, args.inference_batch_size, 'test', str(epoch), output_path=args.base_output_path + args.name+"/")

    if not os.path.exists(saved_model_path+"finished_inference.json"):
        finished_inference = {"time": []}
        w = open(saved_model_path+"finished_inference.json", 'w', encoding='utf-8')
        json.dump(finished_inference, w)
        w.close()

    if args.appoint_epoch != -1:
        print('Start inference at epoch', args.appoint_epoch)
        inference(args.dataset, args.appoint_epoch)

        r = open(saved_model_path+"finished_inference.json", 'r', encoding='utf-8')
        finished_inference = json.load(r)
        r.close()

        finished_inference["time"].append(args.appoint_epoch)
        w = open(saved_model_path + "finished_inference.json", 'w', encoding='utf-8')
        json.dump(finished_inference, w)
        w.close()
        print("finished epoch {} inference".format(args.appoint_epoch))
        exit()

    while True:
        with open(saved_model_path + "checkpoints.json", 'r', encoding='utf-8') as r:
            checkpoints = json.load(r)

        r = open(saved_model_path + "finished_inference.json", 'r', encoding='utf-8')
        finished_inference = json.load(r)
        r.close()

        if len(checkpoints["time"]) == 0:
            print('Inference_mode: wait train finish the first epoch...')
            time.sleep(300)
        else:
            for i in checkpoints["time"]:  # i is the index of epoch
                if i in finished_inference["time"]:
                    print("epoch {} already has been inferenced, skip it".format(i))
                    pass
                else:
                    print('Start inference at epoch', i)
                    inference(args.dataset, i)

                    r = open(saved_model_path + "finished_inference.json", 'r', encoding='utf-8')
                    finished_inference = json.load(r)
                    r.close()

                    finished_inference["time"].append(i)

                    w = open(saved_model_path+"finished_inference.json", 'w', encoding='utf-8')
                    json.dump(finished_inference, w)
                    w.close()
                    print("finished epoch {} inference".format(i))

            print("Inference_mode: current all model checkpoints are completed...")
            print("Inference_mode: finished %d modes" % len(finished_inference["time"]))
            if len(finished_inference["time"]) == args.epoches:
                print("All inference is ended")
                break
            else:
                print('Inference_mode: wait train finish the next epoch...')
                time.sleep(300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)

    parser.add_argument("--name", type=str, default='MIKe')
    parser.add_argument("--base_output_path", type=str, default='output/')
    parser.add_argument("--base_data_path", type=str, default='datasets/')
    parser.add_argument("--dataset", type=str, default='holl_e')  # wizard_of_wikipedia
    parser.add_argument("--GPU", type=int, default=2)

    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--epoches", type=int, default=15)
    parser.add_argument("--accumulation_steps", type=int, default=1)  # with BERT should increase
    parser.add_argument("--lr", type=float, default=0.00002)  # 0.00002
    parser.add_argument("--train_batch_size", type=int, default=32)  # bert 4; no bert 32; debug 2
    parser.add_argument("--inference_batch_size", type=int, default=16)
    parser.add_argument("--appoint_epoch", type=int, default=-1)

    parser.add_argument("--max_episode_length", type=int, default=5)
    parser.add_argument("--context_len", type=int, default=52)
    parser.add_argument("--max_dec_length", type=int, default=52)
    parser.add_argument("--knowledge_sentence_len", type=int, default=34)
    parser.add_argument("--max_knowledge_pool_when_train", type=int, default=32)
    parser.add_argument("--max_knowledge_pool_when_inference", type=int, default=128)

    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embedding_dropout", type=float, default=0.1)
    parser.add_argument("--embedding_size", type=int, default=768)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=5)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--ffn_size", type=int, default=768)

    parser.add_argument("--k_hidden_size", type=int, default=768)
    parser.add_argument("--k_n_layers", type=int, default=5)
    parser.add_argument("--k_n_heads", type=int, default=2)
    parser.add_argument("--k_ffn_size", type=int, default=768)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
    if args.mode == 'inference':
        inference(args)
    elif args.mode == 'train':
        train(args)
    else:
        Exception("no ther mode")