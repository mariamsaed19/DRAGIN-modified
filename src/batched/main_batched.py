import os
import json
import argparse
from tqdm import tqdm
from copy import copy
import logging
from data import StrategyQA, WikiMultiHopQA, HotpotQA, IIRC, ArabicaQA
from generate import *
import debugpy

# Set the port for the debugger (e.g., 5678)
debugpy.listen(5679)
print("Waiting for debugger to attach...")
debugpy.wait_for_client()  # Pause execution until the debugger is attached
print("Debugger attached. Continuing execution...")

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True)
    args = parser.parse_args()
    config_path = args.config_path
    with open(config_path, "r") as f:
        args = json.load(f)
    args = argparse.Namespace(**args)
    args.config_path = config_path
    if "shuffle" not in args:
        args.shuffle = False 
    if "use_counter" not in args:
        args.use_counter = True
    return args


def main():
    args = get_args()
    logger.info(f"{args}")
    start_sample=0
    end_sample=4
    # output dir
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    dir_name = os.listdir(args.output_dir)
    # for i in range(10000):
    #     if str(i) not in dir_name:
    #         args.output_dir = os.path.join(args.output_dir, str(i))
    os.makedirs(args.output_dir,exist_ok=True)
            # break
    logger.info(f"output dir: {args.output_dir}")
    # save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)
    # create output file
    output_file = open(os.path.join(args.output_dir, f"output-{start_sample}-{end_sample}.txt"), "w")

    # load data
    if args.dataset == "strategyqa":
        data = StrategyQA(args.data_path)
    elif args.dataset == "2wikimultihopqa":
        data = WikiMultiHopQA(args.data_path)
    elif args.dataset == "hotpotqa":
        data = HotpotQA(args.data_path)
    elif args.dataset == "iirc":
        data = IIRC(args.data_path)
    elif args.dataset == "arabicaqa":
        data = ArabicaQA(args.data_path)
    else:
        raise NotImplementedError
    data.format(fewshot=args.fewshot)
    data = data.dataset
    if args.shuffle:
        data = data.shuffle(seed=42)
    if args.sample != -1:
        samples = min(len(data), args.sample)
        data = data.select(range(samples))
   
    # 根据 method 选择不同的生成策略
    if args.method == "non-retrieval":
        model = BasicRAG(args)
    elif args.method == "single-retrieval":
        model = SingleRAG(args)
    elif args.method == "fix-length-retrieval" or args.method == "fix-sentence-retrieval":
        model = FixLengthRAG(args)
    elif args.method == "token":
        model = TokenRAG(args)
    elif args.method == "entity":
        model = EntityRAG(args)
    elif args.method == "attn_prob" or args.method == "dragin":
        model = AttnWeightRAG(args)
    else:
        raise NotImplementedError

    logger.info("start inference")
    batch_size = 1
    for i in tqdm(range(start_sample,min(end_sample,len(data)),batch_size)):
        last_counter = copy(model.counter)
        batch = data[i:i+batch_size]
        # print(batch)
        # print("\n Question:",batch["question"],flush=True) 
        pred_all = model.inference(batch["question"], batch["demo"], batch["case"]) #NOTE: modify batch
        # print("****"*5)
        for sample_idx,pred in enumerate(pred_all):
            pred = pred.strip()
            ret = {
                "qid": batch["qid"][sample_idx], 
                "prediction": pred,
            }
            if args.use_counter:
                ret.update(model.counter.calc(last_counter))
            output_file.write(json.dumps(ret,ensure_ascii=False)+"\n")

        # pred = model.inference(batch["question"], batch["demo"], batch["case"])
        # for pred_txt, qid,question in zip(pred,batch["qid"],batch['question']):
        #     # print("Question:",question,"answer:",pred_txt)
        #     ret = {
        #         "qid": qid, 
        #         "prediction": pred_txt,
        #     }
        #     if args.use_counter:
        #         ret.update(model.counter.calc(last_counter))
        # # print('>>>>>>>>> in main, ret: \n', ret, "\n<<<<<<<<<<<<<")
        # # print('###'*50)
        #     output_file.write(json.dumps(ret,ensure_ascii=False)+"\n")
    

if __name__ == "__main__":
    main()