import os
import json
import argparse
from tqdm import tqdm
from copy import copy
import logging
import transformers
from data import StrategyQA, WikiMultiHopQA, HotpotQA, IIRC, ArabicaQA
from generate import *
import ray
# import debugpy

# # Set the port for the debugger (e.g., 5678)
# debugpy.listen(5679)
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()  # Pause execution until the debugger is attached
# print("Debugger attached. Continuing execution...")
# Set logging level to ERROR to suppress warnings
import warnings
ray.init(ignore_reinit_error=True,logging_level=logging.ERROR,num_gpus=1)
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)
# ray.init(ignore_reinit_error=True,address='10.233.97.192:91909')

@ray.remote(num_cpus=0)
class ProgressBar:
    def __init__(self, total):
        self.current_progress = 0
        self.pbar = tqdm(total=total)

    def update(self, n=1):
        self.current_progress += n
        if self.current_progress%1==0:
            self.pbar.update(1)
            # print("Completed:",self.current_progress)

    def get_progress(self):
        return self.current_progress

    def close(self):
    # self.pb.close()
        self.pbar.close()

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

# @ray.remote(num_gpus=0.35)
# def process_batch(args,data,progress_actor):
#     if args.method == "non-retrieval":
#         model = BasicRAG(args)
#     elif args.method == "single-retrieval":
#         model = SingleRAG(args)
#     elif args.method == "fix-length-retrieval" or args.method == "fix-sentence-retrieval":
#         model = FixLengthRAG(args)
#     elif args.method == "token":
#         model = TokenRAG(args)
#     elif args.method == "entity":
#         model = EntityRAG(args)
#     elif args.method == "attn_prob" or args.method == "dragin":
#         model = AttnWeightRAG(args)
#     else:
#         raise NotImplementedError

#     # logger.info("start inference")
#     batch_size = 1
#     batch_res = []
#     for i in range(0,len(data['qid']),batch_size):
#         last_counter = copy(model.counter)
#         pred = model.inference(data["question"][i], data["demo"][i], data["case"][i])
#         pred = pred.strip()
#         ret = {
#             "qid": data["qid"][i], 
#             "prediction": pred,
#         }
#         if args.use_counter:
#             ret.update(model.counter.calc(last_counter))
#         batch_res.append(ret)
#         progress_actor.update.remote(1)
#         # output_file.write(json.dumps(ret,ensure_ascii=False)+"\n")
#     return batch_res

# def main():
#     args = get_args()
#     logger.info(f"{args}")

#     # output dir
#     if os.path.exists(args.output_dir) is False:
#         os.makedirs(args.output_dir)
#     dir_name = os.listdir(args.output_dir)
#     for i in range(10000):
#         if str(i) not in dir_name:
#             args.output_dir = os.path.join(args.output_dir, str(i))
#             os.makedirs(args.output_dir)
#             break
#     logger.info(f"output dir: {args.output_dir}")
#     # save config
#     with open(os.path.join(args.output_dir, "config.json"), "w") as f:
#         json.dump(args.__dict__, f, indent=4)
#     # create output file
#     output_file = open(os.path.join(args.output_dir, "output.txt"), "w")

#     # load data
#     if args.dataset == "strategyqa":
#         data = StrategyQA(args.data_path)
#     elif args.dataset == "2wikimultihopqa":
#         data = WikiMultiHopQA(args.data_path)
#     elif args.dataset == "hotpotqa":
#         data = HotpotQA(args.data_path)
#     elif args.dataset == "iirc":
#         data = IIRC(args.data_path)
#     elif args.dataset == "arabicaqa":
#         data = ArabicaQA(args.data_path)
#     else:
#         raise NotImplementedError
#     data.format(fewshot=args.fewshot)
#     data = data.dataset
#     if args.shuffle:
#         data = data.shuffle(seed=42)
#     if args.sample != -1:
#         samples = min(len(data), args.sample)
#         data = data.select(range(samples))
   
#     # 根据 method 选择不同的生成策略
#     # if args.method == "non-retrieval":
#     #     model = BasicRAG(args)
#     # elif args.method == "single-retrieval":
#     #     model = SingleRAG(args)
#     # elif args.method == "fix-length-retrieval" or args.method == "fix-sentence-retrieval":
#     #     model = FixLengthRAG(args)
#     # elif args.method == "token":
#     #     model = TokenRAG(args)
#     # elif args.method == "entity":
#     #     model = EntityRAG(args)
#     # elif args.method == "attn_prob" or args.method == "dragin":
#     #     model = AttnWeightRAG(args)
#     # else:
#     #     raise NotImplementedError

#     logger.info("start inference")
#     batch_size = 8
#     progress_actor = ProgressBar.remote(len(data))
#     futures = [process_batch.remote(args,data[i:i+batch_size],progress_actor) for i in range(0,len(data),batch_size)]
#     results = []
#     for future in tqdm(ray.get(futures), total=len(range(0,len(data),batch_size)), desc="Collecting Results"):
#         results.extend(future)  # Combine all results
#     ray.get(progress_actor.close.remote())
#     ray.shutdown()

#     for ret in results:
#         output_file.write(json.dumps(ret,ensure_ascii=False)+"\n")

#     # for i in tqdm(range(0,len(data),batch_size)):
#     #     last_counter = copy(model.counter)
#     #     batch = data[i]
#     #     pred = model.inference(batch["question"], batch["demo"], batch["case"])
#     #     pred = pred.strip()
#     #     ret = {
#     #         "qid": batch["qid"], 
#     #         "prediction": pred,
#     #     }
#     #     if args.use_counter:
#     #         ret.update(model.counter.calc(last_counter))
#     #     output_file.write(json.dumps(ret,ensure_ascii=False)+"\n")

#         # pred = model.inference(batch["question"], batch["demo"], batch["case"])
#         # for pred_txt, qid,question in zip(pred,batch["qid"],batch['question']):
#         #     # print("Question:",question,"answer:",pred_txt)
#         #     ret = {
#         #         "qid": qid, 
#         #         "prediction": pred_txt,
#         #     }
#         #     if args.use_counter:
#         #         ret.update(model.counter.calc(last_counter))
#         # # print('>>>>>>>>> in main, ret: \n', ret, "\n<<<<<<<<<<<<<")
#         # # print('###'*50)
#         #     output_file.write(json.dumps(ret,ensure_ascii=False)+"\n")
    

# if __name__ == "__main__":
#     main()

@ray.remote(num_gpus=0.33)
class ModelActor:
    def __init__(self, args):
        if args.method == "non-retrieval":
            self.model = BasicRAG(args)
        elif args.method == "single-retrieval":
            self.model = SingleRAG(args)
        elif args.method in ["fix-length-retrieval", "fix-sentence-retrieval"]:
            self.model = FixLengthRAG(args)
        elif args.method == "token":
            self.model = TokenRAG(args)
        elif args.method == "entity":
            self.model = EntityRAG(args)
        elif args.method in ["attn_prob", "dragin"]:
            self.model = AttnWeightRAG(args)
        else:
            raise NotImplementedError
    
    def inference(self, question, demo, case):
        return self.model.inference(question, demo, case)

    def get_counter(self):
        return copy(self.model.counter)

    def calculate_counter(self, last_counter):
        return self.model.counter.calc(last_counter)

@ray.remote
def process_batch(args, data, progress_actor, model_actor):
    batch_res = []
    for i in range(len(data['qid'])):
        last_counter = ray.get(model_actor.get_counter.remote())
        pred = ray.get(model_actor.inference.remote(data["question"][i], data["demo"][i], data["case"][i]))
        pred = pred.strip()
        ret = {
            "qid": data["qid"][i], 
            "prediction": pred,
        }
        if args.use_counter:
            counter_update = ray.get(model_actor.calculate_counter.remote(last_counter))
            ret.update(counter_update)
        batch_res.append(ret)
        progress_actor.update.remote(1)
    return batch_res

def main():
    args = get_args()
    logger.info(f"{args}")

    # Prepare output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    for i in range(10000):
        if str(i) not in os.listdir(args.output_dir):
            args.output_dir = os.path.join(args.output_dir, str(i))
            os.makedirs(args.output_dir)
            break
    logger.info(f"output dir: {args.output_dir}")
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)
    output_file = open(os.path.join(args.output_dir, "output.txt"), "w")

    # Load and preprocess data
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
    batch_size = 8

    # Create actors
    progress_actor = ProgressBar.remote(len(data))
    num_actors = 4  # Adjust this based on available GPUs/resources
    model_actors = [ModelActor.remote(args) for _ in range(num_actors)]

    # Distribute tasks among actors
    futures = []
    for i, batch_start in enumerate(range(0, len(data), batch_size)):
        model_actor = model_actors[i % num_actors]  # Round-robin distribution
        futures.append(process_batch.remote(args, data[batch_start:batch_start + batch_size], progress_actor, model_actor))

    results = []
    for future in tqdm(ray.get(futures), total=len(futures), desc="Collecting Results"):
        results.extend(future)
    ray.get(progress_actor.close.remote())

    # Write results
    for ret in results:
        output_file.write(json.dumps(ret, ensure_ascii=False) + "\n")

    ray.shutdown()

if __name__ == "__main__":
    main()
