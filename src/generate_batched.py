import numpy as np
import logging
import spacy
import torch
from math import exp
from scipy.special import softmax
from retriever import BM25, SGPT, AraDPR
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import re
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.tagger.default import DefaultTagger
import time
import openai
from sglang.utils import (
    execute_shell_command,
    wait_for_server,
    terminate_process,
    print_highlight,
)
from datasets import Dataset
# from datasets import disable_progress_bar

# # Disable all progress bars
# disable_progress_bar()

logging.basicConfig(
    filename="datasets_log.txt",  # File to write the logs to
    level=logging.INFO,           # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log message format
    filemode="w"                  # Use 'w' to overwrite the file each run, or 'a' to append
)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")
mled = BERTUnfactoredDisambiguator.pretrained()
tagger = DefaultTagger(mled, 'pos')

client = openai.Client(base_url="http://127.0.0.1:30024/v1", api_key="None")
tokenizer = AutoTokenizer.from_pretrained("/scratch/mariam.saeed/side/advanced-nlp/models/aya-8bit")

def generate_response(example):
    response = []
    for input_text in example['input_text']:
        if input_text=="hi":
            response.append("hi")
            continue
        # Tokenize and truncate the input text
        input_ids = tokenizer.encode(input_text, truncation=True, max_length=7100, return_tensors="pt")[0]
        input_text = tokenizer.decode(input_ids)
        
        # Generate response
        while True:
            try:
                outputs = client.chat.completions.create(
                    model="CohereForAI/aya-expanse-8b",
                    messages=[
                        {"role": "user", "content": input_text},
                    ],
                    temperature=0,
                    max_tokens=1024,
                )
                break
            except Exception as e:
                continue
        response.append(outputs.choices[0].message.content)
    return {"response": response}

def generate_response_attn(example):
    response = []
    tmp_all = []
    for input_text in example['input_text']:
        if input_text=="hi":
            response.append("hi")
            tmp_all.append(np.random.rand(5,1,20))
            continue
        # Tokenize and truncate the input text
        input_ids = tokenizer.encode(input_text, truncation=True, max_length=7100, return_tensors="pt")[0]
        input_text = tokenizer.decode(input_ids)
        
        # Generate response
        while True:
            try:
                outputs = client.chat.completions.create(
                    model="CohereForAI/aya-expanse-8b",
                    messages=[
                        {"role": "user", "content": input_text},
                    ],
                    temperature=0,
                    max_tokens=1024,
                    logprobs=True,
                    top_logprobs=150
                )
                break
            except Exception as e:
                continue
        tmp = []
        # for token_probs in outputs[0].outputs[0].logprobs:
        #     curr_probs = []
        #     for idx,item in enumerate(token_probs.values()):
        #         if idx ==20: break
        #         curr_probs.append(item.logprob)
        #     tmp.append([curr_probs])
        for token_probs in outputs.choices[0].logprobs.content:
            curr_probs = []
            for idx,item in enumerate(token_probs.top_logprobs):

                if idx ==20: break
                curr_probs.append(item.logprob)
            tmp.append([curr_probs])
        # print(np.array(tmp).shape)
        response.append(outputs.choices[0].message.content)
        tmp_all.append(np.array(tmp))
    return {"response": response , "softmax_probs":tmp_all}

class BasicGenerator:
    def __init__(self, model_name_or_path):
        logger.info(f"Loading model from {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model_config = AutoConfig.from_pretrained(model_name_or_path,
                    trust_remote_code = "falcon" in model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", 
                    trust_remote_code = "falcon" in model_name_or_path)
        
        if self.model_config.model_type == "llama":
            self.space_token = "▁"
        else:
            self.space_token = " " #self.tokenizer.tokenize(' ')[0]
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    

    def generate(self, input_text, max_length, return_logprobs=False):
        data = {"input_text": input_text}
        dataset = Dataset.from_dict(data)

        # Map the generate_response function with multiprocessing
        result_dataset = dataset.map(
            generate_response,
            batched=True,
            batch_size=2,
            num_proc=32,  # Number of processes to use
        )

        # Extract the responses
        responses = result_dataset["response"]

        return responses, None, None

    def generate_batched(self, input_text, max_length, return_logprobs=False):
        # input_ids = self.tokenizer.batch_encode_plus(input_text, padding=True,return_tensors="pt")['input_ids']
        encoded_input = self.tokenizer.batch_encode_plus(input_text, padding=True,return_tensors="pt")
        input_ids = encoded_input['input_ids']
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = encoded_input['attention_mask'].to(self.model.device)#torch.ones_like(input_ids)
        # input_ids = input_ids.to(self.model.device)
        # input_length = input_ids.shape[1]
        # attention_mask = torch.ones_like(input_ids)

        if return_logprobs:
            outputs = self.model.generate(
                input_ids = input_ids, 
                attention_mask = attention_mask,
                max_new_tokens = max_length, 
                return_dict_in_generate = True, 
                output_scores = True,
            )
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )

            generated_tokens = outputs.sequences[:, input_length:]
            text = self.tokenizer.decode(generated_tokens[0]) # text = "".join(tokens)
            tokens = [self.tokenizer.decode(t) for t in generated_tokens[0]]
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            return text, tokens, logprobs
        
        else:
            # print('>>>>>>>> in model.generate with temperature & top_p')
            outputs = self.model.generate(
                input_ids = input_ids, 
                max_new_tokens = max_length, 
                attention_mask = attention_mask,
                # top_p = 0.85,
                # temperature = 0.2,
            )
            generated_tokens = outputs[:, input_length:]
            text = self.tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)
            return text, None, None
    
    def generate_attn(self, input_text, max_length, solver="max", use_entropy = False, use_logprob = False):
        # input_ids = self.tokenizer.batch_encode_plus(input_text, padding=True,return_tensors="pt")["input_ids"]
        data = {"input_text": input_text}
        # data = {"input_text":["hello","hello","hello","hello"]}
        dataset = Dataset.from_dict(data)

        # Map the generate_response function with multiprocessing
        result_dataset = dataset.map(
            generate_response_attn,
            batched=True,
            batch_size=2,
            num_proc=32,  # Number of processes to use
        )

        # Extract the responses
        responses_text, responses_probs = result_dataset["response"],result_dataset["softmax_probs"]

        # generated_tokens_all = outputs.sequences[:, input_length:]
        tokens_batch = []
        text_batch= []
        seqlist_batch = []
        attns_batch = [] 
        seqlogprobs_batch = [] 
        seqentropies_batch = []
        total_time_attn = 0
        for sample_idx,(res_text,res_prob) in enumerate(zip(responses_text, responses_probs)):
            generated_tokens = self.tokenizer.encode(res_text, return_tensors="pt").to(self.model.device)
            # tokens = self.tokenizer.convert_ids_to_tokens(generated_tokens[0])
            tokens =  [self.tokenizer.decode([token]) for token in generated_tokens[0]]
            tokens = [token for token in tokens if token != self.tokenizer.pad_token]

            text = self.tokenizer.decode(generated_tokens[0],skip_special_tokens=True)
            tokens_batch.append(tokens)
            text_batch.append(text)
            # merge tokens
            range_ = []
            for i, t in enumerate(tokens):
                # if i == 0 or t.startswith(self.space_token) or generated_tokens[0][i] == 13 or tokens[i-1] == '</s>':
                if i == 0 or t.startswith(' ') or generated_tokens[0][i] == self.tokenizer.encode("\n")[-1] or tokens[i-1] == self.tokenizer.eos_token:  #NOTE: we need to replace the space token and 13 with \n and </s> with eos token
                    range_.append([i, i])
                else:
                    range_[-1][-1] += 1

            # attention
            st_time = time.time()
            atten = self.model(generated_tokens, output_attentions=True).attentions[-1][0]

            total_time_attn += time.time()-st_time
            if solver == "max": 
                mean_atten, _ = torch.max(atten, dim=1)
                mean_atten = torch.mean(mean_atten, dim=0)
            elif solver == "avg":
                mean_atten = torch.sum(atten, dim=1)
                mean_atten = torch.mean(mean_atten, dim=0)
                for i in range(mean_atten.shape[0]):
                    mean_atten[i] /= (mean_atten.shape[0] - i)
            elif solver == "last_token":
                mean_atten = torch.mean(atten[:, -1], dim=0)
            else:
                raise NotImplementedError
            if mean_atten.shape[0] > 1 and tokens[0] == self.tokenizer.eos_token: # check here again
                mean_atten = mean_atten / sum(mean_atten[1:]).item()
            # mean_atten = mean_atten[tl:tr]
                
            # regular tokens
            seqlist = []
            attns = []
            for r in range_:
                tokenseq = "".join(tokens[r[0]: r[1]+1]).replace(" ", "")
                value = sum(mean_atten[r[0]: r[1]+1]).item()
                seqlist.append(tokenseq)
                attns.append(value)
            seqlist_batch.append(seqlist)
            attns_batch.append(attns)

            # -log prob
            if use_logprob:
                transition_scores = self.model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True
                )
                logprobs = transition_scores[0]
                logprobs = [p.cpu().numpy() for p in logprobs]
                assert len(tokens) == len(logprobs)
                seqlogprobs = []
                for r in range_:
                    logprobseq = sum(logprobs[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                    seqlogprobs.append(logprobseq)
            else:
                seqlogprobs = None

            # entropy
            if use_entropy and len(res_prob)>0:
                # tmp = []
                # for v in outputs.scores[:len(tokens)]:
                #     tmp.append(v.cpu())
                # softmax_probs = np.exp(np.array(res_prob))
                # print("res:",eval(res_prob).shape)
                softmax_probs = np.exp(np.array(res_prob))
                entropies = -np.sum(softmax_probs * np.log(softmax_probs + 1e-10), axis=-1)
                entropies = [v[0] for v in entropies]
                seqentropies = []
                for r in range_:
                    entropyseq = sum(entropies[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                    seqentropies.append(entropyseq) 
            else:
                seqentropies = None 
            
            seqlogprobs_batch.append(seqlogprobs)
            seqentropies_batch.append(seqentropies)
        # print("total_attn_time",total_time_attn)
        return text_batch, seqlist_batch, attns_batch, seqlogprobs_batch, seqentropies_batch


class Counter:
    def __init__(self):
        self.retrieve = 0
        self.generate = 0
        self.hallucinated = 0
        self.token = 0
        self.sentence = 0

    def add_generate(self, text, tokenizer):
        self.generate += 1
        ids = tokenizer(text, return_tensors="pt")['input_ids'][0].tolist()
        self.token += len(ids)
        sentences = [sent.text for sent in nlp(text).sents]
        self.sentence += len(sentences)

    def calc(self, other_counter):
        return {
            "retrieve_count": self.retrieve - other_counter.retrieve, 
            "generate_count": self.generate - other_counter.generate,
            "hallucinated_count": self.hallucinated - other_counter.hallucinated, 
            "token_count": self.token - other_counter.token, 
            "sentence_count": self.sentence - other_counter.sentence 
        }
         

class BasicRAG:
    def __init__(self, args):
        args = args.__dict__ 
        for k, v in args.items():
            setattr(self, k, v)
        self.generator = BasicGenerator(self.model_name_or_path)
        if "retriever" in self.__dict__:
            self.retriever_type = self.retriever
            if self.retriever_type == "BM25":
                # gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                self.retriever = BM25(
                    tokenizer = self.generator.tokenizer, 
                    index_name = "wiki" if "es_index_name" not in args else self.es_index_name, 
                    engine = "elasticsearch",
                )
            elif self.retriever_type == "SGPT":
                self.retriever = SGPT(
                    model_name_or_path = self.sgpt_model_name_or_path, 
                    sgpt_encode_file_path = self.sgpt_encode_file_path,
                    passage_file = self.passage_file
                )
            elif self.retriever_type == "AraDPR":
                self.retriever = AraDPR("/scratch/mariam.saeed/side/advanced-nlp/data/dpr/wikiAr.tsv")
            
            
            #TODO: add dpr here
            else:
                raise NotImplementedError
        
        self.counter = Counter()

    def retrieve(self, query, topk=1, max_query_length=64):
        self.counter.retrieve += 1
        if self.retriever_type == "BM25":
            _docs_ids, docs = self.retriever.retrieve(
                queries = [query], 
                topk = topk, 
                max_query_length = max_query_length,
            )
            return docs[0]
        elif self.retriever_type == "SGPT":
            docs = self.retriever.retrieve(
                queries = [query], 
                topk = topk,
            )
            return docs[0] 
        elif self.retriever_type == "AraDPR":
            _docs_ids,docs = self.retriever.retrieve(queries=[query],topk=topk)
            return docs[0]
        #TODO: add dpr
        else:
            raise NotImplementedError
    
    def get_top_sentence(self, text):
        match = re.search(r"<reasoning>.*?</answer>", text, re.DOTALL)
        if match:
            extracted_text = match.group(0)
            return extracted_text
        else:
            return ""
        # sentences = [sent.text.strip() for sent in nlp(text).sents]
        # sentences = [sent for sent in sentences if len(sent) > 0]
        # return sentences[0] if len(sentences) > 0 else ""

    def get_last_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[-1] if len(sentences) > 0 else "" 

    def inference(self, question, demo, case):
        # non-retrieval
        assert self.query_formulation == "direct"
        # print("demo:",demo)
        prompt = "".join([d["case"]+"\n" for d in demo])
        if len(demo)>0:
            prompt+="As shown in the previous examples, "
        prompt += case
        text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text
    
    def inference_batched(self, questions, demos, cases):
        # non-retrieval
        assert self.query_formulation == "direct"
        prompts = []
        for demo,case in zip(demos,cases):
            prompt = "".join([d["case"]+"\n" for d in demo])
            prompt += case
            prompts.append(prompt)
        # print('>>>>>>>>> prompt: \n', prompt, "\n <<<<<<<<<<")
        text, _, _ = self.generator.generate(prompts, self.generate_max_length)
        if self.use_counter == True:
            for txt in text:
                self.counter.add_generate(txt, self.generator.tokenizer)
        # print('>>>>>>>>> text: \n', text, "\n <<<<<<<<<<<<<")
        return text
    

class SingleRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def inference(self, question, demo, case):
        assert self.query_formulation == "direct"
        docs = self.retrieve(question, topk=self.retrieve_topk)
        # 对 topk 个 passage 生成 prompt
        prompt = "".join([d["case"]+"\n" for d in demo])
        prompt += "Context:\n"
        for i, doc in enumerate(docs):
            prompt += f"[{i+1}] {doc}\n"
        prompt += "Answer in the same format as before.\n"
        prompt += case
        text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text


class FixLengthRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def inference(self, question, demo, case):
        assert self.query_formulation == "direct"
        text = ""
        retrieve_question = question
        while True:
            old_len = len(text)
            docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
            prompt = "".join([d["case"]+"\n" for d in demo])
            prompt += "Context:\n"
            for i, doc in enumerate(docs):
                prompt += f"[{i+1}] {doc}\n"
            prompt += "Answer in t he same format as before.\n"
            prompt += case + " " + text
            if self.method == "fix-length-retrieval":
                new_text, _, _ = self.generator.generate(prompt, self.fix_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                text = text.strip() + " " + new_text.strip()
                retrieve_question = new_text.strip()
            else:
                # fix sentence
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                new_text = new_text.strip()
                sentences = list(nlp(new_text).sents)
                sentences = [str(sent).strip() for sent in sentences]
                if len(sentences) == 0:
                    break
                text = text.strip() + " " + str(sentences[0])
                retrieve_question = sentences[0]
            
            # 判断 token 的个数要少于 generate_max_length 
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        return text


class TokenRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def modifier(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        tid = 0
        for sid, sent in enumerate(sentences):
            pos = 0
            tr = tid
            while tr < len(tokens):
                apr = sent[pos:].find(tokens[tr])
                if apr == -1:
                    break
                pos = apr + len(tokens[tr])
                tr += 1
            probs = [1 - exp(v) for v in logprobs[tid:tr+1]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)
            if p > self.hallucination_threshold: # hallucination
                # keep sentences before hallucination 
                prev = "" if sid == 0 else " ".join(sentences[:sid])
                # replace all hallucinated tokens in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                # # 这里改成了替换掉最大的那个，而不是所有的
                # max_prob = 0
                # for prob, tok in zip(probs, tokens[tid:tr+1]):
                #     max_prob = max(prob, max_prob)
                for prob, tok in zip(probs, tokens[tid:tr+1]):
                    apr = curr[pos:].find(tok) + pos
                    if prob > self.hallucination_threshold:
                    # if prob == max_prob:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(tok):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(tok)
                return prev, curr, True
            tid = tr + 1
        
        # No hallucination
        return text, None, False
    
    def inference(self, question, demo, case):
        # assert self.query_formulation == "direct"
        text = ""
        while True:
            old_len = len(text)
            prompt = "".join([d["case"]+"\n" for d in demo])
            prompt += case + " " + text
            new_text, tokens, logprobs = self.generator.generate(
                prompt, 
                self.generate_max_length, 
                return_logprobs=True
            )
            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)
            ptext, curr, hallucination = self.modifier(new_text, tokens, logprobs)
            if not hallucination:
                text = text.strip() + " " + new_text.strip()
            else:
                if self.query_formulation == "direct":
                    retrieve_question = curr.replace("[xxx]", "")
                elif self.query_formulation == "forward_all":
                    tmp_all = [question, text, ptext]
                    retrieve_question = " ".join(s for s in tmp_all if len(s) > 0)
                else:
                    raise NotImplemented

                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                prompt = "".join([d["case"]+"\n" for d in demo])
                prompt += "Context:\n"
                for i, doc in enumerate(docs):
                    prompt += f"[{i+1}] {doc}\n"
                prompt += "Answer in the same format as before.\n"
                prompt += case + " " + text + " " + ptext.strip()
                new_text, _, _ = self.generator.generate(prompt, self.generate_max_length)
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                text = text.strip() + " " + ptext.strip() + " " + new_text.strip()
            
            # 判断 token 的个数要少于 generate_max_length 
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        return text
    

class EntityRAG(TokenRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def modifier(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        entity = []
        for sent in sentences:
            doc = nlp(sent)
            li = [ent.text for ent in doc.ents]
            entity.append(li)
        
        belonging = [-1] * len(text)
        pos = 0
        for tid, tok in enumerate(tokens):
            apr = text[pos:].find(tok) + pos
            assert apr != -1
            for j in range(pos, apr+len(tok)):
                belonging[j] = tid
            pos = apr + len(tok)
        
        entity_intv = []
        for sid, sent in enumerate(sentences):
            tmp = []
            pos = text.find(sent)
            for ent in entity[sid]:
                apr = text[pos:].find(ent) + pos
                el = belonging[apr]
                er = belonging[apr + len(ent) - 1]
                tmp.append((el, er))
                pos = apr + len(ent)
            entity_intv.append(tmp)

        entity_prob = []
        for ent_itv_per_sent in entity_intv:
            tmp = []
            for itv in ent_itv_per_sent:
                probs = np.array(logprobs[itv[0]:itv[1]+1])
                p = {
                    "avg": np.mean,
                    "max": np.max,
                    "min": np.min,
                    "first": lambda x: x[0] if len(x) > 0 else 0
                }.get(self.entity_solver, lambda x: 0)(probs)
                tmp.append(p)
            entity_prob.append(tmp)

        for sid in range(len(sentences)):
            if len(entity_prob[sid]) == 0:
                continue
            probs = [1 - exp(v) for v in entity_prob[sid]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)
            if p > self.hallucination_threshold: # hallucination
                # keep sentences before hallucination 
                prev = "" if sid == 0 else " ".join(sentences[:sid])
                # replace all hallucinated entities in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                for prob, ent in zip(probs, entity[sid]):
                    apr = curr[pos:].find(ent) + pos
                    if prob > self.hallucination_threshold:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(ent):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(ent)
                return prev, curr, True
        # No hallucination
        return text, None, False

    def inference(self, question, demo, case):
        return super().inference(question, demo, case)


class AttnWeightRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
    
    def modifier(self, text, tokens, attentions, weight):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        tid = 0
        for sid, sent in enumerate(sentences):
            tl, tr = tid, tid
            if sid == len(sentences) - 1:
                tl, tr = tid, len(tokens)
            else:
                for i in range(tid + 1, len(tokens)):
                    seq = " ".join(tokens[tl:i])
                    if sent in seq:
                        tr = i
                        break
                tid = tr
            # value = attenion * (-log prob)
            attns = attentions[tl:tr]
            attns = np.array(attns) / sum(attns)
            value = [attns[i-tl] * weight[i] * (tr-tl) for i in range(tl, tr)] 
            thres = [1 if v > self.hallucination_threshold else 0 for v in value]
            if 1 in thres:
                # hallucinated
                if "check_real_words" in self.__dict__ and self.check_real_words:
                    try:
                        tags  = tagger.tag(sent.split())
                        real_words = []
                        for tag, token in zip(tags,sent.split()):
                            # print(token.strip(),tag)
                            if tag in ['noun','noun_prop','noun_num','noun_quant','adj','adj_comp','adj_num','adv','adv_interrog','adv_rel','verb','verb_pseudo','abbrev','digit','latin']:
                                real_words.append(token.strip())
                    except:
                        doc = nlp(sent)
                        real_words = set(token.text for token in doc if token.pos_ in 
                            ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
                    # real_words = set(token.text for token in doc if token.pos_ in 
                    #     ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
                    def match(tok):
                        for word in real_words:
                            if word in tok or tok in word:
                                return True
                        return False
                    for i in range(len(thres)):
                        if not match(tokens[tl+i]):
                            thres[i] = 0                
                
                prev = "" if sid == 0 else " ".join(sentences[:sid])
                # curr = " ".join(
                #     [tokens[i] if thres[i] == 0 else "[xxx]" for i in range(len(thres))]
                # )
                return True, prev, tokens[tl:tr], thres
        return False, text, None, None

    def keep_real_words(self, prev_text, curr_tokens, curr_hit):
        curr_text = " ".join(curr_tokens)
        all_text = prev_text + " " + curr_text
        input_ids = self.generator.tokenizer.encode(all_text, return_tensors="pt")
        input_ids = input_ids.to(self.generator.model.device)
        input_length = input_ids.shape[1]
        # tokens_tmp = self.generator.tokenizer.convert_ids_to_tokens(input_ids[0])
        tokens_tmp = [self.generator.tokenizer.decode([token]) for token in input_ids[0]]
        atten_tmp = self.generator.model(input_ids, output_attentions=True).attentions[-1][0]

        # merge tokens
        range_ = []
        for i, t in enumerate(tokens_tmp):
            if i == 0 or t.startswith(self.generator.space_token) or input_ids[0][i] == self.generator.tokenizer.encode("\n")[-1]:
                range_.append([i, i])
            else:
                range_[-1][-1] += 1
        tokens = []
        for r in range_:
            tokenseq = "".join(tokens_tmp[r[0]: r[1]+1]).replace(self.generator.space_token, "")
            tokens.append(tokenseq)

        # 获取幻觉词对应的 attention
        curr_st = len(tokens) - len(curr_tokens)
        atten_tmp = torch.mean(atten_tmp, dim=0)
        attns = []
        for r in range_:
            # att = torch.zeros(atten_tmp.shape[0], input_length)
            att = torch.zeros(input_length)
            for i in range(r[0], r[1] + 1):
                if i == 0:
                    continue
                v = atten_tmp[i-1][:r[0]] # 上一位的
                v = v / v.sum()
                t = torch.zeros(input_length)
                t[:r[0]] = v
                att += t
            att /= (r[1] - r[0] + 1)
            # merge token for att
            att = torch.tensor([att[rr[0]:rr[1]+1].sum() for rr in range_])
            attns.append(att)
            
        # 计算每个超过阈值的 token 在前文的 attentions
        forward_attns = torch.zeros(len(tokens))
        hit_cnt = 0
        for i in range(len(curr_hit)):
            if curr_hit[i] == 1:
                forward_attns += attns[curr_st + i] #NOTE: attns is 2d so we do elment wise addition for any hallucinated token
                hit_cnt += 1
        forward_attns /= hit_cnt
        forward_attns = forward_attns.tolist()

        # 分析词性，保留实词对应的 attns
        # doc = nlp(all_text)
        # real_words = set(token.text for token in doc if token.pos_ in 
        #               ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
        try:
            tags  = tagger.tag(all_text.split())
            real_words = []
            for tag, token in zip(tags,all_text.split()):
                # print(token.strip(),tag)
                if tag in ['noun','noun_prop','noun_num','noun_quant','adj','adj_comp','adj_num','adv','adv_interrog','adv_rel','verb','verb_pseudo','abbrev','digit','latin']:
                    real_words.append(token.strip())
        except:
            doc = nlp(all_text)
            real_words = set(token.text for token in doc if token.pos_ in 
                      ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
        
        def match(token):
            for word in real_words:
                if word in token or token in word:
                    return True
            return False
        
        real_pairs = []
        for i in range(len(tokens)):
            tok, att = tokens[i], forward_attns[i]
            if i >= curr_st and curr_hit[i - curr_st]: #NOTE: skip hallucinated tokens from all tokens --> (if we have 50 and curr_tokens are 5 then we will begin to evaluate this expression starting from i = 45)
                continue
            if match(tok):
                real_pairs.append((att, tok, i))
        
        if "retrieve_keep_top_k" in self.__dict__:
            top_k = min(self.retrieve_keep_top_k, len(real_pairs))
        elif "retrieve_keep_ratio" in self.__dict__:
            top_k = int(len(real_pairs) * self.retrieve_keep_ratio)
        
        real_pairs = sorted(real_pairs, key = lambda x:x[0], reverse=True)
        real_pairs = real_pairs[:top_k]
        real_pairs = sorted(real_pairs, key = lambda x:x[2])
        return " ".join([x[1] for x in real_pairs])
        
    def inference(self, question, demo, case):
        # assert self.query_formulation == "direct"
        # print(question)
        def fetch_last_n_tokens(text, num, tokenizer = self.generator.tokenizer):
            tokens = tokenizer.tokenize(text)
            if num >= len(tokens):
                return text
            last_n_tokens = tokens[-num:]
            last_n_sentence = ' '.join(last_n_tokens)
            return last_n_sentence

        # print("#" * 20)
        text = [""]*len(question)
        flag_halucination = [False]*len(question)
        max_trials = 3
        for trial in range(max_trials):
            # print("Trial",trial)
            # print(demo)

            #NOTE: prompt is the case in first iteration  (case = user instruction)
            start_time = time.time()
            prompts = []
            old_len = []
            
            for sample_idx in range(len(question)):
                old_len.append(text[sample_idx])
                if flag_halucination[sample_idx]:
                    prompts.append("hi")
                else:
                    prompt = "".join([d["case"]+"\n" for d in demo[sample_idx]])
                    tmp_li = [case[sample_idx], text[sample_idx]]
                    prompt += " ".join(s for s in tmp_li if len(s) > 0)
                    prompts.append(prompt)
            # print("First generation prompts",time.time()-start_time)
            # print('####', prompt)
            # prompt += case + " " + text
            #NOTE: gent attn need to be modified (done)
            start_time = time.time()
            new_text, tokens, attns, logprobs, entropies = self.generator.generate_attn(
                prompts, 
                self.generate_max_length, 
                # self.attention_solver, 
                use_entropy = self.method == "dragin", 
                use_logprob = self.method == "attn_prob"
            )
            # print("First generation time",time.time()-start_time)

            prompts = []
            ptexts = []
            start_time = time.time()
            for sample_idx in range(len(question)):
                if flag_halucination[sample_idx]:
                    prompts.append("hi")
                    ptexts.append("")
                    continue
                weight = entropies[sample_idx] if self.method == "dragin" else [-v for v in logprobs[sample_idx]]

                if self.use_counter == True:
                    self.counter.add_generate(new_text[sample_idx], self.generator.tokenizer)
                hallucination, ptext, curr_tokens, curr_hit =  self.modifier(new_text[sample_idx], tokens[sample_idx], attns[sample_idx], weight)
            
                if not hallucination:
                    text[sample_idx] = text[sample_idx].strip() + " " + new_text[sample_idx].strip()
                    prompts.append("hi")
                    ptexts.append("")

                else:
                    temp_hallucination = []
                    for tok,hit in zip(curr_tokens,curr_hit):
                        if hit == 1:
                            temp_hallucination.append(tok)
                    # print("\n Trial",trial,"-->", temp_hallucination)
                    
                    forward_all = [question[sample_idx], text[sample_idx], ptext]
                    forward_all = " ".join(s for s in forward_all if len(s) > 0)


                    if self.query_formulation == "current":
                        retrieve_question = " ".join(curr_tokens)

                    elif self.query_formulation == "current_wo_wrong":
                        retrieve_question = " ".join(
                            list(curr_tokens[i] if curr_hit[i] == 0 else "" for i in range(len(curr_tokens)))
                        )

                    elif self.query_formulation == "forward_all":
                        retrieve_question = forward_all
                    
                    elif self.query_formulation == "last_sentence":
                        retrieve_question = self.get_last_sentence(forward_all)
                    
                    elif self.query_formulation == "last_n_tokens":
                        assert "retrieve_keep_top_k" in self.__dict__
                        retrieve_question = fetch_last_n_tokens(
                            forward_all, self.retrieve_keep_top_k)
                    
                    elif self.query_formulation == "real_words":  #NOTE: we use this
                        # retrieve_question = self.keep_real_words(
                        #     prev_text = question[sample_idx] + " " + text[sample_idx] + " " + ptext, 
                        #     curr_tokens = curr_tokens, 
                        #     curr_hit = curr_hit,
                        # ) 
                        retrieve_question = question[sample_idx]
                    else:
                        raise NotImplemented
                    
                    # print("\n Query:",retrieve_question)
                    docs = self.retrieve(retrieve_question, topk=self.retrieve_topk) # need a better strategy to format question
                    # print("Retrieved docs:\n",docs)
                    # print("---",flush=True)
                    prompt = "".join([d["case"]+"\n" for d in demo[sample_idx]])
                    prompt += "Given the following context:\n" # RE-order and use tags
                    for i, doc in enumerate(docs):
                        prompt += f"[{i+1}] {doc}\n"
                    # prompt += "Answer in the same format as before.\n" # Answer based on the previous context.
                    tmp_li = [case[sample_idx], text[sample_idx], ptext.strip()]
                    prompt += " ".join(s for s in tmp_li if len(s) > 0)
                    prompts.append(prompt)
                    ptexts.append(ptext)
                    # print('#####', prompt)
                    # prompt += case + " " + text + " " + ptext.strip()
            # print("Second prompts ",time.time()-start_time)
            start_time = time.time()
            new_text, _, _ = self.generator.generate(prompts, self.generate_max_length) #NOTE: need modification
            # print("Second generation time",time.time()-start_time)

            start_time = time.time()
            for sample_idx in range(len(question)):
                if prompts[sample_idx]=="hi" or flag_halucination[sample_idx]: continue
                if self.use_counter == True:
                    self.counter.add_generate(new_text[sample_idx], self.generator.tokenizer)
                    self.counter.hallucinated += 1
                new_text[sample_idx] = self.get_top_sentence(new_text[sample_idx])
                tmp_li = [text[sample_idx].strip(), ptexts[sample_idx].strip(), new_text[sample_idx].strip()]
                text[sample_idx] = " ".join(s for s in tmp_li if len(s) > 0)
            # print("Modify out",time.time()-start_time)
            # text = text.strip() + " " + ptext.strip() + " " + new_text.strip()


            
            # 判断 token 的个数要少于 generate_max_length
            stop_condition = True
            start_time = time.time()
            for sample_idx in range(len(question)):
                tokens_count = len(self.generator.tokenizer.encode(text[sample_idx]))
                # print("\nTrial:",trial,"Answer\n",text)
                # if tokens_count > self.generate_max_length or len(text) <= old_len or "</answer>" in text or trial==max_trials-1:
                if tokens_count > self.generate_max_length or "</answer>" in text[sample_idx] or trial==max_trials-1:
                    
                    # print("*"*10,"Stop at trial",trial,"*"*10)
                    stop_condition &= True
                    flag_halucination[sample_idx] = True
                else:
                    stop_condition &= False

                # print("#" * 20)
            # print("Stop condition",time.time()-start_time)
            if stop_condition:
                break
            # print("----"*5)
        return text
