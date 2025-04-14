from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, shutil, re, random, io, requests, ctypes, sys, time, struct
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
import wandb,traceback
import pdb
from config import train_config, ds_config
from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list
from transformers import GenerationConfig

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ["NCCL_P2P_DISABLE"]="1"

wandb_name = train_config['wandb_name']
wandb_project = train_config['wandb_project']
wandb_key = train_config['wandb_key']
model_path = train_config['model_path']
save_path = train_config['save_path']
record_path = train_config['record_path']
gen_data_path = train_config['gen_data_path']
gen_device = train_config['gen_device']   
all_steps = train_config['all_steps']
Q_batch_size = train_config['Q_batch_size']
num_pre_Q = train_config['num_pre_Q']
train_batch_size = train_config['train_batch_size']
gen_update_steps = train_config['gen_update_steps']
save_steps = train_config['save_steps']
compute_gen_logps = train_config['compute_gen_logps']
clip_param = train_config['clip_param']
ref_server = train_config['ref_server']
beta = train_config['beta']
program_path = train_config['gen_data_path']
global update_model_num
update_model_num = 0

# generation_config = GenerationConfig(
#             max_new_tokens=600,
#             temperature=0)

# from math_verify import parse, verify, ExprExtractionConfig
# def reward_correct(grt, answer):
#     pattern = r'\d+\.\d+|\d+/\d+|\d+'  # 正则表达式匹配整数、小数和分数
#     nums = re.findall(pattern, answer)  # 使用正则表达式在answer中查找所有数字
#     if len(nums) == 0:
#         return 0  # 没有找到数字，返回负奖励
#     lastnum = nums[-1]  # 取answer中最后一个数字
#     ans = parse("\\boxed{" + lastnum + "}")
#     ground_truth = parse("\\boxed{" + str(grt) + "}")
#     return 1 if verify(ans, ground_truth) else 0  # 如果匹配，返回奖励1，否则返回奖励0

def get_batch():
    try:
        r = requests.get(f"{ref_server}/get").content
        if r == b'empty': return None
    except: return None
    dd = bytes_list_to_list(r)
    data = json.loads(dd[0]) 
    data['inputs'] = bytes_to_tensor(dd[1])
    data['rewards'] = bytes_to_tensor(dd[2])
    data['refs'] = bytes_to_tensor(dd[3])
    data['gen_logps'] = bytes_to_tensor(dd[4])
    data['acc_scores'] = bytes_to_tensor(dd[5])
    data['format_scores'] = bytes_to_tensor(dd[6])

    # print("!!!batch的key", data.keys())
    return data

def get_per_token_logps(logits, input_ids):
    per_token_logps = [] # Use a loop to reduce memory peak.
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)


def GRPO_step(batch):
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(engine.device)
    print(f"!!! rank: {torch.distributed.get_rank()} inputs shape: {inputs.shape} ")
    advantages = batch['rewards'].to(engine.device).unsqueeze(1)
    logits = engine(inputs).logits
    # print(f"!!! rank: {torch.distributed.get_rank()} cal the logits successfully!!")

    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = inputs[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it 
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:,prompt_length-1:]
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()
    if 'gen_logps' in batch:
        ratio = torch.exp(per_token_logps - batch['gen_logps'].to(engine.device))
        clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    else: 
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        assert compute_gen_logps is False
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    # print(f"!!! rank: {torch.distributed.get_rank()} cal the loss successfully!!")
    return loss

import signal
import time

def handler(signum, frame):
    raise TimeoutError("Code execution timed out")

def gen_worker(Q, physics_device):
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{physics_device}'
    torch.cuda.set_device(0)
    print(f"Generation worker process uses GPU {physics_device}")
    from vllm import LLM, SamplingParams
    vllm_gen = LLM(model=model_path, gpu_memory_utilization=0.5)
    ref_server_ver = 'tensor'  # don't worry, it will auto switch based on the first upload

    #sampling_params = SamplingParams(n=num_pre_Q, temperature=0.9, max_tokens=600)
    gen_logps_sp = SamplingParams(temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1)

    # data_path = "/mnt/remote-data/hjy/public_code/GRPO/data/gsm8k_train.json"
    data_path = train_config['data_path']
    with open(data_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    # QAs = [{'Q': item['question'], 'A': item['answer_detail']} for item in dataset] 
    QAs = [{'Q': item['instruction'], 'A': item['output']} for item in dataset] 
    with open("./system_prompt_refine.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()
  
    # stop_sentences = "<refine>"
    # process_refine_sentences = "Well, let me try to refine the previous generation answer!\n"
    # sampling_params_stop = SamplingParams(n=1, temperature=0.9, max_tokens=800, stop=stop_sentences, include_stop_str_in_output=True)
    sampling_params_stop = SamplingParams(n= num_pre_Q, temperature=0.9, max_tokens=1500)
    # fout_code = open(f'{program_path}', 'w')
    
    # def get_completions(prompts, num):
    #     outputs = vllm_gen.generate(prompts, sampling_params_stop, use_tqdm=False)
    #     responses = [output.outputs[0].text for output in outputs]
    #     if num > 5:
    #         return responses
    #     recursive_ids = []
    #     for i, c in enumerate(responses):
    #         insert_flag = np.random.choice([0, 1])
    #         if c.endswith(tuple(stop_sentences)) and insert_flag==1:
    #             recursive_ids.append(i)
    #             #    pdb.set_trace()
    #             fout_code.write(f"\n===============before function:{c}")
    #     if len(recursive_ids)>0:
    #         recursive_prompts = []
    #         for i in recursive_ids:
    #             responses[i] = responses[i] + process_refine_sentences
    #             recursive_prompts.append(prompts[i] + responses[i])
    #         rec_resps = get_completions(recursive_prompts, num+1)
    #         for org_id, rec_resp in zip(recursive_ids, rec_resps):
    #             responses[org_id] += rec_resp
    #     return responses
    
    def gen_answers(prompts):
        tip_text = []
        for x in prompts:
            # for _ in range(num_pre_Q):
            tip_text.append(tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True))
        # answers = get_completions(tip_text,0)
        voutputs = vllm_gen.generate(tip_text, sampling_params = sampling_params_stop, use_tqdm=False)
        answers = []
        for v in voutputs:
            for z in v.outputs:
                answers.append(z.text)

        return answers

    # from math_verify import parse, verify, ExprExtractionConfig, LatexExtractionConfig

    # def reward_correct(ground_truth, answer):
    #     pattern = r'\\boxed{([^{}]*(?:\{[^{}]*\}[^{}]*)*)}'
    #     boxed_matches_ans =  re.findall(pattern, answer)
    #     boxed_matches_ground_truth =  re.findall(pattern, ground_truth)
    #     if not boxed_matches_ans or not boxed_matches_ground_truth:
    #         return -1.0
    #     boxed_matches_ans = "\\boxed{" + boxed_matches_ans[-1] + "}"
    #     boxed_matches_ground_truth =  "\\boxed{" + boxed_matches_ground_truth[-1] + "}"
    #     if not boxed_matches_ans:
    #         return -1.0  
    #     if boxed_matches_ans == ground_truth:
    #         return 1.0
    #     ans = parse(boxed_matches_ans)
    #     # print(ans)
    #     ground_truth = parse(boxed_matches_ground_truth)
    #     return 1.0 if verify(ans, ground_truth) else -1.0
    from openai import OpenAI
    def reward_correct(question, ground_truth, answer):
        ###使用卡或者api来判断生成的答案是否正确
        client = OpenAI(base_url="http://localhost:8099/v1", api_key="dummy")
        sys_prompt = '''
        You are a judger, you will judge the correctness of the answer to the question.
        Below is a question, a ground truth answer, and an answer generated by an AI assistant, 
        please rate the AI assistant's answers according to the question on a scale from 0 to 1.
        Your output is just a number in the range from 0 to 1.
        '''
        usr_prompt = '''
        ### Question:
        {Question}

        ### Ground Truth:
        {Ground_Truth}

        ### Answer:
        {Answer}
        '''
        usr_prompt = usr_prompt.format(Question = question, Ground_Truth = ground_truth, Answer = answer)

        response = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt}],
        temperature=0.2,
        )
        res = response.choices[0].message.content
        try: 
            reward = float(res)

        except:
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", res)
            if numbers:
            # 取最后一个数字并转换
                last_number = numbers[-1]
                reward = float(last_number)
            else:
            # 如果没有找到任何数字，返回0
                reward = 0.0
        finally:
        # 确保奖励在0-1之间
            reward = max(0.0, min(reward, 1.0))

        return reward
    
    # def reward_format(item, answer):
    # #     pattern = r"^<think>.*?</think>[\n ]<answer>.*?</answer>$"
    #     pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    #     return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else -1
    def reward_format(answer):
        pattern = r"^<think>.*?</think>[\n ]*?<answer>.*?</answer>$"
        think_count = answer.count("<think>") + answer.count("</think>")
        answer_count = answer.count("<answer>") + answer.count("</answer>")
        reward = 1.0 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) and think_count == 2 and answer_count == 2 else 0.0
        return reward

    def reward_refine(prev_answer, refined_answer, question):
        refine_cnt = refined_answer.count("<refine>")
        if refine_cnt > 0: 
            prev_score = reward_correct(question, prev_answer)
            new_score = reward_correct(question, refined_answer)
            if new_score > prev_score:
                return 0.5
            elif new_score < prev_score:
                return -0.5
            else:
                return 0.0
        else:
            return 0.0

    
    # def call_python(answer):
    #     error_cnt = answer.count("Error!")
    #     python_cnt = answer.count("```python")
    #     return (python_cnt - error_cnt) * 0.1

    def gen_samples(inputs):
        prompts = [x["Q"] for x in inputs]
        answers = gen_answers(prompts)
        rewards = []
        scores = []
        record_gen= []
        acc_scores= []
        format_scores =[]
    
        for i, inp in enumerate(inputs):
            pre_Q_correct_acc = 0
            pre_Q_correct_format = 0
            for a in answers[i*num_pre_Q:(i+1)*num_pre_Q]:
                acc_score = reward_correct(inp['Q'], inp['A'], a)
                format_score = reward_format(a)
                refine_score = reward_refine(a)
                # call_python_score = call_python(a)
                acc_scores.append(acc_score)
                format_scores.append(format_score)
                # the acc score will be bigger after 300 steps
                # if update_model_num >= 16:
                #     reward_all = 
                #     # rewards.append(2 * acc_score + format_score + call_python_score)
                #     rewards.append(1.5 * acc_score + format_score)

                # #at the begining, the format score is bigger
                # else:
                #     rewards.append(acc_score + 1.5 * format_score)
                reward_all = acc_score + format_score + refine_score
                rewards.append(reward_all)
                record_gen.append({"question": inp, "answer": a, "acc_score":acc_score, "format_score": format_score, "refine_score":refine_score, "reward_all":reward_all})
                if acc_score>0: pre_Q_correct_acc += 1
                if format_score>0: pre_Q_correct_format +=1
            scores.append((pre_Q_correct_acc, pre_Q_correct_format))
        
        #record the generation data the score
        if os.path.exists(gen_data_path) and os.path.getsize(gen_data_path) > 0:
            with open(gen_data_path, 'r') as f:
                try:
                    gen_data = json.load(f)
                except json.JSONDecodeError:
                    gen_data = [] 
        else:
            gen_data = []  
        gen_data.extend(record_gen)
        with open(gen_data_path, 'w') as file:
            json.dump(gen_data, file, indent=4)

        prompts_text = [tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in prompts]
        return prompts_text, torch.tensor(rewards, dtype=torch.float32), answers, torch.tensor(acc_scores, dtype=torch.float32), torch.tensor(format_scores, dtype=torch.float32), torch.tensor(scores, dtype=torch.float32)

    def try_update_model():
        try:
            new_state_dict = Q.get_nowait()
            print('[VLLM PROC] recving new model ...')
            llm_model = vllm_gen.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(new_state_dict.items())
            print('[VLLM PROC] model updated')
            del new_state_dict
        except:
            #print('[VLLM PROC] no new model')
            return
        
    from torch.nn.utils.rnn import pad_sequence

    fout = open(f'{record_path}', 'w')
    for it in range(999999999):
       #按照顺序的方式来训练模型
        # if it==0:
        #     start = 0
        # else:
        #     start = 1000

        start = 0
        for j in range(start,len(QAs), Q_batch_size):
            inputs = QAs[j:j+Q_batch_size]
            if j % 2 == 0: 
                try_update_model()
            # inputs = random.sample(QAs, Q_batch_size)
            tic = time.time()
            prompt_inputs, rewards, answers, acc_scores, format_scores, scores = gen_samples(inputs)
            # print(f'time: {time.time()-tic:.2f}s    ', 'scores:', scores)
            fout.write(str(scores) + '\n')

            if it % 5 == 0: 
                fout.write(str(inputs[0])+"\n"+str(answers[0]) + '\n\n')
                fout.flush()
                # print('answers:', answers[0])
            ans_token_ids = tokenizer(answers, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)['input_ids']
            for i, pp in enumerate(prompt_inputs):
                prompt_ids = tokenizer(pp, return_tensors="pt", add_special_tokens=False)["input_ids"]
                plen = prompt_ids.shape[1]
                curr_answers = answers[i*num_pre_Q:(i+1)*num_pre_Q]
                curr_ans_ids = ans_token_ids[i*num_pre_Q:(i+1)*num_pre_Q]
                curr_rewards = rewards[i*num_pre_Q:(i+1)*num_pre_Q]
                curr_acc_scores = acc_scores[i*num_pre_Q:(i+1)*num_pre_Q]
                curr_format_scores = format_scores[i*num_pre_Q:(i+1)*num_pre_Q]
                # pdb.set_trace()
                if curr_rewards.max() - curr_rewards.min() < 1e-4: continue

                if ref_server_ver == 'tensor':
                    curr_rewards = (curr_rewards - curr_rewards.mean()) / (curr_rewards.std() + 1e-4)
                    for ii in range(0, num_pre_Q, train_batch_size):
                        sub_rewards = curr_rewards[ii:ii+train_batch_size]
                        sub_ans_ids = curr_ans_ids[ii:ii+train_batch_size]
                        sub_acc_scores = curr_acc_scores[ii:ii+train_batch_size]
                        sub_format_scores = curr_format_scores[ii:ii+train_batch_size]

                        tensor_list = [torch.tensor(lst) for lst in sub_ans_ids]
                        output_ids = pad_sequence(tensor_list, batch_first=True, padding_value=tokenizer.pad_token_id) 
                        Qrep = prompt_ids.repeat(1, output_ids.shape[0]).view(-1, plen)
                        merged_ids = torch.cat([Qrep, output_ids], dim=1)
                        data = [json.dumps({"plen": plen}).encode(), tensor_to_bytes(merged_ids), tensor_to_bytes(sub_rewards)]       

                        if compute_gen_logps:
                            zz = vllm_gen.generate(prompt_token_ids=merged_ids.tolist(), sampling_params=gen_logps_sp, use_tqdm=False)
                            zz = [ xx.prompt_logprobs[plen:] if xx.prompt_logprobs is not None else [] for xx in zz]
                            # zz = [xx.prompt_logprobs[plen:] for xx in zz]
                            if not zz:
                                print("[!!! SPEICIAL CASE]")
                                continue
                            gen_logps = torch.tensor([[list(x.values())[0].logprob for x in xx] for xx in zz])
                            data.append(tensor_to_bytes(gen_logps))
                        
                        data.append(tensor_to_bytes(sub_acc_scores))
                        data.append(tensor_to_bytes(sub_format_scores))
                        # print("!!data length:", len(data))
                        xdata = make_bytes_list(data)
                        # print("!!start to upload")
                        r = requests.post(f"{ref_server}/upload", data=xdata)
                        if r.content == b'string': ref_server_ver = 'string'
                elif ref_server_ver == 'string':
                    xdata = make_bytes_list([json.dumps({"Q": pp[0], "As": curr_answers}).encode(), 
                                            tensor_to_bytes(curr_rewards)])
                    r = requests.post(f"{ref_server}/upload", data=xdata)
                    if r.content == b'tensor': ref_server_ver = 'tensor'


tokenizer = AutoTokenizer.from_pretrained(model_path)
if __name__ == '__main__':
    import deepspeed
    deepspeed.init_distributed()
    if dist.get_rank() == 0:
        print('\nSTART vLLM generation...\n')
        mp.set_start_method('spawn')
        Q = mp.Queue()
        p = mp.Process(target=gen_worker, args=(Q, gen_device))
        p.start()

    model = AutoModelForCausalLM.from_pretrained(model_path, 
            torch_dtype=torch.bfloat16, _attn_implementation="sdpa")

    engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, 
                                                model_parameters=model.parameters())

    progress = range(1, all_steps+1)
    if dist.get_rank() == 0: 
        progress = tqdm(progress)

    total_output_length = 0
    total_acc_correct = 0
    total_format_correct = 0
    total_num = 0

    wandb.login(key=wandb_key)
    wandb.init(project=wandb_project, name=wandb_name)
    
    for step in progress:
        batch = get_batch()
        while batch is None:
            time.sleep(1)
            batch = get_batch()

        # if batch['inputs'].shape[1]>2200:
        #     continue
        if torch.distributed.get_rank() == 0:
            batch_length = (batch['gen_logps'].shape[0] * batch['gen_logps'].shape[1])
            total_output_length += batch_length

            total_acc_correct += ( batch['acc_scores'] > 0).sum().item()
            total_format_correct += ( batch['format_scores'] > 0).sum().item()

            total_num += batch['inputs'].shape[0]
            wandb.log({"avg_output_token_lenght": float(total_output_length / total_num),
                        "acc_correct_ratio": float(total_acc_correct / total_num),
                        "format_correct_ratio": float(total_format_correct / total_num),
                     })
        loss = GRPO_step(batch)
        engine.backward(loss)
        # print(f"!!!!rank:{torch.distributed.get_rank()} backward successfully ")
        engine.step()

        if dist.get_rank() == 0:
            progress.set_description(f"Loss: {loss.item():.6f}")

        if step % gen_update_steps == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print('[TRAINING PROC] sending latest state_dict ...')
                state_dict = engine.module.state_dict()
                Q.put(state_dict)
                print('[TRAINING PROC] send state_dict ok!')
                update_model_num += 1
                print('!!The number of update the genmodel:',update_model_num)
            dist.barrier()

        if step % save_steps == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print('saving model')
                save_name = f"{save_path}/step_{step}"
                state_dict = engine.module.state_dict()
                state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
                engine.module.save_pretrained(save_name, state_dict=state_dict)
                tokenizer.save_pretrained(save_name)
            dist.barrier()