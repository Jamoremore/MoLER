import os
import numpy as np
import requests
import json
import time
import logging
import re
from typing import List, Dict, Tuple, Optional
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from beir.datasets.data_loader import GenericDataLoader
from prompt import *
from transformers import AutoTokenizer


NUM_RUNS = 3  
MAX_RETRIES = 2 
MAX_WORKERS = 20  
DEFAULT_TIMEOUT = 300  

enable_thinking=False

class LLMEvaluator:
	def __init__(self, model_name: str, api_key: str, base_url: str):
		os.environ.pop("http_proxy", None)
		os.environ.pop("https_proxy", None)
		os.environ.pop("all_proxy", None)
		
		datapath="/mnt/data/tenant-home_speed/Model/Qwen3/Qwen3-0.6B"
		self.tokenizer = AutoTokenizer.from_pretrained(datapath,padding_side="left")
		self.model_name = model_name
		self.client = OpenAI(api_key=api_key, base_url=base_url)
	
	def get_model_response(self, prompt: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[str]:
		try:
			if enable_thinking==False:
				# print("nonthink")
				response = self.client.chat.completions.create(
					model=self.model_name,
					messages=[{"role": "user", "content": prompt}],
					temperature=0.7,
					top_p=0.8,
					timeout=timeout,
					extra_body={"chat_template_kwargs": {"enable_thinking": False,"top_k":20}}
				)
			else:
				# print("think")
				response = self.client.chat.completions.create(
					model=self.model_name,
					messages=[{"role": "user", "content": prompt}],
					temperature=0.6,
					top_p=0.95,
					timeout=timeout,
					extra_body={"chat_template_kwargs": {"enable_thinking": True,"top_k":20}}
				)
			if response is None:
				raise ValueError("API returned None response")
			
			if not response.choices:
				raise ValueError("API response contains no choices")
			
			if not response.choices[0].message or not response.choices[0].message.content:
				raise ValueError("API response has empty content")
			
			content = response.choices[0].message.content
			return content.strip() if content else None
		except Exception as e:
			logging.warning(f"API request failed: {e}")
			raise
	
	def extract_subqueries(self, prompt_str,num_query):
		subquerys = []
		identifiers = ["Sub-query {}".format(i) for i in range(1,num_query+1)]

		for i in range(num_query):
			start_idx = prompt_str.find(identifiers[i])
			if start_idx == -1:
				subquerys.append("")
				continue
			
			content_start = start_idx + len(identifiers[i])
			
			if i < num_query-1:
				end_idx = prompt_str.find(identifiers[i+1], content_start)
			else:
				end_idx = len(prompt_str)

			subquery = prompt_str[content_start:end_idx].strip()
			if subquery[0]==":" or subquery[0]=="：":
				subquery=subquery[1:].strip()
			subquerys.append(subquery)
	
		return subquerys

	def process_single_query_mmlf(self, index: int, query: str, num_query: int) -> Tuple[int, str]:
		for attempt in range(MAX_RETRIES):
			try:
				# 2 steps RAG
				example = ""
				for i in range(1, num_query+1):
					example += "Sub-query {}:\n".format(i)

				query1 = MQR_PROMPT.format(cnt=num_query, query=query, example=example)
				res = self.get_model_response(query1)
				token = self.calculate_qwen_token_length(res)
				if res:
					if '</think>' in res:
						res = res.split('</think>')[-1].strip()
					subquerys = self.extract_subqueries(res,num_query)
					
					ans = []

					for i in subquerys:
						query2 = CQE_PROMPT_1.format(original_query=query, sub_query=i)
						res_i = self.get_model_response(query2)
						token += self.calculate_qwen_token_length(res_i)
						if '</think>' in res_i:
							res_i = res_i.split('</think>')[-1].strip()
						res_i = res_i.split('</think>')[-1].strip()
						ans.append(res_i)
					return index, ans, token
			except Exception as e:
				logging.error(f"Attempt {attempt + 1} failed for index {index}: {e}")
			time.sleep(2 ** attempt)
		logging.warning(f"Max retries reached for index {index}. Skipping...")
		return index, ''

	def process_single_query_mqr(self, index: int, query: str, num_query: int) -> Tuple[int, str]:
		for attempt in range(MAX_RETRIES):
			try:
				# 2 steps RAG
				example = ""
				for i in range(1, num_query+1):
					example += "Sub-query {}:\n".format(i)

				query1 = MQR_PROMPT.format(cnt=num_query, query=query, example=example)
				res = self.get_model_response(query1)
				token = self.calculate_qwen_token_length(res)
				if res:
					if '</think>' in res:
						res = res.split('</think>')[-1].strip()
					subquerys = self.extract_subqueries(res,num_query)
					return index, subquerys, token
			except Exception as e:
				logging.error(f"Attempt {attempt + 1} failed for index {index}: {e}")
			time.sleep(2 ** attempt)
		logging.warning(f"Max retries reached for index {index}. Skipping...")
		return index, ''
	
	def process_single_query_cqe(self, index: int, query: str, num_query: int) -> Tuple[int, str, int]:
		for attempt in range(MAX_RETRIES):
			try:
				query1 = CQE_PROMPT_ONLY.format(original_query=query)
				res = self.get_model_response(query1)
				token = self.calculate_qwen_token_length(res)
				if res:
					if '</think>' in res:
						res = res.split('</think>')[-1]
					res = res.split('</think>')[-1].strip()
					token += self.calculate_qwen_token_length(res)
					return index, res, token
			except Exception as e:
				logging.error(f"Attempt {attempt + 1} failed for index {index}: {e}")
			time.sleep(2 ** attempt)
		logging.warning(f"Max retries reached for index {index}. Skipping...")
		return index, '', -1
	
	
	def process_single_query_mslf(self, index: int, query: str, num_query: int) -> Tuple[int, str, int]:
		for attempt in range(MAX_RETRIES):
			try:
				# 2 steps RAG
				example = ""
				for i in range(1, num_query+1):
					example += "Sub-query {}:\n".format(i)

				query1 = MQR_PROMPT.format(cnt=num_query, query=query, example=example)
				res = self.get_model_response(query1)
				token = self.calculate_qwen_token_length(res)
				if res:
					if '</think>' in res:
						res = res.split('</think>')[-1]
					query2 = CQE_PROMPT.format(original_query=query, sub_query=res)
					res = self.get_model_response(query2)
					if res:
						token += self.calculate_qwen_token_length(res)
						res = res.split('</think>')[-1].strip()
						return index, res, token
			except Exception as e:
				logging.error(f"Attempt {attempt + 1} failed for index {index}: {e}")
			time.sleep(2 ** attempt)
		logging.warning(f"Max retries reached for index {index}. Skipping...")
		return index, '', -1
	
	def concurrent_query_processing(self, query_data: List[str], test_type, num_query):
		output_dict = {}
		
		with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
			if test_type == "MMLF":
				futures = {
					executor.submit(self.process_single_query_mmlf, index, query, num_query): index
					for index, query in enumerate(query_data)
				}
			elif test_type == "MSLF":
				futures = {
					executor.submit(self.process_single_query_mslf, index, query, num_query): index
					for index, query in enumerate(query_data)
				}
			elif test_type == "MQR":
				futures = {
					executor.submit(self.process_single_query_mqr, index, query, num_query): index
					for index, query in enumerate(query_data)
				}
			elif test_type == "CQE":
				futures = {
					executor.submit(self.process_single_query_cqe, index, query, num_query): index
					for index, query in enumerate(query_data)
				}
			
			for future in tqdm(as_completed(futures), total=len(futures), desc="Talking"):
				index = futures[future]
				index, result, token = future.result()

				output_dict[index] = {}
				output_dict[index]["result"] = result
				output_dict[index]["token"] = token

		return output_dict
	
	def calculate_qwen_token_length(self, text: str) -> int:
		try:
			inputs = self.tokenizer(text, return_tensors="pt")
			return inputs.input_ids.shape[1]

		except Exception as e:
			print(f"计算token长度时出错: {str(e)}")
			raise 

	def process_single_answer(self, index: int, query, test_type,data_name):
		for attempt in range(MAX_RETRIES):
			try:
				res={}
				if data_name=="nfcorpus":
					n1=1000
					n2=10
				elif data_name=="scifact":
					n1=10
					n2=10
				if test_type == "MMLF" or test_type == "MQR":
					response_1 = requests.post(
						query["url"],
						json={"n": n1, "multi_query": query["answer"], "query_id": query["query_id"]}
					)
					response_2 = requests.post(
						query["url"],
						json={"n": n2, "multi_query": query["answer"], "query_id": query["query_id"]}
					)
				elif test_type == "MSLF" or test_type == "CQE":
					response_1 = requests.post(
						query["url"],
						json={"n": n1, "query": query["answer"], "query_id": query["query_id"]}
					)
					response_2 = requests.post(
						query["url"],
						json={"n": n2, "query": query["answer"], "query_id": query["query_id"]}
					)
				res["recall"] = response_1.json()["Recall@"+str(n1)]
				res["nDCG"] = response_2.json()["NDCG@"+str(n2)]
				res["token"] = query["token"]
				if res:
					return index, res
			except Exception as e:
				logging.error(f"Attempt {attempt + 1} failed for index {index}: {e}")
			time.sleep(2 ** attempt)
		logging.warning(f"Max retries reached for index {index}. Skipping...")
		return index, ''

	def concurrent_answer_evaluating(self, query_data, test_type, data_name):
		recalls = {}
		nDCGs = {}
		token = {}
		output_dict = {}

		with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
			futures = {
				executor.submit(self.process_single_answer, index, query, test_type, data_name): index
				for index, query in enumerate(query_data)
			}
			
			for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
				index = futures[future]
				index, result = future.result()
				output_dict[index]=result

		recalls["results"]=[output_dict[i]["recall"] for i in output_dict]
		nDCGs["results"] = [output_dict[i]["nDCG"] for i in output_dict]
		token["results"] = [output_dict[i]["token"] for i in output_dict]			
		recalls["average"] = np.mean(recalls["results"])
		nDCGs["average"] = np.mean(nDCGs["results"])
		token["average"] = np.mean(token["results"])
		return recalls, nDCGs, token
	
	@staticmethod
	def evaluate_prediction(correct: str, predicted: str) -> Tuple[str, str, bool]:
		correct_set = LLMEvaluator.normalize_answer(correct)
		predicted_set = LLMEvaluator.normalize_answer(predicted)
		is_correct = correct_set == predicted_set
		return correct_set, predicted_set, is_correct
	
	def evaluate_dataset(self, dataset_paths, test_type, num_runs: int = NUM_RUNS,num_query=3):
		print("test_type is "+test_type+";num runs is "+str(num_runs))
		for i, data_path in enumerate(dataset_paths):
			name_without_extension = os.path.splitext(os.path.basename(data_path["PATH"]))[0]
			corpus, queries, qrels = GenericDataLoader(data_path["PATH"]).load(split='test')
			dataset_results = []
			mean_recalls = []
			mean_nDCGs = []
			mean_token = []
			prompt_list = [queries[i] for i in queries]
			queries_id = [i for i in queries]
			for run in range(num_runs):
				res_dict = self.concurrent_query_processing(prompt_list, test_type,num_query)
				query_data = \
					[{"url":data_path["url"], "answer":res_dict[j]["result"], "query_id":queries_id[j], "token":res_dict[j]["token"]}for j in range(len(res_dict))]
				recalls, nDCGs, token = self.concurrent_answer_evaluating(query_data, test_type, name_without_extension)

				mean_recalls.append(recalls["average"])
				mean_nDCGs.append(nDCGs["average"])
				mean_token.append(token["average"])

				dataset_results.append({"id":run,"recalls":recalls,"nDCGs":nDCGs,"token":token})
				print(name_without_extension + "_" + test_type + "_" + str(num_query) + f'，第{run + 1}次recall：{recalls["average"]}, nDCG:{nDCGs["average"]}, token:{token["average"]}')
			print(name_without_extension + "_" + test_type + "_" + str(num_query) + f'，{num_runs}次平均recall：{np.mean(mean_recalls)},平均nDCG：{np.mean(mean_nDCGs)},平均token：{np.mean(mean_token)}')
			print('-' * 40)
	

if __name__ == "__main__":
	BASE_URL = "http://10.208.68.61:6006/v1/"

	base_model="Qwen3-17b"

	enable_thinking=False

	# nfcorpus
	MODELS = [base_model, "mol_nfc","mol_grpo_nfc","mol_drgrpo_nfc"]
	for MODEL_NAME in MODELS:
		print(MODEL_NAME + " is processing")
		API_KEY = "10295152"

		Evaluator = LLMEvaluator(
			model_name=MODEL_NAME,
			api_key=API_KEY,
			base_url=BASE_URL
		)
		DATASETS = [
			{"PATH": "/mnt/data/leo/server/datasets/nfcorpus", "url": "http://10.208.65.74:30804/service16/v1/v3"}
		]
		Evaluator.evaluate_dataset(DATASETS, test_type="MMLF",num_query=3)

	MODELS = [base_model, "mol_nfc","mol_drgrpo_nfc"]
	for MODEL_NAME in MODELS:
		print(MODEL_NAME + " is processing")
		API_KEY = "10295152"

		Evaluator = LLMEvaluator(
			model_name=MODEL_NAME,
			api_key=API_KEY,
			base_url=BASE_URL
		)
		DATASETS = [
			{"PATH": "/mnt/data/leo/server/datasets/nfcorpus", "url": "http://10.208.65.74:30804/service16/v1/v2"}
		]
		Evaluator.evaluate_dataset(DATASETS, test_type="MSLF",num_query=3)

	MODELS = [base_model, "mol_drgrpo_nfc"]
	for MODEL_NAME in MODELS:
		print(MODEL_NAME + " is processing")
		API_KEY = "10295152"

		Evaluator = LLMEvaluator(
			model_name=MODEL_NAME,
			api_key=API_KEY,
			base_url=BASE_URL
		)
		DATASETS = [
			{"PATH": "/mnt/data/leo/server/datasets/nfcorpus", "url": "http://10.208.65.74:30804/service16/v1/v1"}
		]
		Evaluator.evaluate_dataset(DATASETS, test_type="CQE",num_query=3)
		DATASETS = [
			{"PATH": "/mnt/data/leo/server/datasets/nfcorpus", "url": "http://10.208.65.74:30804/service16/v1/v3"}
		]
		Evaluator.evaluate_dataset(DATASETS, test_type="MQR",num_query=3)


	MODELS = ["mol_drgrpo_nfc"]
	nlist=[1,2,3,4,6,8]
	for MODEL_NAME in MODELS:
		for n in nlist:
			print(MODEL_NAME + "and num =" + str(n)+ " is processing")
			API_KEY = "10295152"

			Evaluator = LLMEvaluator(
				model_name=MODEL_NAME,
				api_key=API_KEY,
				base_url=BASE_URL
			)
			DATASETS = [
				{"PATH": "/mnt/data/leo/server/datasets/nfcorpus", "url": "http://10.208.65.74:30804/service16/v1/v3"}
			]
			Evaluator.evaluate_dataset(DATASETS, test_type="MMLF",num_query=n)

	# scifact
	MODELS = [base_model, "mol_sci","mol_grpo_sci","mol_drgrpo_sci"]
	for MODEL_NAME in MODELS:
		print(MODEL_NAME + " is processing")
		API_KEY = "10295152"

		Evaluator = LLMEvaluator(
			model_name=MODEL_NAME,
			api_key=API_KEY,
			base_url=BASE_URL
		)
		DATASETS = [
			{"PATH": "/mnt/data/leo/server/datasets/scifact", "url": "http://10.208.65.74:30804/service16/v1/v6"}
		]
		Evaluator.evaluate_dataset(DATASETS, test_type="MMLF",num_query=3)

	MODELS = [base_model, "mol_sci","mol_drgrpo_sci"]
	for MODEL_NAME in MODELS:
		print(MODEL_NAME + " is processing")
		API_KEY = "10295152"

		Evaluator = LLMEvaluator(
			model_name=MODEL_NAME,
			api_key=API_KEY,
			base_url=BASE_URL
		)
		DATASETS = [
			{"PATH": "/mnt/data/leo/server/datasets/scifact", "url": "http://10.208.65.74:30804/service16/v1/v5"}
		]
		Evaluator.evaluate_dataset(DATASETS, test_type="MSLF",num_query=3)

	MODELS = [base_model, "mol_drgrpo_sci"]
	for MODEL_NAME in MODELS:
		print(MODEL_NAME + " is processing")
		API_KEY = "10295152"

		Evaluator = LLMEvaluator(
			model_name=MODEL_NAME,
			api_key=API_KEY,
			base_url=BASE_URL
		)
		DATASETS = [
			{"PATH": "/mnt/data/leo/server/datasets/scifact", "url": "http://10.208.65.74:30804/service16/v1/v4"}
		]
		Evaluator.evaluate_dataset(DATASETS, test_type="CQE",num_query=3)
		DATASETS = [
			{"PATH": "/mnt/data/leo/server/datasets/scifact", "url": "http://10.208.65.74:30804/service16/v1/v6"}
		]
		Evaluator.evaluate_dataset(DATASETS, test_type="MQR",num_query=3)


	MODELS = ["mol_drgrpo_sci"]
	nlist=[1,2,3,4,6,8]
	for MODEL_NAME in MODELS:
		for n in nlist:
			print(MODEL_NAME + "and num =" + str(n)+ " is processing")
			API_KEY = "10295152"

			Evaluator = LLMEvaluator(
				model_name=MODEL_NAME,
				api_key=API_KEY,
				base_url=BASE_URL
			)
			DATASETS = [
				{"PATH": "/mnt/data/leo/server/datasets/scifact", "url": "http://10.208.65.74:30804/service16/v1/v6"}
			]
			Evaluator.evaluate_dataset(DATASETS, test_type="MMLF",num_query=n)

	enable_thinking=True

	# nfcorpus
	MODELS = [base_model ]
	for MODEL_NAME in MODELS:
		print(MODEL_NAME + " is processing")
		API_KEY = "10295152"
		RESULT_DIR = "./result/0812/qwen3-1.7b/mmlf/nfc/result-" + MODEL_NAME

		Evaluator = LLMEvaluator(
			model_name=MODEL_NAME,
			api_key=API_KEY,
			base_url=BASE_URL
		)
		DATASETS = [
			{"PATH": "/mnt/data/leo/server/datasets/nfcorpus", "url": "http://10.208.65.74:30804/service16/v1/v3"}
		]
		Evaluator.evaluate_dataset(DATASETS, test_type="MMLF",num_query=3)

	# scifact
	MODELS = [base_model]
	for MODEL_NAME in MODELS:
		print(MODEL_NAME + " is processing")
		API_KEY = "10295152"
		RESULT_DIR = "./result/0812/qwen3-1.7b/mmlf/sci/result-" + MODEL_NAME

		Evaluator = LLMEvaluator(
			model_name=MODEL_NAME,
			api_key=API_KEY,
			base_url=BASE_URL
		)
		DATASETS = [
			{"PATH": "/mnt/data/leo/server/datasets/scifact", "url": "http://10.208.65.74:30804/service16/v1/v6"}
		]
		Evaluator.evaluate_dataset(DATASETS, test_type="MMLF",num_query=3)

