
import json
import os
import random
import pandas
from PIL import Image
import time
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import re
import ast

def isfile(path):
    return os.path.isfile(path)

def load_json_data(path):
    with open(path,'r',encoding='utf-8') as json_f:
        task_data = json.load(json_f)
        return task_data
    
def save_json_data(path,data):
    with open(path,'w',encoding='utf-8') as json_f:
        json.dump(data,json_f,ensure_ascii=False,indent=4)
        
root_path = 'xxx/VLADBench'
out_path = 'xxx/output/'
all_task_json = 'xxx/VLADBench/all_task.json'
MODEL = "Qwen2_VL_7B"


# # default: Load the model on the available device(s)
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "/data1/yueli/VLM_Models/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
# )
# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/data1/yueli/VLM_Models/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained("/data1/yueli/VLM_Models/Qwen2-VL-7B-Instruct")
with open(all_task_json,'r',encoding='utf-8') as json_f:
    all_tasks = json.load(json_f)
    
for fir_ind, fir_task in enumerate(all_tasks):
    sec_tasks = all_tasks[fir_task]
    for sec_ind, sec_task in enumerate(sec_tasks):
        third_tasks = sec_tasks[sec_task]
        for third_ind, third_task in enumerate(third_tasks):
            print('First Task:{}, Second Task:{}, Third Task:{}'.format(fir_task,sec_task,third_task))
            third_task_json_path = os.path.join(root_path, fir_task, sec_task, third_task+'_E.json')
            if isfile(third_task_json_path):
                if not os.path.exists(os.path.join(out_path, fir_task, sec_task, third_task)):
                    os.makedirs(os.path.join(out_path, fir_task, sec_task, third_task))
            else:
                print('ERROR:{}'.format('Json loading failed!'))

            if isfile(f'{os.path.join(out_path, fir_task, sec_task, third_task)}/{MODEL}.json'):
                print('{} file exist!'.format(third_task))
                third_task_data = load_json_data(f'{os.path.join(out_path, fir_task, sec_task, third_task)}/{MODEL}.json')
            else:
                third_task_data = load_json_data(third_task_json_path)
            start_time = time.time()
            
            for d_ind, sample in enumerate(third_task_data):
                country = sample['country']
                if 'prediction' in sample:continue
                else: sample['prediction'] = ['']*len(sample['questions'])
                for q_ind, ques in enumerate(sample['questions']):
                    img_path, qe = ques.split(';')[0], ''.join(ques.split(';')[1:])
                    if 'located at the bounding box' in qe: 
                        dimensions = re.search(r'(\d+) and (\d+)', qe)
                        width, height = float(dimensions.group(1)), float(dimensions.group(2))
                        re_qe = re.sub(r'The width and height of the image are \d+ and \d+\.', '', qe)
                        bounding_box = re.search(r'\[\d+, \d+, \d+, \d+\]', re_qe)
                        if bounding_box:
                            bounding_box = ast.literal_eval(bounding_box.group(0))
                            new_bbox = [int(1000*bounding_box[0]/width), int(1000*bounding_box[1]/height), int(1000*bounding_box[2]/width), int(1000*bounding_box[3]/height)]
                            re_qe = re.sub(r'\[\d+, \d+, \d+, \d+\]', f'{new_bbox}', re_qe)
                        else:
                            print('bounding box error')
                        qe = re_qe

                    if ('[' in img_path) and (']' in img_path):
                        seq = img_path.strip('[]')
                        all_imgs = [os.path.join(root_path, fir_task, sec_task, third_task,sample['sequence'],img_) for img_ in sample[seq]]
                        qe = f'The sequence is from {country}. {qe}'
                        messages = [
                                        {
                                            "role": "user",
                                            "content": [
                                                {
                                                    "type": "video",
                                                    "video": all_imgs,
                                                },
                                                {
                                                    "type": "text",
                                                    "text": qe},
                                            ],
                                        }
                                    ]
                        text = processor.apply_chat_template(
                                                                messages, tokenize=False, add_generation_prompt=True
                                                            )
                        image_inputs, video_inputs = process_vision_info(messages)
                        inputs = processor(
                                            text=[text],
                                            images=image_inputs,
                                            videos=video_inputs,
                                            padding=True,
                                            return_tensors="pt",
                                        )
                        inputs = inputs.to("cuda")

                        # Inference
                        generated_ids = model.generate(**inputs, max_new_tokens=128)
                        generated_ids_trimmed = [
                                                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                                                ]
                        output_text = processor.batch_decode(
                                                                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                                                            )
                        # predictions_vlm.append(output_text[-1])
                        sample['prediction'][q_ind] = output_text[-1]
                        print(sample['reference'][q_ind],output_text[-1])
                    else:
                        qe = f'The image is from {country}. {qe}'
                        image = Image.open(os.path.join(root_path, fir_task, sec_task, third_task,img_path))
                        messages = [
                                        {
                                            "role": "user",
                                            "content": [
                                                {
                                                    "type": "image",
                                                    "image": image,
                                                },
                                                {
                                                    "type": "text",
                                                    "text": qe
                                                },
                                                    ],
                                        }
                                    ]
                        text = processor.apply_chat_template(
                                                                messages, tokenize=False, add_generation_prompt=True
                                                            )   
                        image_inputs, video_inputs = process_vision_info(messages)
                        inputs = processor(
                                            text=[text],
                                            images=image_inputs,
                                            videos=video_inputs,
                                            padding=True,
                                            return_tensors="pt",
                                        )
                        inputs = inputs.to("cuda")
                        generated_ids = model.generate(**inputs, max_new_tokens=128)
                        generated_ids_trimmed = [
                                                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                                                ]
                        output_text = processor.batch_decode(
                                                                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                                                            )
                        sample['prediction'][q_ind] = output_text[-1]
                        print(sample['reference'][q_ind],output_text[-1])
            print('Third Task: {}, Time Consumption: {}'.format(third_task,time.time()-start_time))
            save_json_data(f'{os.path.join(out_path, fir_task, sec_task, third_task)}/{MODEL}.json', third_task_data)