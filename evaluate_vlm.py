from .evaluate_utils import*
import pandas as pd
import numpy as np


out_path = './output/'
all_task_json = './all_task.json'
All_MODELS = ["Qwen25_VL_7B","Qwen25_VL_72B"]
all_tasks = load_json_data(all_task_json)
all_results = []
total_results = [] 
save_path = './all_results_metric.xlsx'
weights = {
    'Vehicle_Recognition':[0.3,0.5,0.2],
    'VRU_Recognition':[0.3,0.5,0.2],
    'Obstruction_Recognition':[0.3,0.5,0.2],
    'Sign_Sign_Relation':[0.3,0.5,0.2],
    'Sign_Lane_Relation':[0.3,0.5,0.2],
    'Light_Lane_Relation':[0.3,0.5,0.2],
    'Lane_Speed_Relation':[0.3,0.5,0.2],
    'Lane_Change_Relation':[0.3,0.5,0.2],
    'VRU_Cutin':[0.7,0.1,0.2],
    'Vehicle_Cutin':[0.7,0.1,0.2],
    'VRU_Cross':[0.7,0.1,0.2],
    'Key_Obsturction_Detection':[0.8,0,0.2],
    'Risk_Prediction':[0.7,0.1,0.2],
    'Spatial_Temporal_Reasoning':[0.4,0.4,0.2]
}


for fir_ind, fir_task in enumerate(all_tasks):
    sec_tasks = all_tasks[fir_task]
    for sec_ind, sec_task in enumerate(sec_tasks):
        if sec_task=='Ego_trajectory_Planning':continue
        third_tasks = sec_tasks[sec_task]
        third_rows = 0
        for third_ind, third_task in enumerate(third_tasks):
            third_rows +=1
            model_scores = [third_task]
            print('********************First Task:{}, Second Task:{}, Third Task:{}********************'.format(fir_task,sec_task,third_task))
            for MODEL in All_MODELS:
                third_task_json_path = os.path.join(out_path, fir_task, sec_task, third_task,f'{MODEL}.json')
                if not isfile(third_task_json_path):
                    print('ERROR:{}'.format('Json loading failed!'))
                    model_scores.append(0)
                    continue
                third_task_data = load_json_data(third_task_json_path)
                ques_total_num,right_num,obey_insytruction,others = func_mapping[third_task](third_task_data,MODEL)
                # print('Model: {}, Third Task: {}, Total Questions number: {}, others: {}, Accuary: {}, Obey: {}'.format(MODEL, third_task, ques_total_num,others, right_num, obey_insytruction))
                if third_task in weights:
                    weight = weights[third_task]
                else:
                    weight = [0, 0.8, 0.2]
                temp_score = 100*others*weight[0] + 100*right_num*weight[1] + 100*obey_insytruction*weight[2]
                print('Model: {}, Third Task: {}, model_scores: {}'.format(MODEL, third_task, temp_score))
                
                model_scores.append(temp_score)
            model_scores.insert(1,ques_total_num)
            all_results.append(model_scores)
            total_results.append(model_scores)
            
        all_results = weighted_row_sum(all_results,third_rows)
        
total_ = weighted_total(total_results)
all_results.append(total_)
df = pd.DataFrame(all_results, columns=['Task','num']+All_MODELS)
df.to_excel(save_path, sheet_name='Sheet1', index=False)