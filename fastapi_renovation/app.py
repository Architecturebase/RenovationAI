from fastapi import FastAPI, File, UploadFile, Form, Request, Depends, Path
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from typing import Annotated, Optional
from pydantic import BaseModel

import shutil
import os
import numpy as np

from fastapi.middleware.cors import CORSMiddleware

from moviepy.editor import VideoFileClip
import itertools
from itertools import chain
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ultralytics import YOLO
import cv2
import torch
import re
import time

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "free-gpt.ru",
    "http://free-gpt.ru",
    "https://free-gpt.ru"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "DELETE", "PATCH", "PUT"],
    allow_headers=["Content-Type", "Set-Cookie", "Access-Control-Allow-Headers", "Access-Control-Allow-Origin",
                   "Authorization"],
)

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")



class UploadedParams(BaseModel):
    obj: str
    building: str
    entrance: str = ""
    floor: str = ""
    apartment: str = ""


#все классы
names = {0: 'Balcony door', 1: 'Balcony long window', 2: 'Bathtub', 3: 'Battery', 4: 'Ceiling', 5: 'Chandelier', 6: 'Door', 
         7: 'Electrical panel', 8: 'Fire alarm', 9: 'Good Socket', 10: 'Gutters', 11: 'Laminatte', 12: 'Light switch', 
         13: 'Plate', 14: 'Sink', 15: 'Toilet', 16: 'Unfinished socket', 17: 'Wall tile', 18: 'Wallpaper', 19: 'Window', 
         20: 'Windowsill', 21: 'bare_ceiling', 22: 'bare_wall', 23: 'building_stuff', 24: 'bulb', 25: 'floor_not_screed', 
         26: 'floor_screed', 27: 'gas_blocks', 28: 'grilyato', 29: 'junk', 30: 'painted_wall', 31: 'pipes', 
         32: 'plastered_walls', 33: 'rough_ceiling', 34: 'sticking_wires', 35: 'tile', 36: 'unfinished_door', 
         37: 'unnecessary_hole'}
names = {y: x for x, y in names.items()}

interested_classes = ['Bathtub', 'Battery', 'Ceiling', 'rough_ceiling', 'bare_ceiling', 'Laminatte', 'floor_screed', 'floor_not_screed', 
'tile', 'painted_wall', 'plastered_walls', 'gas_blocks', 'bare_wall', 'Wall tile', 'Gutters', 'Good Socket', 'Light switch',
'Unfinished socket', 'Door', 'unfinished_door', 'Chandeilier', 'Toilet', 'junk', 'sticking_wires', 'Window', 'Sink']





#создании функции для создания или загрузки в нужную директорию
def create_directory(directory_path: str) -> str:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


#функции для работы с результатами модели
def get_clean_boxes(results):
    clean_boxes_list = []
    for i in range(len(results)):
        if len(results[i].boxes.cls) != 0:
            clean_boxes_list.append(results[i].boxes.cls)
        else:
            continue
    #         clean_boxes_list.append(torch.tensor([]))
    return clean_boxes_list

def clean_tensors(clean_boxes): 
    ids = []
    ids_flatten = []
    for i in range(len(clean_boxes)):
        if len(clean_boxes[i]) <= 1:
            ids.append(int(clean_boxes[i].item()))
            ids_flatten.append(int(clean_boxes[i].item()))
        else:
            long_ids = []
            for k in range(len(clean_boxes[i])):
                long_ids.append(clean_boxes[i][k].item())
            long_ids = [int(x) for x in long_ids]
            ids.append(tuple(long_ids))
            
    ids_clean = [item for sublist in ids for item in (sublist if isinstance(sublist, tuple) else [sublist])]

    ids_sorted = [k for k,_g in itertools.groupby(ids)]
    ids_flatten_sorted = [k for k, _g in itertools.groupby(ids_flatten)]
    return ids_clean

def get_frames_amount(source):
    cap = cv2.VideoCapture(source)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length

def get_fps(source):
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps

def get_classes_amount(interested_classes, ids, decimator):
    classes_amount = {}
    for j in range(len(interested_classes)):
        label_num = names.get(interested_classes[j])
        label_amount = ids.count(label_num)
        if label_amount < decimator / 6:
            label_amount = 0
        classes_amount[label_num] = label_amount
    return classes_amount


# функция преобразования атрибута объекта в номер для занесения в строку
def last_occur(some_folder, substring):
    for item in reversed(some_folder):
        if substring in item:
            last_occurence = item
            break
    return last_occurence

def only_number(occur):
    numbers = re.findall(r'\d+', occur)
    if numbers:
        result = int(''.join(numbers))
    else:
        result = occur
    return result

# функция для открытия всех csv в ЖК и формирование датафрейма
def all_csv_opener(directory):
    
    dataframes = []
    dataframe_list = []
    folder_list = []
    
    building_l = []
    entrance_l = []
    floor_l = []
    
    for root, dirs, files in os.walk(directory):
        folder_name = os.path.basename(root)
        folder_list.append(folder_name)
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                dataframes.append(df)
                dataframe_name = folder_name
                dataframe_list.append(dataframe_name)
                
                building_occur = last_occur(folder_list, 'building')
                building_num = only_number(building_occur)
                building_l.append(building_num)
                
                entrance_occur = last_occur(folder_list, 'entrance')
                entrance_num = only_number(entrance_occur)
                entrance_l.append(entrance_num)
                
                floor_occur = last_occur(folder_list, 'floor')
                floor_num = only_number(floor_occur)
                floor_l.append(floor_num)
                
    combined_df = pd.concat(dataframes, ignore_index = True)
    combined_df = combined_df.drop('Unnamed: 0', axis = 1)
    combined_df['этажи'] = floor_l
    combined_df['подъезд'] = entrance_l
    combined_df['строение'] = building_l
    combined_df['этажи'] = combined_df['этажи'].astype(str)
    combined_df['подъезд'] = combined_df['подъезд'].astype(str)
    combined_df['строение'] = combined_df['строение'].astype(str)
#     dataframe_name = os.path.splitext(file)[0]
    
    return combined_df, dataframe_list, folder_list


#напиши функцию, которая по дефолту установит 0 или None в зависимости от инпутов формы

def check_inputs(building_n, entrance_n, combined_df):
        
    if building_n == None:
        building_n = combined_df['строение'].unique()
    else:
        building_n = [building_n]
    
    if ((entrance_n == None) or (entrance_n == "")):
        entrance_n = combined_df['подъезд'].unique()
    else:
        entrance_n = [entrance_n]
        
    
    return building_n, entrance_n, combined_df
    
def construct_df(building_n, entrance_n, combined_df):
    df_c = combined_df #df_c is df_construct
    dfs_list = []
    
    df_constructed = df_c[(df_c['подъезд'].isin(entrance_n))
                      & (df_c['строение'].isin(building_n))]
    dfs_list.append(df_constructed)
    
    df_constructed = pd.concat(dfs_list, axis = 1)
    # df_constructed = df_constructed.sort_values('этажи', ascending=False)

    return df_constructed

# функция для построения графика
def plot_by_building(agg_df, params: UploadedParams):

    agg_building = agg_df['строение'].unique()
    df_pivot_1 = agg_df
    agg_cols = agg_df.columns[:-3]
    
    
    for s in range(len(agg_building)):
        lm=0
        for l in range(len(agg_cols)):

            lm += 1
            print(lm)
            df_pivot_1_total = pd.pivot_table(df_pivot_1,
                                             index = 'этажи',
                                             columns = 'подъезд',
                                             values = agg_cols[l],
                                             aggfunc = np.mean)

            plt.figure(figsize = (12, 8))
            sns.set(font_scale = 1.2)
            colors = sns.dark_palette("#69d", reverse=True, as_cmap=True)
            sns.heatmap(df_pivot_1_total, cmap=colors, annot=True, fmt='.2f', linewidths=1, linecolor='white', cbar=False, square=True, alpha=0.8)
            
            # Получаем список меток оси Y
            y_labels = plt.gca().get_yticklabels()

            # Создаем перевернутый список меток
            reversed_y_labels = list(reversed([label.get_text() for label in y_labels]))

            # Устанавливаем перевернутые метки на оси Y
            plt.gca().set_yticklabels(reversed_y_labels)
            
            
            title_font = {'fontname': 'Arial', 'size': '20', 'weight': 'bold'}
            plt.title(f'{agg_cols[l].replace("_", " ")}', **title_font)
            plt.tight_layout()

            # Изменяем размер шрифта меток осей
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            # Изменяем размер шрифта названия оси x
            plt.xlabel('Подъезды', fontsize=18)

            # Изменяем размер шрифта названия оси y
            plt.ylabel('Этажи', fontsize=18)

            save_path_map = f'score_maps/{params.obj}/{params.building}'
            print(save_path_map)

            plt.savefig(f"{save_path_map}/{params.obj}_{params.building}_{lm}.jpg")
            plt.figure().clear()






# описание бек-енда приложения

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/info/")
async def info(
    request: Request,
    obj: Annotated[str, Form(...)],
    building: Annotated[str, Form(...)],
    entrance: Annotated[str, Form(...)],
    floor: Annotated[str, Form(...)],
    apartment: Annotated[str, Form(...)]
):
    global uploaded_params
    uploaded_params = UploadedParams(
        obj=obj,
        building=building,
        entrance=entrance,
        floor=floor,
        apartment=apartment,
    )

    # request.app.state.uploaded_params = uploaded_params
    
    return templates.TemplateResponse("video.html", {"request": request, "object": uploaded_params.obj, "building": uploaded_params.building, "entrance": uploaded_params.entrance, "floor": uploaded_params.floor, "apartment": uploaded_params.apartment})



@app.post("/upload/file")
async def create_upload_file(request: Request, file: UploadFile):
    
    # uploaded_params = request.app.state.uploaded_params
    
    # global uploaded_params

    filename = f"{uploaded_params.apartment}.mp4"

    create_directory(f"video_base/{uploaded_params.obj}")
    create_directory(f"video_base/{uploaded_params.obj}/building_{uploaded_params.building}")    
    create_directory(f"video_base/{uploaded_params.obj}/building_{uploaded_params.building}/entrance_{uploaded_params.entrance}")
    create_directory(f"video_base/{uploaded_params.obj}/building_{uploaded_params.building}/entrance_{uploaded_params.entrance}/floor_{uploaded_params.floor}")


    save_directory = f"video_base/{uploaded_params.obj}/building_{uploaded_params.building}/entrance_{uploaded_params.entrance}/floor_{uploaded_params.floor}"
    save_path = os.path.join(save_directory, filename)
    
   # Проверяем тип файла
    if file.content_type != "video/mp4":
        # Создаем временный файл на диске и сохраняем содержимое файла в него
        temp_path = os.path.join(save_directory, f"{file.filename}.temp")
        with open(temp_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
        
        # Конвертируем временный файл в MP4 формат
        video_clip = VideoFileClip(temp_path)
        video_clip.write_videofile(save_path)
        
        # Удаляем временный файл
        os.remove(temp_path)
    else:
        # Если файл уже является MP4, сохраняем его без изменений
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

    source = save_path
    model_path = 'weights/best.pt'
    model = YOLO(model_path)
    results = model.predict(source, imgsz=640, conf=0.4, save=True)

    detect_dir = 'runs/detect'

    #открываем папку директории с моделью и берем последнюю созданную папку

    folders = [f for f in os.listdir(detect_dir) if os.path.isdir(os.path.join(detect_dir, f))]
    latest_folder = max(folders, key=lambda x: os.path.getmtime(os.path.join(detect_dir, x)))
    latest_folder_path = os.path.join(detect_dir, latest_folder)
    video_files = [f for f in os.listdir(latest_folder_path) if os.path.isfile(os.path.join(latest_folder_path, f)) and f.endswith('.mp4')]
    video_path = save_directory
    if len(video_files) != 1:
        return {"error": "Ошибка: Не найден единственный видеофайл в последней папке."}
    else:
        video_file_path = os.path.join(latest_folder_path, video_files[0])
        destination_dir = save_directory
        destination_path = os.path.join(destination_dir, video_files[0])
        with open(video_file_path, 'rb') as source_file, open(destination_path, 'wb') as destination_file:
            shutil.copyfileobj(source_file, destination_file)

    # теперь идет обработка результатов модели
    num_classes = [int(x) for x in range(len(names))]

    clean_boxes = get_clean_boxes(results)
    ids = clean_tensors(clean_boxes)
    # fps = get_fps(source)
    frames_amount = get_frames_amount(source)
    decimator = 10
    classes_amount = get_classes_amount(interested_classes, ids, decimator)

    def safe_division(numerator, denominator):
        try:
            result = numerator / denominator
        except ZeroDivisionError:
            result = 0
        return result

    battery_completion = safe_division(classes_amount.get(3), classes_amount.get(19))
    ceiling_completion = safe_division(classes_amount.get(4), classes_amount.get(4) + classes_amount.get(33) + classes_amount.get(21))
    rough_ceiling_completion = safe_division(classes_amount.get(33), classes_amount.get(33) + classes_amount.get(21))
    laminate_completion = safe_division(classes_amount.get(11), classes_amount.get(11) + classes_amount.get(26) + classes_amount.get(25))
    floor_screed_completion = safe_division(classes_amount.get(26), classes_amount.get(26) + classes_amount.get(25))
    tile_completion = safe_division(classes_amount.get(35), classes_amount.get(35) + classes_amount.get(26) + classes_amount.get(25))
    wall_completion = safe_division((classes_amount.get(30) + classes_amount.get(17)), classes_amount.get(17) + classes_amount.get(30) + classes_amount.get(32) + classes_amount.get(27) + classes_amount.get(22))
    plastered_completion = safe_division(classes_amount.get(32), classes_amount.get(32) + classes_amount.get(27) + classes_amount.get(22))
    socket_completion = safe_division((classes_amount.get(9) + classes_amount.get(12)), classes_amount.get(9) + classes_amount.get(16) + classes_amount.get(12))
    door_completion = safe_division(classes_amount.get(6), classes_amount.get(6) + classes_amount.get(36))
    bare_ceiling_completion = safe_division(classes_amount.get(21), classes_amount.get(4) + classes_amount.get(33) + classes_amount.get(21))



    if classes_amount.get(15) > 10:
        toilet_completion = 1
    else:
        toilet_completion = 0

    if classes_amount.get(2) > 5:
        bathtub_completion = 1
    else:
        bathtub_completion = 0

    if classes_amount.get(29) > 5:
        junk = 1
    else:
        junk = 0
    
    if classes_amount.get(14) > 10:
        sink_completion = 1
    else:
        sink_completion = 0

    if battery_completion > 0.5:
        battery_completion = 1
    else:
        battery_completion = battery_completion

    if door_completion <= 0.6:
        door_completion = door_completion / 3
    
    if wall_completion >= 0.98 and plastered_completion <= 0.02:
        plastered_completion = 1

    if laminate_completion >= 0.98 and floor_screed_completion <= 0.02:
        floor_screed_completion = 1

    if ceiling_completion >= 0.98 and rough_ceiling_completion <= 0.02 and bare_ceiling_completion <= 0.05:
        rough_ceiling_completion = 1



    completion_list = [battery_completion, ceiling_completion, rough_ceiling_completion, laminate_completion,
                  floor_screed_completion, tile_completion, wall_completion,
                  plastered_completion, socket_completion, door_completion, bathtub_completion, toilet_completion, sink_completion, junk]
    
    df_dict = {'процент_установки_батарей': battery_completion, 
               'процент_чистовой_отделки_потолка': ceiling_completion,
            'черновая_отделка_потолка': rough_ceiling_completion, 'процент голого потолка': bare_ceiling_completion, 'процент_покрытия_ламинат': laminate_completion,
            'процент_готовности_стяжки_на_полу': floor_screed_completion, 'процент_покрытия_плитка': tile_completion,
            'процент_готовности_стен': wall_completion, 'процент_шпаклевки_стен': plastered_completion, 'процент_установки_розеток_переключателей': socket_completion,
            'процент_установки_дверей': door_completion, 'процент_установки_ванны': bathtub_completion, 'процент_установки_унитазов': toilet_completion,'наличие_мусора': junk, 
            'процент_установки_батарей': battery_completion, 'процент_установки_раковин' : sink_completion}
    
    df = pd.DataFrame(df_dict, index = [0])
    path_csv = f"video_base/{uploaded_params.obj}/building_{uploaded_params.building}/entrance_{uploaded_params.entrance}/floor_{uploaded_params.floor}/{uploaded_params.apartment}"
    df.to_csv(f"{path_csv}.csv", index=False)

    # формирование скор-карт и их сохранение в специальную папку
    plt.figure(figsize=(12, 8))
    sns.set(font_scale=1.2)
    colors = sns.color_palette("Blues", as_cmap=True)
    # Построение тепловой карты с расстоянием между ячейками
    sns.heatmap(df, cmap=colors, annot=True, fmt='.2f', linewidths=1, linecolor='white', cbar=False, square=True, alpha=0.8)
    # Задаем стиль заголовка
    title_font = {'fontname': 'Arial', 'size': '20', 'weight': 'bold'}
    plt.title('Готовность отделки', **title_font)
    plt.tight_layout()
    path_jpg = f"video_base/{uploaded_params.obj}/building_{uploaded_params.building}/entrance_{uploaded_params.entrance}/floor_{uploaded_params.floor}/{uploaded_params.apartment}"
    plt.savefig(f"{path_jpg}.jpg")
    plt.figure().clear()
        
    return templates.TemplateResponse("download.html", {"request": request, "video_path": video_path, "message" : "Видео загружено успешно, предикт сделан"})



@app.get("/file/download")
async def download_file(request: Request):
    # uploaded_params = request.app.state.uploaded_params
    global uploaded_params

    filename = f"{uploaded_params.apartment}.mp4"

    return FileResponse(path=f"video_base/{uploaded_params.obj}/building_{uploaded_params.building}/entrance_{uploaded_params.entrance}/floor_{uploaded_params.floor}/{uploaded_params.apartment}.mp4",
                        filename=filename,
                        media_type='video/mp4')
                        # headers={"Cache-Control": "no-cache"})


@app.get("/data/download/csv")
async def download_data(request: Request):
    
    # uploaded_params = request.app.state.uploaded_params
    global uploaded_params

    filename_csv = f"{uploaded_params.apartment}.csv"

    return FileResponse(path=f"video_base/{uploaded_params.obj}/building_{uploaded_params.building}/entrance_{uploaded_params.entrance}/floor_{uploaded_params.floor}/{uploaded_params.apartment}.csv",
                        filename=filename_csv,
                        media_type='text/csv')
                        # headers={"Cache-Control": "no-cache"})





@app.get("/upload/score_maps")
def get_photo1(request: Request):
    
    global uploaded_params

    filename_jpg = f'{uploaded_params.apartment}.jpg'

    

    return FileResponse(path=f"video_base/{uploaded_params.obj}/building_{uploaded_params.building}/entrance_{uploaded_params.entrance}/floor_{uploaded_params.floor}/{uploaded_params.apartment}.jpg",
                        filename= filename_jpg,
                        media_type="image/jpeg")
                        # headers={"Cache-Control": "no-cache"})


# Создание ендпоинтов для получения суммарной скор карты по объекту
@app.get("/summary")
async def summary(request: Request):
    return templates.TemplateResponse("summary.html", {"request": request})

@app.post("/get/summary")
async def get_summary(request: Request,
    obj_s: Annotated[str, Form(...)],
    building_s: Annotated[str, Form(...)],
    entrance_s: Optional[str] = Form(None)
):
    global uploaded_params_s
    uploaded_params_s = UploadedParams(
        obj=obj_s,
        building=building_s,
        entrance=entrance_s or ""
    )

   
    create_directory(f'score_maps/{uploaded_params_s.obj}')
    create_directory(f'score_maps/{uploaded_params_s.obj}/{uploaded_params_s.building}')
    combined_df = all_csv_opener(f'video_base/{uploaded_params_s.obj}')[0]


    # сохраняем сводный df в папку
    combined_df.to_csv(f'score_maps/{uploaded_params_s.obj}/{uploaded_params_s.building}/{uploaded_params_s.obj}_{uploaded_params_s.building}.csv', index=False)

    floors_list = all_csv_opener(f'video_base/{uploaded_params_s.obj}')[1]

    # проверка инпутов из формы для получения разреза аналитики
    var_checker = check_inputs(
                               uploaded_params_s.building,
                               uploaded_params_s.entrance,
                               combined_df)
    
    print(var_checker[0], var_checker[1])
    
    # получение нужного датафрейма для скор карт
    df_to_plot = construct_df(
                              var_checker[0], 
                              var_checker[1],  
                              combined_df)
    
  
    

    plot_by_building(df_to_plot, uploaded_params_s)


    return templates.TemplateResponse("summary.html", {"request": request})
    




# функции обработчики для скачивания тотальных данных
@app.get("/summary/total_data")
async def download_tota_csv(request: Request):
    global uploaded_params_s
    total_csv_path = f'score_maps/{uploaded_params_s.obj}/{uploaded_params_s.building}/{uploaded_params_s.obj}_{uploaded_params_s.building}.csv'

    return FileResponse(path=total_csv_path,
                    filename=f'{uploaded_params_s.obj}.csv',
                    media_type='text/csv')


# функция для отображения фото на сайте
@app.get("/summary/data_{id}")
def get_stats_id(request: Request, id: int):
        
    global uploaded_params_s

    filename_jpg = f"score_maps/{uploaded_params_s.obj}/{uploaded_params_s.building}/{uploaded_params_s.obj}_{uploaded_params_s.building}_{id}.jpg"

    return FileResponse(path=filename_jpg, 
                        media_type="image/jpeg")


    



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
