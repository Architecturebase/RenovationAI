from fastapi import FastAPI, File, UploadFile, Form, Request, Depends, Path
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from typing import Annotated
from pydantic import BaseModel

import shutil
import os

from fastapi.middleware.cors import CORSMiddleware

import moviepy.editor as VideoFileClip
import itertools
from itertools import chain
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ultralytics import YOLO
import cv2
import torch

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
    entrance: str
    floor: str
    apartment: str

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
'Unfinished socket', 'Door', 'unfinished_door', 'Chandeilier', 'Toilet', 'junk', 'sticking_wires', 'Window']





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
    apartment: Annotated[str, Form(...)],
    # uploaded_params: UploadedParams
):
    global uploaded_params
    uploaded_params = UploadedParams(
        obj=obj,
        building=building,
        entrance=entrance,
        floor=floor,
        apartment=apartment,
    )
    
    return templates.TemplateResponse("video.html", {"request": request, "object": uploaded_params.obj, "building": uploaded_params.building, "entrance": uploaded_params.entrance, "floor": uploaded_params.floor, "apartment": uploaded_params.apartment})


@app.post("/upload/file")
async def create_upload_file(request: Request, file: UploadFile):
    global uploaded_params
    filename = f"{uploaded_params.apartment}.mp4"

    create_directory(f"video_base/{uploaded_params.obj}")
    create_directory(f"video_base/{uploaded_params.obj}/{uploaded_params.building}")    
    create_directory(f"video_base/{uploaded_params.obj}/{uploaded_params.building}/{uploaded_params.entrance}")
    create_directory(f"video_base/{uploaded_params.obj}/{uploaded_params.building}/{uploaded_params.entrance}/{uploaded_params.floor}")


    save_directory = f"video_base/{uploaded_params.obj}/{uploaded_params.building}/{uploaded_params.entrance}/{uploaded_params.floor}"
    save_path = os.path.join(save_directory, filename)
    
    # Проверка типа файла
    if file.content_type != "video/mp4":
        # Создание временного пути для сохранения временного файла
        temp_path = os.path.join(save_directory, f"{uploaded_params.apartment}.temp")

        # Конвертация файла в MP4 формат
        video_clip = VideoFileClip(file.file.filename)
        video_clip.write_videofile(temp_path)

        # Переименование и перемещение временного файла
        os.rename(temp_path, save_path)
    else:
        # Если файл уже является MP4, сохраняем его без изменений
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
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

    battery_completion = classes_amount.get(3) /  classes_amount.get(19) #процент установки батарей
    ceiling_completion = classes_amount.get(4) / (classes_amount.get(4) + classes_amount.get(33) + classes_amount.get(21)) #процент окончания чистовой отделки потолка
    rough_ceiling_completion = classes_amount.get(33) / (classes_amount.get(33) + classes_amount.get(21)) #процент окончания черновой отделки потолка
    laminate_completion = classes_amount.get(11) / (classes_amount.get(11) + classes_amount.get(26) + classes_amount.get(25))
    floor_screed_completion = classes_amount.get(26) / (classes_amount.get(26) + classes_amount.get(25)) #процент готовности чистовой отделки пола
    tile_completion = classes_amount.get(35) / (classes_amount.get(35) + classes_amount.get(26) + classes_amount.get(25)) * 2.5
    wall_completion = (classes_amount.get(30) + classes_amount.get(17)) / (classes_amount.get(17) + classes_amount.get(30)  + classes_amount.get(32) + classes_amount.get(27) + classes_amount.get(22))
    plastered_completion = classes_amount.get(32) / (classes_amount.get(32) + classes_amount.get(27) + classes_amount.get(22))
    socket_completion = (classes_amount.get(9) + classes_amount.get(12)) / (classes_amount.get(9) + classes_amount.get(16) + classes_amount.get(12))
    door_completion = (classes_amount.get(6) ) / (classes_amount.get(6) + classes_amount.get(36)) #процент установки дверей
    bare_ceiling_completion = classes_amount.get(21) / (classes_amount.get(4) + classes_amount.get(33) + classes_amount.get(21))


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

    if battery_completion > 0.5:
        battery_completion = 1
    else:
        battery_completion = battery_completion

    if door_completion <= 0.6:
        door_completion = door_completion / 3

    completion_list = [battery_completion, ceiling_completion, rough_ceiling_completion, laminate_completion,
                  floor_screed_completion, tile_completion, wall_completion,
                  plastered_completion, socket_completion, door_completion, bathtub_completion, toilet_completion, junk]
    
    df_dict = {'процент_установки_батарей': battery_completion, 
               'процент_чистовой_отделки_потолка': ceiling_completion,
            'черновая_отделка_потолка': rough_ceiling_completion, 'процент голого потолка': bare_ceiling_completion, 'процент_покрытия_ламинат': laminate_completion,
            'процент_готовности_стяжки_на_полу': floor_screed_completion, 'процент_покрытия_плитка': tile_completion,
            'процент_готовности_стен': wall_completion, 'процент_шпаклевки_стен': plastered_completion, 'процент_установки_розеток_переключателей': socket_completion,
            'процент_установки_дверей': door_completion, 'процент_установки_ванны': bathtub_completion, 'процент_установки_унитазов': toilet_completion,'наличие_мусора': junk, 
            'процент_установки_батарей': battery_completion}
    
    df = pd.DataFrame(df_dict, index = [0])
    path_csv = f"video_base/{uploaded_params.obj}/{uploaded_params.building}/{uploaded_params.entrance}/{uploaded_params.floor}/{uploaded_params.apartment}"
    df.to_csv(f"{path_csv}.csv")

    # формирование скор-карт и их сохранение в специальную папку
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, cmap='YlGnBu', annot=True, fmt='.2f', linewidths=2)
    plt.title('Готовность отделки')
    path_jpg = f"video_base/{uploaded_params.obj}/{uploaded_params.building}/{uploaded_params.entrance}/{uploaded_params.floor}/{uploaded_params.apartment}"
    plt.tight_layout()
    plt.savefig(f"{path_jpg}.jpg")
    plt.figure().clear()
        
    return templates.TemplateResponse("download.html", {"request": request, "video_path": video_path, "message" : "Видео загружено успешно, предикт сделан"})



@app.get("/file/download")
async def download_file():
    global uploaded_params
    if isinstance(uploaded_params, UploadedParams):
        filename = f"{uploaded_params.apartment}.mp4"

        return FileResponse(path=f"video_base/{uploaded_params.obj}/{uploaded_params.building}/{uploaded_params.entrance}/{uploaded_params.floor}/{uploaded_params.apartment}.mp4",
                            filename=filename,
                            media_type='video/mp4')
                            # headers={"Cache-Control": "no-cache"})


@app.get("/data/download/csv")
async def download_data():
    global uploaded_params
    if isinstance(uploaded_params, UploadedParams):
        filename_csv = f"{uploaded_params.apartment}.csv"

        return FileResponse(path=f"video_base/{uploaded_params.obj}/{uploaded_params.building}/{uploaded_params.entrance}/{uploaded_params.floor}/{uploaded_params.apartment}.csv",
                            filename=filename_csv,
                            media_type='text/csv')
                            # headers={"Cache-Control": "no-cache"})

@app.get("/upload/score_maps")
def get_photo1():
    global uploaded_params
    filename_jpg = f"video_base/{uploaded_params.obj}/{uploaded_params.building}/{uploaded_params.entrance}/{uploaded_params.floor}/{uploaded_params.apartment}.jpg"
    return FileResponse(filename_jpg, media_type="image/jpeg")





if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
