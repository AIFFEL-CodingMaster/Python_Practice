import os
import re
import numpy as np
from PIL import Image
from tensorflow import keras
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render

model_path = os.path.join("model", "model.h5")
model = keras.models.load_model(model_path)

label = {
    0: "호랑이",
    1: "사자" 
}

def predict(request):
    file_path = request.FILES['filePath']
    if re.findall("[ㄱ-ㅎ가-힣]", file_path.name) != []:
        tail = os.path.splitext(file_path.name)[1]
        random_number = str(np.random.randint(0, 99999999))
        file_path.name = "upload_image_" + random_number + tail

    fs = FileSystemStorage()
    file = fs.save(file_path.name, file_path)
    file = fs.url(file)

    # "." + "/uploads/파일명"
    image = Image.open("." + file)
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image)

    normalized_image_array = (image.astype(np.float32) / 127.0) - 1
    # (1, 224,224,3)
    normalized_image_array = normalized_image_array[np.newaxis, :,:,:]

    pred = model.predict(normalized_image_array)
    pred = np.argmax(pred[0]) # 0 or 1

    predicted_label = label[pred] # 호랑이 or 사자

    message = f"예측 결과는 {predicted_label} 입니다"

    context = {
        "message" : message
    }

    return render(request, "result.html", context)
