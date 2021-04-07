import os 
import argparse
from glob import glob
import tensorflow as tf
import pandas as pd


from make_tfrecord import MakeTFRecord
from preprocessing import delete_mat, delete_4_channel, label_encoding
from dataloader import TFRecordLoader
from model import MakeModel, TestModel
from prediction import Prediction

def preprocessing_1(data_path):
    data_path = data_path + "*"
    data_list = glob(data_path)

    # 전처리
    data_list = delete_mat(data_list)
    data_list = delete_4_channel(data_list)

    data_class = label_encoding(data_list)
    return data_list, data_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['tfr', 'train', 'test'], help="TFRecord 만들기 or 모델 학습 or 모델 테스트")
    parser.add_argument("--data_path", type=str, default="./", help="데이터가 들어있는 디렉토리 경로")
    parser.add_argument("--tfr_path", type=str, default="./", help="tfrecord가 저장될 디렉토리")
    parser.add_argument("--img_size", type=int, default=224, help="이미지 사이즈 입력")
    parser.add_argument("--epoch", type=int, default=20, help="에포크 수 입력")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기 입력")
    parser.add_argument("--patience", type=int, default=1, help="몇 번 참을 것인가?")
    parser.add_argument("--model_path", type=str, default="./", help="모델이 저장될 경로 입력")
    parser.add_argument("--train_size_rate", type=float, default=0.8, help="학습 데이터 비율 입력")
    parser.add_argument("--model_name", choices=["e0", "mobilev2"], default="e0", help="불러올 모델 이름 입력")
    parser.add_argument("--add_layer", type=str, nargs="+", help="dense or batch")
    parser.add_argument("--dense_nums", nargs="+", help="dense의 뉴런 수 입력")
    parser.add_argument("--dense_activation", nargs="+", help="activation function 입력")
    parser.add_argument("--hist_path", type=str, help="학습 히스토리를 저장하는 경로")
    parser.add_argument("--test_path", type=str, help="테스트 데이터의 경로" )
    args = parser.parse_args()


    data_list, data_class = preprocessing_1(args.data_path)
    if args.mode == "tfr":

        IMG_SIZE = args.img_size
        tfrecord = MakeTFRecord(
            data_list = data_list,
            tfr_path = args.tfr_path,
            data_class = data_class
        )

        if args.img_size != 224:
            tfrecord.change_img_size(args.img_size)
        # tfrecord 만들기
        tfrecord()
    
    elif args.mode == "train":
        # hyper parameters
        epoch = args.epoch 
        batch_size = args.batch_size
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=args.patience)
        model_save = tf.keras.callbacks.ModelCheckpoint(
            filepath=args.model_path,
            monitor='val_loss',
            save_best_only=True, 
            verbose=1
        )

        # dataload
        data_load = TFRecordLoader(
            args.tfr_path,
            args.img_size,
            len(data_class),
            args.train_size_rate,
            batch_size
        )

        train, valid, steps = data_load()

        # make model 
        make_model = MakeModel(args.model_name)
        # 추가할 layer
        layers = ()

        args.add_layer = args.add_layer[0].split(",")
        args.dense_nums = args.dense_nums[0].split(",")
        args.dense_activation = args.dense_activation[0].split(",")

        # layer 추가 
        for layer in args.add_layer:
            if layer == "dense":
                nums = int(args.dense_nums[0])
                activation = args.dense_activation[0]
                layers += ( make_model.add_dense_layer(nums, activation), )

                args.dense_nums.pop(0)
                args.dense_activation.pop(0)
            
            elif layer == "batch":
                layers += ( make_model.add_batch_norm(), )
            
            else:
                raise ValueError("dense or batch만 입력")
        
        model = make_model.make_model_with_FCL(args.img_size, layers)

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

        history = model.fit(
            train,
            validation_data=valid,
            steps_per_epoch=steps,
            epochs=epoch,
            callbacks=[early_stopping, model_save]
        )

        hist = pd.DataFrame(history.history)
        hist.to_pickle(args.hist_path)

    elif args.mode == "test":
        test_model = TestModel(args.model_path)
        test_model = test_model.test_model()

        pred = Prediction()

        pred.predict_test(args.test_path, test_model, args.img_size)