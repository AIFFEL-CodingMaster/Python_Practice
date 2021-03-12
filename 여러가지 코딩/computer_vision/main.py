import os
import argparse
from glob import glob
import tensorflow as tf 

from make_tfrecord import MakeTFRecord
from preprocessing import delete_mat, delete_4_channel, label_encoding
from dataloader import TFRecordLoader
from model import MakeModel
from compile_option import Optimizer, Loss

def to_bool(x):
    if x.lower() in ['true','t']:
        return True
    elif x.lower() in ['false','f']:
        return False
    else:
        raise argparse.ArgumentTypeError('Bool 값을 넣으세요')

def preprocessing_1(data_path):
    data_path = data_path + "*"
    data_list = glob(data_path)
    # 전처리 
    data_list = delete_mat(data_list)
    data_list = delete_4_channel(data_list)

    data_class = label_encoding(data_list) # dictionary
    return data_list, data_class


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['tfr', 'train', 'test'], help="TFRecord 만들기 or 모델 학습 or 모델 테스트")
    parser.add_argument("--data_path", type=str, default="./", help="데이터가 들어있는 디렉토리 경로")
    parser.add_argument("--tfr_path", type=str, default="./", help="tfrecord가 저장될 디렉토리")
    parser.add_argument("--img_size", type=int, default=224, help="이미지 사이즈 입력")
    parser.add_argument("--model_name", choices=["e0", "mobilev2"], default="e0", help="모델이름을 넣어 모델 불러오기")
    parser.add_argument("--add_layer", type=str, nargs="+", help="dense 나 batch_norm 입력") 
    parser.add_argument("--dense_activation", nargs="+" ,help="activation function입력" )
    parser.add_argument("--dense_nums", nargs="+", help="히든레이어의 노드 수")
    parser.add_argument("--optimizer", type=str, default="adam" ,help="옵티마이저 선택 adam or sgd")
    parser.add_argument("--lr", type=float, default=0.01 ,help="옵티마이저의 학습률 선택")
    parser.add_argument("--loss", type=str, default="sc", help="loss 선택")
    parser.add_argument("--model_save_path", type=str, default="./model.h5", help="모델 저장 경로")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 사이즈 입력")
    parser.add_argument("--epoch", type=int, default=20, help="에포크 수 결정")
    parser.add_argument("--patience", type=int, default=0, help="몇 번까지 참아줄 것인가!?")
    parser.add_argument("--lr_schedule", type=to_bool, default='false', help=" 러닝레이트 스케쥴러를 쓸 것인지 아닌지?")
    parser.add_argument("--train_size_rate", type=float, default=0.8, help="trainset의 비율")

    args = parser.parse_args()
    
    data_list, data_class = preprocessing_1(args.data_path)

    if args.mode == "tfr":
        IMG_SIZE = args.img_size
        tfrecord = MakeTFRecord(
            data_list=data_list,
            tfr_path=args.tfr_path,
            data_class= data_class
            )
        
        if args.img_size > 224:
            tfrecord.change_img_size(args.img_size)
        # tfrecord 만들기
        tfrecord()
    
    elif args.mode == "train":
        # hyper parameter
        epoch = args.epoch
        batch_size = args.batch_size
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=args.patience)
        model_save = tf.keras.callbacks.ModelCheckpoint(
                filepath=args.model_save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1,
            )

        data_load = TFRecordLoader(args.tfr_path, args.img_size, len(data_class), args.train_size_rate, args.batch_size)
        train, valid, steps = data_load()

        make_model = MakeModel(args.model_name)

        layers = ()
        args.add_layer = args.add_layer[0].split(",")
        args.dense_nums = args.dense_nums[0].split(",")
        args.dense_activation = args.dense_activation[0].split(",")
        # layer 추가
        for layer in args.add_layer:
            if layer =="dense":
                nums = int(args.dense_nums[0])
                activation = args.dense_activation[0]
                layers +=  (make_model.add_dense_layer(nums, activation) ,)
                
                args.dense_nums.pop(0)
                args.dense_activation.pop(0)
                
            elif layer =="batch":
                layers += (make_model.add_batch_norm(),)
            else:
                raise ValueError("dense or batch만 입력하라!")
        
        model = make_model.make_model_with_FCL(args.img_size, layers)
        ## 3월 11일 모델 만들기까지 완료

        # 컴파일부터 시작
        optimizer = Optimizer(args.optimizer, lr=args.lr)
        loss = Loss(args.loss)

        model.compile(
            optimizer=optimizer.optim,
            loss=loss.loss ,
            metrics=["accuracy"]
        )

        # model 학습
        if args.lr_schedule == True:
            lr_schedule = 0 
            history = model.fit(
                train, 
                validation_data=valid,
                steps_per_epoch= steps,
                epochs=epoch,
                callbacks=[early_stopping, model_save ,lr_schedule]
            )
        
        else:
            history = model.fit(
                train, 
                validation_data=valid,
                steps_per_epoch= steps,
                epochs=epoch,
                callbacks=[early_stopping, model_save]
            )
        