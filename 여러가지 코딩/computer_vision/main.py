import os
import argparse
from glob import glob

from make_tfrecord import MakeTFRecord
from preprocessing import delete_mat, delete_4_channel, label_encoding
from dataloader import TFRecordLoader
from model import MakeModel

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
    parser.add_argument("--dense_activation", nargs="+" help="activation function입력" )
    parser.add_argument("--dense_nums", nargs="+", help="히든레이어의 노드 수")

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
        data_load = TFRecordLoader(args.tfr_path, args.img_size, len(data_class))
        dataset = data_load()

        make_model = MakeModel(args.model_name)

        layers = ()
        # layer 추가
        for layer in args.add_layer:
            if layer =="dense":
                nums = int(args.dense_nums[0])
                activation = args.dense_activation[0]
                layers +=  (make_model.add_dense_layer(nums, activation),)
                
                args.dense_nums.pop(0)
                args.dense_activation.pop(0)
                
            elif layer =="batch":
                layers += (make_model.add_batch_norm())
            else:
                raise ValueError("dense or batch만 입력하라!")
        
        model = make_model.make_model_with_FCL(args.img_size, layers)
        ## 3월 11일 모델 만들기까지 완료