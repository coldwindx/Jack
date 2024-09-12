# eval
# python core/test.py --cls_model SingleChannelPredictor \
#                     --cls_dataset SingleChannelDataset \
#                     --model /home/zhulin/models/single_channel_transformer.ckpt \
#                     --dataset /home/zhulin/datasets/cdatasets.test.5.csv

# 【Train】
# python core/test.py --cls_model SingleChannelPredictor \
#                     --cls_dataset MultipleChannelDataset \
#                     --dataset /mnt/sdd1/data/zhulin/jack/mcdatasets.train.csv \
#                     --output /home/zhulin/models/single_channel_transformer.pt

python core/data.py --task add_family \
                    --json /home/zhulin/workspace/Sun-agent/build/cdatasets.eval.json \
                    --csv /mnt/sdd1/data/zhulin/jack/cdatasets.eval.csv

# python core/data.py --json /home/zhulin/workspace/Sun-agent/build/mcdatasets.eval.json \
#                     --csv /mnt/sdd1/data/zhulin/jack/mcdatasets.eval.csv

# python core/data.py --json /home/zhulin/workspace/Sun-agent/build/cdatasets.train.zl.json,/home/zhulin/datasets/cdatasets.train.5.json \
#                     --csv /mnt/sdd1/data/zhulin/jack/cdatasets.train.csv
