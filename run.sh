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

python core/train.py --cls_model DeepRanPredictor \
                    --cls_dataset DeepRanDataset \
                    --dataset /mnt/sdd1/data/zhulin/jack/cdatasets.train.5.csv \
                    --batch_size 1024 \
                    --output /home/zhulin/models/deep_ran.pt

# python core/data.py --task add_family \
#                     --json /home/zhulin/workspace/Sun-agent/build/cdatasets.eval.json \
#                     --csv /mnt/sdd1/data/zhulin/jack/cdatasets.eval.csv

# python core/data.py --json /home/zhulin/workspace/Sun-agent/build/mcdatasets.eval.json \
#                     --csv /mnt/sdd1/data/zhulin/jack/mcdatasets.eval.csv

# python core/data.py --json /home/zhulin/workspace/Sun-agent/build/cdatasets.train.zl.json,/home/zhulin/datasets/cdatasets.train.5.json \
#                     --csv /mnt/sdd1/data/zhulin/jack/cdatasets.train.csv

# [Train Fasttext]
# python core/pretrain.py --task TfidfVectorizer --dataset /home/zhulin/datasets/cdatasets.train.5.csv --output tfidf.ph
# python core/test.py --fasttext ./common/fasttext.ph \
#                     --tfidf ./common/tfidf.ph \
#                     --dataset /home/zhulin/datasets/cdatasets.test.5.csv
