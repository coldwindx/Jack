# 【eval】
# python core/eval.py --cls_model SingleChannelPredictor \
#                     --cls_dataset SingleChannelDataset \
#                     --model /home/zhulin/models/single_channel_transformer.ckpt \
#                     --dataset /mnt/sdd1/data/zhulin/jack/cdatasets.test.5.csv

# python core/eval.py --cls_model DeepRanPredictor \
#                     --cls_dataset DeepRanDataset \
#                     --model /mnt/sdd1/data/zhulin/jack/models/DeepRan-14.ckpt \
#                     --dataset /mnt/sdd1/data/zhulin/jack/cdatasets.test.5.csv \
#                     --batch_size 1024 \
#                     --output /mnt/sdd1/data/zhulin/jack/scores/DeepRanPredictor.npy

python core/eval.py --pretrain /mnt/sdd1/data/zhulin/pretrain/bert_pretrain_uncased/ \
                    --cls_model SingleChannelPredictor \
                    --cls_dataset SingleChannelDataset \
                    --model /mnt/sdd1/data/zhulin/jack/models/epoch=15-step=1599472.ckpt \
                    --dataset /mnt/sdd1/data/zhulin/jack/cdatasets.test.5.csv \
                    --batch_size 8 \
                    --output /mnt/sdd1/data/zhulin/jack/scores/Transformer.npy




# 【Train】
# python core/test.py --cls_model SingleChannelPredictor \
#                     --cls_dataset MultipleChannelDataset \
#                     --dataset /mnt/sdd1/data/zhulin/jack/mcdatasets.train.csv \
#                     --output /home/zhulin/models/single_channel_transformer.pt

# python core/train.py --cls_model DeepRanPredictor \
#                     --cls_dataset DeepRanDataset \
#                     --dataset /mnt/sdd1/data/zhulin/jack/cdatasets.train.5.csv \
#                     --batch_size 8 \
#                     --max_epochs 2 \
#                     --output /home/zhulin/models/deep_ran.pt

# python core/data.py --task add_family \
#                     --json /home/zhulin/workspace/Sun-agent/build/cdatasets.eval.json \
#                     --csv /mnt/sdd1/data/zhulin/jack/cdatasets.eval.csv

# python core/data.py --json /home/zhulin/workspace/Sun-agent/build/mcdatasets.eval.json \
#                     --csv /mnt/sdd1/data/zhulin/jack/mcdatasets.eval.csv

# python core/data.py --json /home/zhulin/workspace/Sun-agent/build/cdatasets.train.zl.json,/home/zhulin/datasets/cdatasets.train.5.json \
#                     --csv /mnt/sdd1/data/zhulin/jack/cdatasets.train.csv

# python core/data.py --task model_2_model --model /mnt/sdd1/data/zhulin/jack/models/Transformer.ckpt
# python core/data.py --task model_2_model --model /mnt/sdd1/data/zhulin/jack/models/epoch=15-step=1599472.ckpt


# [Train Fasttext]
# python core/pretrain.py --task TfidfVectorizer --dataset /home/zhulin/datasets/cdatasets.train.5.csv --output tfidf.ph
# python core/test.py --fasttext ./common/fasttext.ph \
#                     --tfidf ./common/tfidf.ph \
#                     --dataset /home/zhulin/datasets/cdatasets.test.5.csv
