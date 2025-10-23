Transformer - xlmroberta
python src/train_transformer.py --train data/train.csv --val data/val.csv --test data/test.csv --use_class_weights --out_dir models/transformer/distilmbert

Baseline
python src/train_baseline.py --train data/train.csv --val data/val.csv --test data/test.csv --out_dir models/baseline

RNN
python src/train_custom_rnn.py --train data/train.csv --val data/val.csv --test data/test.csv --out_dir models/custom_rnn

Predict rnn
python src/predict_rnn.py --model_dir models/custom_rnn --input_csv data/example.csv --output_csv reports/preds.csv --proba



  
  
  
