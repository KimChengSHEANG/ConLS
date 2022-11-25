export RESOURSES=./resources
export Result_DIR=./results

if [ ! -d $Result_DIR ] 
then
    mkdir $Result_DIR    
fi

dataset=lex.mturk.txt
python3 LSBert2.py \
  --do_eval \
  --do_lower_case \
  --num_selections 10 \
  --prob_mask 0.5 \
  --eval_dir $RESOURSES/datasets/$dataset \
  --bert_model bert-large-uncased-whole-word-masking \
  --max_seq_length 250 \
  --word_embeddings ./resources/crawl-300d-2M-subword/crawl-300d-2M-subword.vec\
  --word_frequency $RESOURSES/SUBTLEX_frequency.xlsx\
  --ppdb $RESOURSES/ppdb-2.0-tldr\
  --output_SR_file $Result_DIR/${dataset}_results.txt \
  | tee $Result_DIR/${dataset}_outputs.txt



