### BLUE

sbatch --job-name=pm-adv submit-job-m3g-V100.sh pretrain-adv.sh 1e5
sbatch --job-name=pm-entropy submit-job-m3g-V100.sh pretrain-entropy.sh 1e5
sbatch --job-name=pm-pos submit-job-m3g-V100.sh pretrain-pos.sh 1e5
sbatch --job-name=pm-rand submit-job-m3g-V100.sh pretrain-rand.sh 1e5

sbatch --job-name=biosses-base submit-job-rqtp.sh eval_biosses.sh base /scratch/da33/trang/masked-lm/models/bert_base_uncased 3103
sbatch --job-name=biosses-rand submit-job-rqtp.sh eval_biosses.sh rand /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-9e6-random 3103
sbatch --job-name=biosses-pos submit-job-rqtp.sh eval_biosses.sh pos /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-9e6-pos 3103
sbatch --job-name=biosses-entropy submit-job-rqtp.sh eval_biosses.sh entropy /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-9e6-entropy 3103
sbatch --job-name=biosses-adv submit-job-rqtp.sh eval_biosses.sh adv /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-9e6-adv 3103
  "init_checkpoint": "/scratch/da33/trang/masked-lm/models/bert_base_uncased/bert_model.ckpt",


sbatch --job-name=chemprot-base submit-job-rqtp.sh eval_chemprot.sh base /scratch/da33/trang/masked-lm/models/bert_base_uncased 3103
sbatch --job-name=chemprot-rand submit-job-rqtp.sh eval_chemprot.sh rand /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-9e6-random 3103
sbatch --job-name=chemprot-pos submit-job-rqtp.sh eval_chemprot.sh pos /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-9e6-pos 3103
sbatch --job-name=chemprot-entropy submit-job-rqtp.sh eval_chemprot.sh entropy /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-9e6-entropy 3103
sbatch --job-name=chemprot-adv submit-job-rqtp.sh eval_chemprot.sh adv /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-9e6-adv 3103

sbatch --job-name=ddi-base submit-job-rqtp.sh eval_ddi.sh base /scratch/da33/trang/masked-lm/models/bert_base_uncased 3103
sbatch --job-name=ddi-rand submit-job-rqtp.sh eval_ddi.sh rand /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-9e6-random 3103
sbatch --job-name=ddi-pos submit-job-rqtp.sh eval_ddi.sh pos /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-9e6-pos 3103
sbatch --job-name=ddi-entropy submit-job-rqtp.sh eval_ddi.sh entropy /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-9e6-entropy 3103
sbatch --job-name=ddi-adv submit-job-rqtp.sh eval_ddi.sh adv /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-9e6-adv 3103

sbatch --job-name=hoc-base submit-job-rqtp.sh eval_hoc.sh base /scratch/da33/trang/masked-lm/models/bert_base_uncased 3103
sbatch --job-name=hoc-rand submit-job-rqtp.sh eval_hoc.sh rand /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-9e6-random 3103
sbatch --job-name=hoc-pos submit-job-rqtp.sh eval_hoc.sh pos /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-9e6-pos 3103
sbatch --job-name=hoc-entropy submit-job-rqtp.sh eval_hoc.sh entropy /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-9e6-entropy 3103
sbatch --job-name=hoc-adv submit-job-rqtp.sh eval_hoc.sh adv /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-9e6-adv 3103

sbatch --job-name=chem-base submit-job-rqtp.sh eval_chem.sh base /scratch/da33/trang/masked-lm/models/bert_base_uncased 3103
sbatch --job-name=chem-rand submit-job-rqtp.sh eval_chem.sh rand /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-9e6-random 3103
sbatch --job-name=chem-pos submit-job-rqtp.sh eval_chem.sh pos /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-9e6-pos 3103
sbatch --job-name=chem-entropy submit-job-rqtp.sh eval_chem.sh entropy /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-9e6-entropy 3103
sbatch --job-name=chem-adv submit-job-rqtp.sh eval_chem.sh adv /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-9e6-adv 3103

sbatch --job-name=disease-base submit-job-rqtp.sh eval_disease.sh base /scratch/da33/trang/masked-lm/models/bert_base_uncased 3103
sbatch --job-name=disease-rand submit-job-rqtp.sh eval_disease.sh rand /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-9e6-random 3103
sbatch --job-name=disease-pos submit-job-rqtp.sh eval_disease.sh pos /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-9e6-pos 3103
sbatch --job-name=disease-entropy submit-job-rqtp.sh eval_disease.sh entropy /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-9e6-entropy 3103
sbatch --job-name=disease-adv submit-job-rqtp.sh eval_disease.sh adv /scratch/da33/trang/masked-lm/models/pubmed/models/pubmed-da-9e6-adv 3103