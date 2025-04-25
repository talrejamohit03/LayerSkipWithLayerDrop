#!/bin/bash
#SBATCH -JLayerSkipWithLayerDrop             # Job name
#SBATCH -N1 --ntasks-per-node=4          # Number of nodes and cores per node required
#SBATCH --gres=gpu:H100:1                # GPU type (H100) and number of GPUs 
#SBATCH --mem-per-gpu=30GB              # Memory per CPU core, 8 CPUs/GPU 
#SBATCH -t45                             # Duration of the job (Ex: 15 mins)
#SBATCH -oReport-%j.out                  # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail preferences
#SBATCH --mail-user=mtalreja6@gatech.edu # E-mail address for notifications
cd $SLURM_SUBMIT_DIR                     # Change to working directory

# module load anaconda3                    # Load module dependencies
source ~/.bashrc
huggingface-cli login --token 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
conda activate /storage/ice1/4/0/mtalreja6/layerskip/LayerSkip/layer_skip_env         # Activate the conda environment
which python    
srun torchrun eval.py --model facebook/layerskip-llama2-7B     --tasks hellaswag     --limit 10     --generation_strategy self_speculative     --exit_layer 8     --num_speculations 6     --output_dir ./logs
