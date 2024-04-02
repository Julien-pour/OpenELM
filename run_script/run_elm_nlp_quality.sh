sh export_key.sh # script to export the key
cd ..
seed=1
path="/home/flowers/work/OpenELM/logs/elm/env=p3_probsol_Chat_IMGEP_smart/24-02-16_16:11/step_80"
python run_elm.py --config-name=elm_nlp_quality seed=$seed env.seed=$seed qd.seed=$seed 
#for starting from a previous archive add: 'qd.loading_snapshot_map=True' 'qd.log_snapshot_dir=$path'
