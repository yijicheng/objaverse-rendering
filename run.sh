
# # 46207
# python3 scripts/distributed.py --num_gpus 8 --workers_per_gpu 8 --start $1 --end $2 --lvis --output_dir views_lvis_sphere_fix_lightning_transparent_bkg

# 549922
python3 scripts/distributed.py --num_gpus 8 --workers_per_gpu 8 --start $1 --end $2 --cap3d_hq --output_dir views_cap3d_hq_sphere_fix_lightning_transparent_bkg

