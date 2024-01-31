#python cl_train_glb.py --client_num 1 --gpu 1 --version cl_rf_cn1_c1 --num_epochs 1000
#python cl_train_glb.py --client_num 2 --gpu 1 --version cl_rf_cn1_c2 --num_epochs 1000
#python cl_train_glb.py --client_num 3 --gpu 1 --version cl_rf_cn4_dup_c3_m3 --num_epochs 1000
#python cl_train_glb.py --client_num 4 --gpu 1 --version cl_rf_cn4_dup_c4_m3 --num_epochs 1000

python fl_train_clsPasData_async_18.py --client_num  4 --c_rounds 2000 --round_per_train 100 --version brats20_rf_c4_3m_dup50 --resume 0
python fl_train_clsPasData_async_18.py --client_num  4 --c_rounds 2000 --round_per_train 100 --version brats20_rf_c4_3m_dup50 --resume 1
