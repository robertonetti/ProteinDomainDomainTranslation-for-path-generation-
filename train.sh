python3 -m train --trainset "data/evolution_data_csv/data_evo_cond_longer_d=490_train_transformed.csv" \
			    --valset "data/evolution_data_csv/data_evo_cond_longer_d=490_test_transformed.csv" \
			    --save "models/evo_d=490_shallow" \
				--load "" \
			    --modelconfig "shallow.config.json" \
				--outputfile "output.txt" \
				--run_name "d=490_shallowconfig" \
