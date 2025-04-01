python3 -m train --trainset "data/evolution_data_csv/data_evo_cond_longer_d=490_train_transformed.csv" \
			    --valset "data/evolution_data_csv/data_evo_cond_longer_d=490_test_transformed.csv" \
			    --save "models/evo_d=490.pth.tar" \
				--load "" \
			    --modelconfig "large.config.json" \
				--outputfile "output_d=490.txt" \
				--run_name "d=490_largeconfig" \
