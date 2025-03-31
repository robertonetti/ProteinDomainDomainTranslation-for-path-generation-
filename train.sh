python3 -m train --trainset "data/evolution_data_csv/data_evo_cond_d=49_train_transformed.csv" \
			    --valset "data/evolution_data_csv/data_evo_cond_d=49_test_transformed.csv" \
			    --save "models/evo_d=49.pth.tar" \
				--load "" \
			    --modelconfig "shallow.config.json" \
				--outputfile "output_d=49.txt"
