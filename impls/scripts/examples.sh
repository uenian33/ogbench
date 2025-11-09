python main_reachability.py --agent_type=rws --dataset_type=ogbench --dataset_name=antmaze-large-explore-v0 \
                            --hidden_dims=256,256,256 --train_steps=50000 --batch_size=128 --discount=0.995 \
                            --num_skip_states=50 --run_group=rws_antmaze-large-explore-v0 --viz_interval=10000 \
                            --save_interval=250000 

python main_reachability.py --agent_type=rws --dataset_type=maze     --maze_buffer=env/A_star_buffer.pkl\
                            --hidden_dims=256,256,256 --train_steps=50000 --batch_size=128 --discount=0.995\
                            --num_skip_states=50 --run_group=rws_pointmaze-large-stitch-v0 --viz_interval=10000\
                             --save_interval=250000 


python main_reachability.py --agent_type=rws --dataset_type=maze     --maze_buffer=env/A_star_buffer.pkl                            --hidden_dims=256,256,256 --train_steps=50000 --batch_size=128 --discount=0.7                            --num_skip_states=50 --run_group=rws_pointmaze-large-stitch-v0 --viz_interval=10000                             --save_interval=250000 


python main.py   --eval_episodes=50   --agent=agents/gciql.py   --agent.alpha=0.003   --agent.load_rws_path=exp/ReachabilityEstimation/PointMaze_RWS_rws/sd042_20251027_174648   --agent.load_rws_epoch=50000
