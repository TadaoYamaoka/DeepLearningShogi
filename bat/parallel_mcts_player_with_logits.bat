@echo off
call activate DeepLearningShogi
python -m dlshogi.usi.usi_parallel_mcts_player_with_logits 2>NUL
