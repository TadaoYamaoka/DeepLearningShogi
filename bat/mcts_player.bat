@echo off
call activate DeepLearningShogi
python -m dlshogi.usi.usi_mcts_player 2>NUL
